"""Real-time inference runtime for the exoskeleton control loop.

``ExoController`` ties together:

    raw sensor frame
      -> SensorAdapter        (exo frame -> training convention)
      -> FeaturePipeline      (column ordering, stance passthrough)
      -> ObservationBuffer    (rolling window)
      -> inference backend     (z-score -> TCN -> torque N·m/kg)
      -> AssistanceController (stance gate, scale, ramp, limits)
      -> command torque (N·m)

The backend is TorchScript (``best.ts``), ONNX Runtime (``best.onnx``), or
TensorRT (``best.engine`` on a Jetson), selected by ``DeployConfig.backend``.
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

import numpy as np

from ..config import DeployConfig
from ..data.feature_pipeline import FeaturePipeline
from .assistance import AssistanceController
from .sensor_adapter import SensorAdapter


class InferenceBackend:
    """Wraps a compiled model; ``predict((C, T)) -> torque (N·m/kg)``."""

    def __init__(self, run_dir: Path, backend: str, device: str):
        self.backend = backend
        if backend == "jit":
            import torch
            self._torch = torch
            self.device = torch.device(device if _device_ok(device) else "cpu")
            self.model = torch.jit.load(str(run_dir / "best.ts"), map_location=self.device).eval()
        elif backend == "onnx":
            import onnxruntime as ort
            self.session = ort.InferenceSession(
                str(run_dir / "best.onnx"), providers=["CPUExecutionProvider"])
            self._input = self.session.get_inputs()[0].name
        elif backend == "trt":
            from .tensorrt import TRTRunner
            self.runner = TRTRunner(run_dir / "best.engine")
        else:
            raise ValueError(f"unknown backend {backend!r}; choose jit|onnx|trt")

    def predict(self, window: np.ndarray) -> float:
        x = window[np.newaxis].astype(np.float32)          # (1, C, T)
        if self.backend == "jit":
            t = self._torch.from_numpy(x).to(self.device)
            with self._torch.inference_mode():
                return float(self.model(t).reshape(-1)[0])
        if self.backend == "onnx":
            return float(self.session.run(None, {self._input: x})[0].reshape(-1)[0])
        return float(self.runner.predict(x).reshape(-1)[0])

    def warmup(self, num_channels: int, window_length: int) -> None:
        self.predict(np.zeros((num_channels, window_length), dtype=np.float32))


def _device_ok(device: str) -> bool:
    import torch
    if device == "cuda":
        return torch.cuda.is_available()
    if device == "mps":
        return torch.backends.mps.is_available()
    return True


class ObservationBuffer:
    """Fixed-length rolling buffer of feature frames."""

    def __init__(self, window_length: int, num_channels: int):
        self.window_length = window_length
        self.num_channels = num_channels
        self._buf: collections.deque = collections.deque(maxlen=window_length)

    def push(self, frame: np.ndarray) -> None:
        if frame.shape[0] != self.num_channels:
            raise ValueError(f"expected {self.num_channels} channels, got {frame.shape[0]}")
        self._buf.append(frame.astype(np.float32))

    @property
    def ready(self) -> bool:
        return len(self._buf) == self.window_length

    def window(self) -> np.ndarray:
        """``(C, T)`` array in model-input layout."""
        return np.stack(self._buf, axis=1)

    def reset(self) -> None:
        self._buf.clear()


class ExoController:
    def __init__(self, run_dir: str | Path, deploy_cfg: DeployConfig,
                 subject_mass_kg: float):
        run_dir = Path(run_dir)
        with open(run_dir / "deploy_metadata.json") as f:
            self.meta = json.load(f)

        self.feature_names: list[str] = self.meta["feature_names"]
        self.pipeline = FeaturePipeline(self.feature_names)
        self.adapter = SensorAdapter(deploy_cfg.sensor_adapter)
        self.assistance = AssistanceController(deploy_cfg)
        self.subject_mass_kg = subject_mass_kg

        self.backend = InferenceBackend(run_dir, deploy_cfg.backend, deploy_cfg.device)
        self.buffer = ObservationBuffer(self.meta["window_length"],
                                        self.meta["num_input_channels"])
        self.backend.warmup(self.meta["num_input_channels"], self.meta["window_length"])

    def reset(self) -> None:
        self.buffer.reset()
        self.assistance.reset()
        self.adapter.reset()

    def step(self, raw_frame: dict[str, float], dt: float) -> dict[str, float]:
        """Process one sensor frame; returns the command and diagnostics."""
        adapted = self.adapter.transform_frame(raw_frame, dt)
        feat = self.pipeline.transform(_single_row_df(adapted, self.feature_names))
        self.buffer.push(feat[0])

        predicted = 0.0
        if self.buffer.ready:
            predicted = self.backend.predict(self.buffer.window())

        cmd = self.assistance.update(
            predicted, self.subject_mass_kg,
            raw_frame.get("heel_fsr_raw", 0.0), raw_frame.get("toe_fsr_raw", 0.0), dt,
        )
        return {
            "command_nm": cmd.torque_nm,
            "predicted_nm_per_kg": predicted,
            "stance": float(cmd.stance),
            "ramp": cmd.ramp,
            "buffer_ready": float(self.buffer.ready),
        }


def _single_row_df(values: dict[str, float], columns: list[str]):
    import pandas as pd
    return pd.DataFrame([[values[c] for c in columns]], columns=columns)
