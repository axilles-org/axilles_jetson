"""Export a trained checkpoint to a self-contained deployment module.

``DeployModule`` bakes in both scalers so the robot feeds raw (physical-unit)
sensor windows and receives torque in N·m/kg directly:

    x_raw (1, C, T)  ->  z-score with scaler_x
                     ->  TCN
                     ->  last timestep
                     ->  inverse z-score with scaler_y
                     ->  divide by subject mass
                     ->  torque (N·m/kg)

The result is TorchScript-traced (``best.ts``) and, optionally, ONNX-exported for
a TensorRT build on the Jetson.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..config import Config
from ..data.scalers import ScalerBundle
from ..models import TCN
from ..training.subject_embedding import SubjectIndex


@dataclass
class ExportArtifacts:
    torchscript: Path
    onnx: Path | None
    metadata: Path


class DeployModule(nn.Module):
    """TCN wrapped with input/output normalisation and mass division."""

    def __init__(
        self,
        model: TCN,
        x_mean: np.ndarray,
        x_scale: np.ndarray,
        y_mean: float,
        y_scale: float,
        subject_mass_kg: float,
        subject_idx: int,
        use_subject_embedding: bool,
    ):
        super().__init__()
        self.model = model
        self.register_buffer("x_mean", torch.tensor(x_mean, dtype=torch.float32).view(1, -1, 1))
        self.register_buffer("x_scale", torch.tensor(x_scale, dtype=torch.float32).view(1, -1, 1))
        self.register_buffer("y_mean", torch.tensor(y_mean, dtype=torch.float32))
        self.register_buffer("y_scale", torch.tensor(y_scale, dtype=torch.float32))
        self.register_buffer("subject_mass", torch.tensor(subject_mass_kg, dtype=torch.float32))
        self.register_buffer("subject_idx", torch.tensor([subject_idx], dtype=torch.long))
        self.use_subject_embedding = use_subject_embedding

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        x = (x_raw - self.x_mean) / self.x_scale
        sidx = self.subject_idx if self.use_subject_embedding else None
        out = self.model(x, sidx)                 # (1, 1, T)
        last = out[0, :, -1]                      # (1,)
        moment_nm = last * self.y_scale + self.y_mean
        return moment_nm / self.subject_mass      # N·m/kg


def build_deploy_module(
    run_dir: str | Path,
    subject_mass_kg: float,
    subject_demo: tuple[float, float, str] | None = None,
    checkpoint: str = "best.pt",
) -> tuple[DeployModule, dict]:
    """Load a run and return a ready-to-trace :class:`DeployModule` plus metadata.

    ``subject_demo`` is ``(height_m, weight_kg, gender)`` for the deployment
    subject; when the model uses a subject embedding it selects the nearest
    training subject's embedding row.
    """
    run_dir = Path(run_dir)
    cfg = Config.load(run_dir / "config.yaml")
    scalers = ScalerBundle.load(run_dir)
    feature_names = cfg.data.feature_names()

    with open(run_dir / "model_meta.json") as f:
        meta = json.load(f)
    num_subjects = meta["num_training_subjects"]

    subject_idx = 0
    if cfg.model.use_subject_embedding:
        if subject_demo is None:
            raise ValueError("subject_demo is required for a subject-embedding model")
        idx = SubjectIndex(mapping=dict(meta["subject_index"]),
                           num_training_subjects=num_subjects)
        vec = scalers.demographics.vector(*subject_demo)
        # nearest training row by cosine over the demographic vector
        train_vecs = _training_vectors(cfg, scalers)
        subject_idx = idx.mapping[max(train_vecs, key=lambda s: _cos(vec, train_vecs[s]))]

    model = TCN.from_config(cfg.model, in_channels=len(feature_names),
                            out_channels=1, num_subjects=num_subjects)
    model.load_state_dict(torch.load(run_dir / checkpoint, map_location="cpu"))
    model.eval()

    x_mean, x_scale = scalers.subset_x(feature_names)
    deploy = DeployModule(
        model=model,
        x_mean=x_mean, x_scale=x_scale,
        y_mean=float(scalers.scaler_y.mean_[0]),
        y_scale=float(scalers.scaler_y.scale_[0]),
        subject_mass_kg=subject_mass_kg,
        subject_idx=subject_idx,
        use_subject_embedding=cfg.model.use_subject_embedding,
    ).eval()

    info = {
        "feature_names": feature_names,
        "window_length": cfg.data.window_length,
        "num_input_channels": len(feature_names),
        "output_units": "Nm/kg",
        "subject_mass_kg": subject_mass_kg,
        "subject_idx": subject_idx,
        "applies_input_scaler": True,
        "applies_output_scaler": True,
        "control_rate_hz": cfg.deploy.control_rate_hz,
    }
    return deploy, info


def export(
    run_dir: str | Path,
    subject_mass_kg: float,
    subject_demo: tuple[float, float, str] | None = None,
    onnx: bool = False,
    checkpoint: str = "best.pt",
) -> ExportArtifacts:
    run_dir = Path(run_dir)
    deploy, info = build_deploy_module(run_dir, subject_mass_kg, subject_demo, checkpoint)

    dummy = torch.zeros(1, info["num_input_channels"], info["window_length"])
    with torch.no_grad():
        eager = deploy(dummy)
        traced = torch.jit.trace(deploy, dummy)
        traced_out = traced(dummy)
    max_err = (eager - traced_out).abs().max().item()
    if max_err > 1e-5:
        raise RuntimeError(f"TorchScript trace mismatch: {max_err:.2e}")

    ts_path = run_dir / "best.ts"
    traced.save(str(ts_path))

    onnx_path = None
    onnx_max_err = None
    if onnx:
        onnx_path = run_dir / "best.onnx"
        torch.onnx.export(
            deploy, dummy, str(onnx_path),
            input_names=["window"], output_names=["torque_nm_per_kg"],
            opset_version=17, dynamo=False,
        )
        onnx_max_err = _verify_onnx(onnx_path, deploy, info)
        if onnx_max_err > 1e-4:
            raise RuntimeError(f"ONNX output mismatch: {onnx_max_err:.2e}")

    meta_path = run_dir / "deploy_metadata.json"
    with open(meta_path, "w") as f:
        json.dump({**info, "trace_max_error": max_err,
                   "onnx_max_error": onnx_max_err}, f, indent=2)

    return ExportArtifacts(torchscript=ts_path, onnx=onnx_path, metadata=meta_path)


def _verify_onnx(onnx_path: Path, deploy: DeployModule, info: dict) -> float:
    import numpy as np
    import onnxruntime as ort

    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, info["num_input_channels"], info["window_length"]), dtype=np.float32)
    with torch.no_grad():
        eager = deploy(torch.from_numpy(x)).cpu().numpy()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"window": x})[0]
    return float(np.abs(eager - got).max())


def _training_vectors(cfg: Config, scalers: ScalerBundle) -> dict[str, np.ndarray]:
    import pandas as pd
    df = pd.read_parquet(cfg.paths.demographics()).drop_duplicates("subject")
    df = df[df["subject"].isin(cfg.data.split.train)]
    return {r["subject"]: scalers.demographics.vector(r["height_m"], r["weight_kg"], r["gender"])
            for _, r in df.iterrows()}


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
