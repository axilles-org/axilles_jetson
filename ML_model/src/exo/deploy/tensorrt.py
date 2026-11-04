"""TensorRT engine build and runtime for Jetson deployment.

Not exercised on the development machine — ``tensorrt`` and ``pycuda`` ship only
in the NVIDIA Jetson container. The engine is built once on-device from the ONNX
export::

    python scripts/export_jit.py --run <run_dir> --mass <kg> --onnx
    # on the Jetson:
    python -c "from exo.deploy.tensorrt import build_engine; \\
               build_engine('<run_dir>/best.onnx', '<run_dir>/best.engine', fp16=True)"

``TRTRunner`` mirrors ``JITRunner``'s ``predict(window) -> torque`` interface so
``ExoController`` is backend-agnostic.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import platform
import tempfile

import numpy as np


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _device_fingerprint() -> dict:
    import pycuda.autoinit  # noqa: F401
    import pycuda.driver as cuda
    import tensorrt as trt
    dev = cuda.Context.get_device()
    return dict(tensorrt=trt.__version__, cuda_driver=cuda.get_driver_version(),
                gpu=dev.name(), compute_capability=list(dev.compute_capability()),
                platform=platform.platform())


def _verify_engine(engine_path: Path, onnx_path: Path) -> float:
    """Numerical check before an engine can enter the motor-control path."""
    import onnxruntime as ort
    reference = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    runner = TRTRunner(engine_path)
    shape = tuple(reference.get_inputs()[0].shape)
    rng = np.random.default_rng(0)
    error = 0.0
    for scale in (0.0, 1.0, 10.0):
        x = (rng.standard_normal(shape) * scale).astype(np.float32)
        expected = reference.run(None, {reference.get_inputs()[0].name: x})[0].reshape(-1)
        actual = runner.predict(x).reshape(-1)
        if not np.isfinite(actual).all() or not np.allclose(actual, expected, atol=1e-4, rtol=1e-4):
            raise RuntimeError('TensorRT output does not match ONNX; engine was not activated')
        error = max(error, float(np.max(np.abs(actual - expected))))
    return error


def ensure_engine(run_dir: str | Path) -> Path:
    """Build/cache a local FP32 engine for the exact ONNX export and device.

    Wearer-specific export changes the ONNX hash, invalidating the old engine.
    Called before ExoController and before motor construction/arming.
    """
    run_dir = Path(run_dir)
    onnx = run_dir / 'best.onnx'
    engine = run_dir / 'best.engine'
    metadata = run_dir / 'best.engine.json'
    if not onnx.is_file():
        raise FileNotFoundError(f'TensorRT requires an ONNX export: {onnx}')
    try:
        identity = dict(onnx_sha256=_sha256(onnx), device=_device_fingerprint(),
                        precision='fp32', validation_version=1)
    except ImportError as exc:
        raise RuntimeError('TensorRT GPU inference requires TensorRT and PyCUDA on the Jetson') from exc
    if engine.is_file() and metadata.is_file():
        try:
            cached = json.loads(metadata.read_text())
            if (cached.get('identity') == identity and
                    cached.get('engine_sha256') == _sha256(engine)):
                print('TensorRT: using verified engine for this model and device', flush=True)
                return engine
        except (ValueError, OSError):
            pass
    print('TensorRT: building and checking GPU engine before starting controller...', flush=True)
    with tempfile.TemporaryDirectory(dir=run_dir, prefix='.trt-build-') as temporary:
        candidate = Path(temporary) / 'best.engine'
        build_engine(onnx, candidate, fp16=False)
        error = _verify_engine(candidate, onnx)
        meta = dict(identity=identity, engine_sha256=_sha256(candidate), onnx_max_error=error)
        candidate_meta = Path(temporary) / 'best.engine.json'
        candidate_meta.write_text(json.dumps(meta, indent=2) + '\n')
        candidate.replace(engine)
        candidate_meta.replace(metadata)
    return engine


def build_engine(onnx_path: str | Path, engine_path: str | Path, fp16: bool = True,
                 workspace_mb: int = 512) -> Path:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise RuntimeError(f"failed to parse {onnx_path}:\n{errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb << 20)
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed")
    Path(engine_path).write_bytes(serialized)
    return Path(engine_path)


class TRTRunner:
    """Single-window inference over a serialized TensorRT engine."""

    def __init__(self, engine_path: str | Path):
        import pycuda.autoinit  # noqa: F401  (initialises the CUDA context)
        import pycuda.driver as cuda
        import tensorrt as trt

        self._cuda = cuda
        logger = trt.Logger(trt.Logger.WARNING)
        self._logger = logger
        self._runtime = trt.Runtime(logger)
        self.engine = self._runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        if self.engine is None:
            raise RuntimeError(f'Could not load TensorRT engine: {engine_path}')
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError('Could not create TensorRT execution context')
        if not hasattr(self.engine, 'num_io_tensors') or not hasattr(self.context, 'execute_async_v3'):
            raise RuntimeError('This backend requires TensorRT with named I/O and execute_async_v3 (8.5+)')
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        inputs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        outputs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError('Expected one TensorRT input and one output')
        self._in_name, self._out_name = inputs[0], outputs[0]
        self._in_shape = tuple(self.engine.get_tensor_shape(self._in_name))
        self._out_shape = tuple(self.engine.get_tensor_shape(self._out_name))
        if any(d <= 0 for d in (*self._in_shape, *self._out_shape)):
            raise ValueError('Expected static positive TensorRT tensor shapes')
        if any(np.dtype(trt.nptype(self.engine.get_tensor_dtype(n))) != np.dtype(np.float32)
               for n in (self._in_name, self._out_name)):
            raise ValueError('Expected FP32 TensorRT input/output tensors')
        self._d_in = cuda.mem_alloc(int(np.prod(self._in_shape)) * 4)
        self._d_out = cuda.mem_alloc(int(np.prod(self._out_shape)) * 4)
        self._stream = cuda.Stream()
        self._h_in = cuda.pagelocked_empty(int(np.prod(self._in_shape)), np.float32)
        self._h_out = cuda.pagelocked_empty(int(np.prod(self._out_shape)), np.float32)

    def predict(self, window: np.ndarray) -> np.ndarray:
        """window: (C, T) or (1, C, T) float32 -> torque (Nm/kg)."""
        x = np.ascontiguousarray(window.reshape(self._in_shape), dtype=np.float32)
        np.copyto(self._h_in, x.ravel())
        self._cuda.memcpy_htod_async(self._d_in, self._h_in, self._stream)
        self.context.set_tensor_address(self._in_name, int(self._d_in))
        self.context.set_tensor_address(self._out_name, int(self._d_out))
        if not self.context.execute_async_v3(self._stream.handle):
            raise RuntimeError('TensorRT execution failed')
        self._cuda.memcpy_dtoh_async(self._h_out, self._d_out, self._stream)
        self._stream.synchronize()
        return self._h_out.copy()
