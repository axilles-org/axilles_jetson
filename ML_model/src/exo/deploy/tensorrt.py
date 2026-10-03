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

import numpy as np


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
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        self.context = self.engine.create_execution_context()

        self._in_name = self.engine.get_tensor_name(0)
        self._out_name = self.engine.get_tensor_name(1)
        self._in_shape = tuple(self.engine.get_tensor_shape(self._in_name))
        self._out_shape = tuple(self.engine.get_tensor_shape(self._out_name))

        self._d_in = cuda.mem_alloc(int(np.prod(self._in_shape)) * 4)
        self._d_out = cuda.mem_alloc(int(np.prod(self._out_shape)) * 4)
        self._stream = cuda.Stream()
        self._h_out = cuda.pagelocked_empty(int(np.prod(self._out_shape)), np.float32)

    def predict(self, window: np.ndarray) -> np.ndarray:
        """window: (C, T) or (1, C, T) float32 -> torque (Nm/kg)."""
        x = np.ascontiguousarray(window.reshape(self._in_shape), dtype=np.float32)
        self._cuda.memcpy_htod_async(self._d_in, x, self._stream)
        self.context.set_tensor_address(self._in_name, int(self._d_in))
        self.context.set_tensor_address(self._out_name, int(self._d_out))
        self.context.execute_async_v3(self._stream.handle)
        self._cuda.memcpy_dtoh_async(self._h_out, self._d_out, self._stream)
        self._stream.synchronize()
        return self._h_out.copy()
