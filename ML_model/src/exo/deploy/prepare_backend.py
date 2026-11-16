"""Prepare wearer-specific inference files before opening sensors or motors."""
from pathlib import Path

from ..config import Config
from .export import export


def prepare_backend(run_dir: Path, cfg: Config, mass: float,
                    height: float | None = None, gender: str | None = None) -> None:
    backend = cfg.deploy.backend
    if backend not in ('jit', 'onnx', 'trt'):
        raise ValueError(f'Unsupported inference backend: {backend}')
    if (height is None) != (gender is None):
        raise ValueError('Provide both height and gender for wearer-specific export')
    if height is not None:
        checkpoint = 'deploy.pt' if (run_dir / 'deploy.pt').exists() else 'best.pt'
        export(run_dir, mass, subject_demo=(height, mass, gender),
               onnx=backend in ('onnx', 'trt'), checkpoint=checkpoint)
        print(f'Exported {checkpoint} for {mass} kg, {height} m, {gender}', flush=True)
    if backend == 'trt':
        from .tensorrt import ensure_engine
        ensure_engine(run_dir)
