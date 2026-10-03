"""Device selection and mixed-precision / compile helpers."""
from __future__ import annotations

from contextlib import nullcontext

import torch

from ..config import PerfConfig


def select_device(preference: str = "auto") -> torch.device:
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_amp_dtype(mode: str, device: torch.device) -> torch.dtype | None:
    """None disables autocast. bf16 is preferred on CUDA Ampere+; fp16 otherwise."""
    if mode == "off" or device.type == "cpu":
        return None
    if mode == "bf16":
        return torch.bfloat16
    if mode == "fp16":
        return torch.float16
    # auto
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if device.type == "cuda":
        return torch.float16
    return None  # MPS autocast is unreliable; keep fp32


def apply_backend_flags(perf: PerfConfig, device: torch.device) -> None:
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = perf.tf32
        torch.backends.cudnn.allow_tf32 = perf.tf32
        torch.backends.cudnn.benchmark = perf.cudnn_benchmark


def maybe_compile(model: torch.nn.Module, perf: PerfConfig, device: torch.device) -> torch.nn.Module:
    if not perf.compile or device.type != "cuda" or not hasattr(torch, "compile"):
        return model
    try:
        return torch.compile(model, mode=perf.compile_mode)
    except Exception as exc:  # noqa: BLE001 - compilation is best-effort
        print(f"[perf] torch.compile disabled: {exc}")
        return model


def autocast_context(dtype: torch.dtype | None, device: torch.device):
    if dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)
