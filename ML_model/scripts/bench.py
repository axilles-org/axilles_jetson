"""Benchmark training throughput and inference latency.

    python scripts/bench.py --train        # samples/s: eager vs amp vs compiled
    python scripts/bench.py --infer --run runs/<run_dir>   # per-window latency
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from exo.config import Config, replace
from exo.models import TCN
from exo.training.runtime_utils import (
    autocast_context,
    maybe_compile,
    resolve_amp_dtype,
    select_device,
)


def bench_train(cfg: Config, iters: int = 60) -> None:
    device = select_device(cfg.train.device)
    x = torch.randn(cfg.train.batch_size, cfg.data.num_features, cfg.data.window_length, device=device)
    y = torch.randn(cfg.train.batch_size, 1, cfg.data.window_length, device=device)
    sidx = torch.zeros(cfg.train.batch_size, dtype=torch.long, device=device)

    def make_model() -> TCN:
        return TCN.from_config(cfg.model, in_channels=cfg.data.num_features,
                               out_channels=1, num_subjects=13).to(device)

    configs = [("eager", "off", False)]
    if device.type == "cuda":
        configs += [("amp", "auto", False), ("amp+compile", "auto", True)]
    elif device.type == "mps":
        configs += [("amp", "auto", False)]

    for label, amp_mode, compile_ in configs:
        model = make_model()
        model = maybe_compile(model, replace(cfg, **{"train.perf.compile": compile_}).train.perf, device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        dtype = resolve_amp_dtype(amp_mode, device)

        for _ in range(10):  # warmup
            _step(model, opt, x, y, sidx, dtype, device)
        _sync(device)
        t0 = time.perf_counter()
        for _ in range(iters):
            _step(model, opt, x, y, sidx, dtype, device)
        _sync(device)
        dt = time.perf_counter() - t0
        print(f"{label:14s}  {iters * cfg.train.batch_size / dt:8.0f} samples/s  "
              f"({1000 * dt / iters:.1f} ms/step)")


def bench_infer(run_dir: str, backends: list[str], iters: int = 500) -> None:
    import json

    import numpy as np

    from exo.deploy.runtime import InferenceBackend

    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "deploy_metadata.json").read_text())
    c, t = meta["num_input_channels"], meta["window_length"]
    window = np.zeros((c, t), dtype=np.float32)
    budget = 1000.0 / meta["control_rate_hz"]

    for backend in backends:
        try:
            runner = InferenceBackend(run_dir, backend, "cpu")
        except (FileNotFoundError, ImportError, ValueError) as exc:
            print(f"{backend:6s}  skipped ({exc})")
            continue
        for _ in range(20):
            runner.predict(window)
        samples = []
        for _ in range(iters):
            t0 = time.perf_counter()
            runner.predict(window)
            samples.append((time.perf_counter() - t0) * 1000.0)
        samples.sort()
        p99 = samples[int(0.99 * iters)]
        flag = "OK" if p99 < budget else "OVER BUDGET"
        print(f"{backend:6s}  p50={statistics.median(samples):.3f} ms  "
              f"p99={p99:.3f} ms  budget={budget:.1f} ms  {flag}")


def _step(model, opt, x, y, sidx, dtype, device):
    opt.zero_grad(set_to_none=True)
    with autocast_context(dtype, device):
        loss = torch.nn.functional.mse_loss(model(x, sidx), y)
    loss.backward()
    opt.step()


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--infer", action="store_true")
    ap.add_argument("--run", default=None)
    ap.add_argument("--backends", nargs="+", default=["jit", "onnx", "trt"])
    args = ap.parse_args()

    cfg = Config.load(args.config)
    if args.train:
        bench_train(cfg)
    if args.infer:
        if not args.run:
            raise SystemExit("--infer requires --run")
        bench_infer(args.run, args.backends)


if __name__ == "__main__":
    main()
