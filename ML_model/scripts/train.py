"""Train the torque-prediction model.

    python scripts/train.py
    python scripts/train.py --config configs/default.yaml --epochs 1 --no-wandb
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prepare import prepare

from exo.config import Config, replace
from exo.training.trainer import Trainer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--no-fetch", action="store_true", help="skip the HF dataset fetch")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    prepare(cfg, do_fetch=not args.no_fetch, verbose=True)
    overrides: dict = {}
    if args.epochs is not None:
        overrides["train.epochs"] = args.epochs
    if args.run_name:
        overrides["train.wandb_run_name"] = args.run_name
    if overrides:
        cfg = replace(cfg, **overrides)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = cfg.paths.runs() / f"{cfg.train.wandb_run_name}_{stamp}"

    trainer = Trainer(cfg, run_dir, use_wandb=not args.no_wandb)
    best = trainer.fit()
    print(f"done. best checkpoint: {best}")


if __name__ == "__main__":
    main()
