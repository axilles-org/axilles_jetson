"""Evaluate a trained checkpoint on a held-out split.

    python scripts/evaluate.py --run runs/<run_dir>
    python scripts/evaluate.py --run runs/<run_dir> --split val --no-plots
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from exo.config import Config
from exo.training.evaluate import Evaluator
from exo.viz.plots import plot_trial_reconstruction


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run directory containing config.yaml and best.pt")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.run)
    cfg = Config.load(run_dir / "config.yaml")
    evaluator = Evaluator(cfg, str(run_dir / args.checkpoint))
    report, results = evaluator.run(split=args.split)

    print(f"\n{'trial':<34} {'RMSE':>8} {'MAE':>8} {'R2':>7}  (windowed, Nm/kg)")
    print("-" * 62)
    for name, m in sorted(report.per_trial.items()):
        w = m["windowed"]
        print(f"{name:<34} {w['rmse_nm_per_kg']:>8.4f} {w['mae_nm_per_kg']:>8.4f} {w['r2']:>7.3f}")

    print(f"\n{'subject':<10} {'RMSE':>8} {'MAE':>8} {'normMAE':>9} {'R2':>7}")
    print("-" * 46)
    for s, m in sorted(report.per_subject.items()):
        print(f"{s:<10} {m['rmse_nm_per_kg']:>8.4f} {m['mae_nm_per_kg']:>8.4f} "
              f"{m['normalized_mae']:>9.4f} {m['r2']:>7.3f}")

    print("\noverall (windowed) :", _fmt(report.overall_windowed))
    print("overall (streaming):", _fmt(report.overall_streaming))

    out_dir = run_dir / f"eval_{args.split}"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "report.json", "w") as f:
        json.dump({"per_trial": report.per_trial, "per_subject": report.per_subject,
                   "overall_windowed": report.overall_windowed,
                   "overall_streaming": report.overall_streaming}, f, indent=2)

    if not args.no_plots:
        for r in results:
            plot_trial_reconstruction(
                r.name, r.pred_windowed, r.truth, r.valid,
                report.per_trial[r.name]["windowed"], out_dir / f"{r.name}.png")
        print(f"plots -> {out_dir}")


def _fmt(m: dict[str, float]) -> str:
    return (f"RMSE={m['rmse_nm_per_kg']:.4f}  MAE={m['mae_nm_per_kg']:.4f}  "
            f"normMAE={m['normalized_mae']:.4f}  R2={m['r2']:.3f}")


if __name__ == "__main__":
    main()
