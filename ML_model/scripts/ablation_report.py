"""Score trained ablation variants on the test split: streaming R² and RMSE, overall and per subject.

    python scripts/ablation_report.py --out runs/report/4_ablation \
        runs/ablation/<id>/original runs/ablation/<id>/tx_all runs/ablation/<id>/tx_all_drop_gyro ...

The run directory name is the variant name. Streaming = the last step of each test window
(only past data), as deployed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from exo.config import Config
from exo.training.evaluate import Evaluator

LABELS = {
    "original": "Original data, all inputs (reference)",
    "tx_all": "Transformed, all inputs",
    "tx_all_drop_foot_accel": "Without foot accel",
    "tx_all_drop_shank_accel": "Without shank accel",
    "tx_all_drop_accel": "Without accel (foot + shank)",
    "tx_all_drop_foot_gyro": "Without foot gyro",
    "tx_all_drop_shank_gyro": "Without shank gyro",
    "tx_all_drop_gyro": "Without gyro (foot + shank)",
    "tx_all_drop_stance": "Without stance flag",
    "tx_all_drop_ankle": "Without ankle angle",
    "imu_only": "IMU only (no stance, no ankle angle)",
    "no_imu": "No IMU (stance + ankle angle only)",
}


def r2(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(1.0 - np.sum((pred - truth) ** 2) / np.sum((truth - truth.mean()) ** 2))


def score(run_dir: Path) -> list[dict]:
    cfg = Config.load(run_dir / "config.yaml")
    _, trials = Evaluator(cfg, str(run_dir / "best.pt")).run(split="test")
    pairs: dict[str, list] = {}
    for t in trials:
        ok = ~np.isnan(t.pred_streaming) & t.valid
        pairs.setdefault(t.subject, []).append((t.pred_streaming[ok], t.truth[ok]))
    pairs["all"] = [p for subj in list(pairs.values()) for p in subj]
    rows = []
    for subject, parts in pairs.items():
        pred = np.concatenate([p for p, _ in parts])
        truth = np.concatenate([t for _, t in parts])
        rows.append({"variant": run_dir.name, "subject": subject, "inputs": len(cfg.data.feature_names()),
                     "r2": r2(pred, truth), "rmse_nm_per_kg": float(np.sqrt(np.mean((pred - truth) ** 2)))})
    return rows


def plot(df: pd.DataFrame, path: Path) -> None:
    overall = df[df.subject == "all"].set_index("variant").sort_values("r2", ascending=False)
    subjects = df[df.subject != "all"]
    fig, ax = plt.subplots(figsize=(11, 0.55 * len(overall) + 1.8))
    y = np.arange(len(overall))
    ref = overall.loc["tx_all", "r2"] if "tx_all" in overall.index else None
    colors = ["0.6" if v == "original" else "tab:green" if v == "tx_all" else "tab:blue" for v in overall.index]
    ax.barh(y, overall.r2, color=colors, alpha=0.85, label="All 6 test subjects")
    for i, v in enumerate(overall.index):
        s = subjects[subjects.variant == v]
        ax.scatter(s.r2, np.full(len(s), i), color="k", s=14, zorder=3,
                   label="Individual test subject" if i == 0 else None)
        ax.text(1.002, i, f"{overall.loc[v, 'r2']:.3f}  (RMSE {overall.loc[v, 'rmse_nm_per_kg']:.3f})",
                va="center", fontsize=9)
    if ref is not None:
        ax.axvline(ref, color="tab:green", ls="--", lw=1)
    ax.set_yticks(y, [LABELS.get(v, v) for v in overall.index])
    ax.invert_yaxis()
    ax.set_xlim(0.75, 1.0)
    ax.set_xlabel("Test R² (streaming prediction vs measured ankle moment)")
    ax.set_title("Held-out GaTech test subjects (n = 6): R² by input set\n"
                 "inputs in the exo frame (Amily's transform) unless noted; RMSE in N·m/kg",
                 fontweight="bold")
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=2, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 0.8, 1))
    fig.savefig(path, dpi=140)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    df = pd.DataFrame([row for run in a.runs for row in score(run)])
    a.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out / "r2_by_variant_and_subject.csv", index=False)
    plot(df, a.out / "r2_by_variant.png")
    table = df.pivot(index="variant", columns="subject", values="r2")
    table = table[["all"] + sorted(c for c in table.columns if c != "all")].sort_values("all", ascending=False)
    pd.set_option("display.width", 200)
    print(table.round(3).to_string())


if __name__ == "__main__":
    main()
