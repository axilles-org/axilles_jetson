"""Review a mock/powered deployment log.

    python scripts/plot_mock_run.py logs/mock_run.csv [--processed <npz dir>]

No torque labels on the exo, so instead of RMSE we check: the prediction is
periodic and gait-locked, near zero in swing and plantarflexion in stance,
physiologically plausible in magnitude, and its cycle average sits in the GaTech
test band. Writes <csv>.review.png. See docs/DEPLOYMENT.md.
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


def _heel_strikes(foot_gyro_sagittal: np.ndarray, rate_hz: float) -> np.ndarray:
    from exo.deploy.adapter_fit import heel_strike_indices
    return heel_strike_indices(foot_gyro_sagittal, rate_hz)


def _cycle_average(sig: np.ndarray, events: np.ndarray, n: int = 100):
    if events.size < 4:
        return None, None
    cyc = []
    for a, b in zip(events[:-1], events[1:]):
        seg = sig[a:b]
        if len(seg) < 5:
            continue
        cyc.append(np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(seg)), seg))
    arr = np.stack(cyc)
    return arr.mean(0), arr.std(0)


def _gatech_moment_band(processed_dir: str, n: int = 100):
    """Gait-cycle-averaged ankle moment (N.m/kg) from processed GaTech trials."""
    import glob
    import os

    from exo.deploy.adapter_fit import heel_strike_indices, phase_average

    files = sorted(glob.glob(os.path.join(processed_dir, "*.npz")))[:40]
    if not files:
        return None, None
    cols = list(np.load(files[0])["feature_columns"])
    gyro_i = cols.index("imu_foot_Gyro_Y")
    curves = []
    for f in files:
        z = np.load(f)
        feats, tgt = z["features"], z["target"].reshape(-1)
        ev = heel_strike_indices(feats[:, gyro_i], 100.0)
        if ev.size < 4:
            continue
        # target is already physical N.m; normalise by a nominal 72 kg for the band
        curves.append(phase_average(tgt / 72.0, ev))
    if not curves:
        return None, None
    arr = np.stack(curves)
    return arr.mean(0), arr.std(0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--rate", type=float, default=100.0)
    ap.add_argument("--processed", default=None,
                    help="GaTech processed dir for the reference band (optional)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df = df[df["buffer_ready"] == 1].reset_index(drop=True)
    if len(df) < 200:
        raise SystemExit(f"only {len(df)} ready frames - record a longer run")

    t = df["t_s"].to_numpy()
    pred = df["predicted_nm_per_kg"].to_numpy()
    stance = df["stance"].to_numpy()
    foot_gy = df["foot_gy"].to_numpy()

    events = _heel_strikes(foot_gy, args.rate)
    pred_avg, pred_std = _cycle_average(pred, events)
    stance_avg, _ = _cycle_average(stance.astype(float), events)

    ref_avg = ref_std = None
    if args.processed:
        ref_avg, ref_std = _gatech_moment_band(args.processed)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # 1 - raw prediction time series with stance shading
    ax = axes[0, 0]
    ax.plot(t, pred, lw=0.8, color="tab:red")
    ax.fill_between(t, pred.min(), pred.max(), where=stance > 0.5, alpha=0.12,
                    color="tab:blue", step="mid", label="stance")
    ax.set_title("Predicted ankle moment (live)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("N.m/kg")
    ax.legend(loc="upper right", fontsize=8)

    # 2 - gait-cycle average of the prediction, vs GaTech band
    ax = axes[0, 1]
    phase = np.linspace(0, 100, 100)
    if ref_avg is not None:
        ax.fill_between(phase, ref_avg - ref_std, ref_avg + ref_std, alpha=0.25,
                        color="tab:blue", label="GaTech test band")
        ax.plot(phase, ref_avg, color="tab:blue", lw=1)
    if pred_avg is not None:
        ax.plot(phase, pred_avg, color="tab:red", lw=2, label="exo, ML (cycle avg)")
        ax.fill_between(phase, pred_avg - pred_std, pred_avg + pred_std, alpha=0.15,
                        color="tab:red")
    ax.set_title(f"Gait-cycle average  ({events.size - 1} cycles)")
    ax.set_xlabel("gait cycle (%)")
    ax.set_ylabel("N.m/kg")
    ax.legend(loc="upper right", fontsize=8)

    # 3 - prediction vs stance flag overlay (cycle average)
    ax = axes[1, 0]
    if pred_avg is not None:
        ax.plot(phase, pred_avg / (np.abs(pred_avg).max() + 1e-9), color="tab:red",
                lw=2, label="prediction (norm.)")
    if stance_avg is not None:
        ax.plot(phase, stance_avg, color="tab:green", lw=2, label="stance flag")
    ax.set_title("Prediction vs stance gate (cycle average)")
    ax.set_xlabel("gait cycle (%)")
    ax.legend(loc="upper right", fontsize=8)

    # 4 - numeric checks
    ax = axes[1, 1]
    ax.axis("off")
    swing_mask = stance < 0.5
    stance_mask = stance > 0.5
    checks = [
        ("ready frames", f"{len(df)}"),
        ("gait cycles", f"{events.size - 1}"),
        ("pred peak (most negative)", f"{pred.min():+.3f} N.m/kg"),
        ("pred mean | swing", f"{pred[swing_mask].mean():+.3f} N.m/kg"),
        ("pred mean | stance", f"{pred[stance_mask].mean():+.3f} N.m/kg"),
        ("|pred| swing / |pred| stance",
         f"{np.abs(pred[swing_mask]).mean() / (np.abs(pred[stance_mask]).mean() + 1e-9):.2f}"),
        ("prediction std / mean |.|",
         f"{pred.std() / (np.abs(pred).mean() + 1e-9):.2f}  (periodicity)"),
    ]
    if pred_avg is not None:
        peak_phase = phase[np.argmin(pred_avg)]
        checks.append(("cycle-avg peak at", f"{peak_phase:.0f}% (expect ~45-60%)"))
    y = 0.95
    ax.text(0.0, 1.0, "sanity checks", fontsize=11, fontweight="bold", va="top")
    for k, v in checks:
        ax.text(0.0, y, f"{k:32s} {v}", fontsize=9, family="monospace", va="top")
        y -= 0.11

    verdict = []
    if pred[swing_mask].mean() > -0.2 and pred[stance_mask].mean() < -0.2:
        verdict.append("OK: near-zero in swing, plantarflexion in stance")
    else:
        verdict.append("CHECK: swing/stance split weak - review FSR thresholds / phase")
    if pred.min() > -2.0 and pred.min() < -0.4:
        verdict.append("OK: peak magnitude physiologically plausible")
    else:
        verdict.append(f"CHECK: peak {pred.min():+.2f} N.m/kg outside ~[-1.5, -0.6]")
    ax.text(0.0, y - 0.03, "\n".join(verdict), fontsize=9, va="top",
            color="darkgreen" if all(s.startswith("OK") for s in verdict) else "darkred")

    fig.suptitle(f"Mock deployment review  —  {Path(args.csv).name}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = args.out or str(Path(args.csv).with_suffix(".review.png"))
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")
    for k, v in checks:
        print(f"  {k:32s} {v}")
    for line in verdict:
        print(f"  {line}")


if __name__ == "__main__":
    main()
