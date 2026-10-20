"""Compare two trained models' torque predictions on the held-out GaTech test subjects.

Both models predict every test trial (streaming: last step of each window, as deployed);
predictions and the measured moment are split into strides and scored with
waveform_metrics: each model vs the measured moment, and the two models vs each other.

    python scripts/compare_models.py \
      --model "Original GaTech=runs/ablation/<id>/original" \
      --model "Transformed (tx_all)=runs/ablation/<id>/tx_all" \
      --out runs/report/model_comparison
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

from waveform_metrics import compute_all

from exo.config import Config, replace
from exo.data.gait_cycles import N_PHASE, heel_strikes_from_stance, stride_curves
from exo.training.evaluate import Evaluator

FS = 100
TRUTH = "Measured (inverse dynamics)"
MOMENT_LABEL = "Ankle moment (N·m/kg)\n← plantarflexion"
METRIC_KEYS = ("rmse", "nrmse", "r2", "pearson_r", "peak_value_error", "peak_timing_error_pct",
               "rom_error", "dtw_distance", "cmc", "bland_altman_bias", "bland_altman_loa", "icc")


def predict_test(run_dir: Path) -> tuple[dict, dict[str, dict]]:
    """Window stride 1, so the streaming prediction exists at every sample, as when deployed."""
    cfg = replace(Config.load(run_dir / "config.yaml"), **{"data.stride": 1})
    report, trials = Evaluator(cfg, str(run_dir / "best.pt")).run(split="test")
    out = {}
    for t in trials:
        d = np.load(cfg.paths.processed() / f"{t.name}.npz", allow_pickle=True)
        stance = d["features"][:, list(d["feature_columns"]).index("stance")]
        truth = np.where(t.valid, t.truth, np.nan)
        out[t.name] = {"subject": t.subject, "pred": t.pred_streaming, "truth": truth,
                       "stance": stance[:len(truth)]}
    return report.overall_streaming, out


def strides_by_subject(signals: dict[str, dict[str, dict]]) -> dict[str, dict[str, np.ndarray]]:
    """{subject: {series name: (n_strides, N_PHASE)}}; strides kept only where every series is valid."""
    names = list(signals)
    first = signals[names[0]]
    out: dict[str, dict[str, list]] = {}
    for trial, ref in first.items():
        n = min(len(signals[m][trial]["pred"]) for m in names)
        stack = np.stack([ref["truth"][:n]] + [signals[m][trial]["pred"][:n] for m in names], 1)
        curves = stride_curves(stack, heel_strikes_from_stance(ref["stance"][:n]), FS)
        subj = out.setdefault(ref["subject"], {k: [] for k in [TRUTH, *names]})
        for i, key in enumerate([TRUTH, *names]):
            subj[key].append(curves[..., i])
    return {s: {k: np.concatenate(v) for k, v in d.items()} for s, d in out.items()}


def score(a: np.ndarray, b: np.ndarray) -> dict:
    m = compute_all(a.mean(0), b.mean(0), exo_strides=a, gatech_strides=b)
    return {k: getattr(m, k) for k in METRIC_KEYS}


def continuous_agreement(signals: dict[str, dict[str, dict]]) -> dict:
    (na, a), (nb, b) = signals.items()
    pa = np.concatenate([a[t]["pred"] for t in a])
    pb = np.concatenate([b[t]["pred"][:len(a[t]["pred"])] for t in a])
    ok = ~(np.isnan(pa) | np.isnan(pb))
    return {"pair": f"{nb} vs {na}", "pearson_r": float(np.corrcoef(pa[ok], pb[ok])[0, 1]),
            "rmse_nm_per_kg": float(np.sqrt(np.mean((pa[ok] - pb[ok]) ** 2)))}


def plot_strides(strides: dict, names: list[str], path: Path) -> None:
    subjects = sorted(strides)
    cols = 3
    rows = -(-len(subjects) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows), sharey=True, squeeze=False)
    pct = np.linspace(0, 100, N_PHASE)
    colors = dict(zip(names, ("tab:orange", "tab:green", "tab:blue", "tab:red")))
    for ax, subj in zip(axes.flat, subjects):
        s = strides[subj]
        m, sd = s[TRUTH].mean(0), s[TRUTH].std(0)
        ax.fill_between(pct, m - sd, m + sd, color="0.85")
        ax.plot(pct, m, color="0.2", lw=2.2, ls="--", label=f"{TRUTH}, mean ± 1 SD")
        for name in names:
            ax.plot(pct, s[name].mean(0), color=colors[name], lw=2, label=name)
        ax.axhline(0, color="0.6", lw=0.8)
        ax.set(title=f"{subj} ({len(s[TRUTH])} strides)", xlabel="Gait cycle (%)", xlim=(0, 100))
    for ax in axes[:, 0]:
        ax.set_ylabel(MOMENT_LABEL)
    for ax in axes.flat[len(subjects):]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
    fig.suptitle("Held-out GaTech test subjects: stride-averaged ankle moment", fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=140)


def plot_trial(signals: dict, trial: str, path: Path, seconds: float = 10.0) -> None:
    names = list(signals)
    ref = signals[names[0]][trial]
    t = np.arange(len(ref["truth"])) / FS
    start = t[np.flatnonzero(~np.isnan(ref["pred"]))[0]] + 5
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(t, 0, 1, where=ref["stance"] > 0, color="0.93", step="mid",
                    transform=ax.get_xaxis_transform(), label="Stance")
    ax.plot(t, ref["truth"], color="0.2", lw=2.2, ls="--", label=TRUTH)
    for name, color in zip(names, ("tab:orange", "tab:green")):
        ax.plot(t, signals[name][trial]["pred"], color=color, lw=1.6, label=name)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set(xlim=(start, start + seconds), xlabel="Time (s)", ylabel=MOMENT_LABEL,
           title=f"Test trial {trial}: measured vs predicted ankle moment")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=140)


def plot_peaks(strides: dict, names: list[str], path: Path) -> None:
    peaks = {k: np.concatenate([s[k].min(1) for s in strides.values()]) for k in [TRUTH, *names]}
    pairs = [(names[0], TRUTH), (names[1], TRUTH), (names[1], names[0])]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, (y, x) in zip(axes, pairs):
        ax.scatter(peaks[x], peaks[y], s=6, alpha=0.35)
        lim = (min(peaks[x].min(), peaks[y].min()) - 0.1, max(peaks[x].max(), peaks[y].max()) + 0.1)
        ax.plot(lim, lim, color="0.3", lw=1, ls="--", label="y = x")
        r = np.corrcoef(peaks[x], peaks[y])[0, 1]
        ax.set(xlim=lim, ylim=lim, xlabel=f"{x}\npeak plantarflexion (N·m/kg)",
               ylabel=f"{y}\npeak (N·m/kg)", title=f"Per-stride peak, r = {r:.3f}")
        ax.set_aspect("equal")
        ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=140)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, metavar="NAME=RUN_DIR")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    if len(a.model) != 2:
        raise SystemExit("give exactly two --model")

    models = dict(spec.split("=", 1) for spec in a.model)
    names = list(models)
    overall, signals = {}, {}
    for name, run_dir in models.items():
        overall[name], signals[name] = predict_test(Path(run_dir))

    strides = strides_by_subject(signals)
    comparisons = [(names[0], TRUTH), (names[1], TRUTH), (names[1], names[0])]
    rows = [{"subject": subj, "comparison": f"{x} vs {y}", "strides": len(s[TRUTH]), **score(s[x], s[y])}
            for subj, s in sorted(strides.items()) for x, y in comparisons]
    per_subject = pd.DataFrame(rows)

    a.out.mkdir(parents=True, exist_ok=True)
    per_subject.to_csv(a.out / "per_subject.csv", index=False)
    numeric = per_subject.drop(columns=["subject", "bland_altman_loa"])
    summary = {
        "models": models,
        "test_subjects": sorted(strides),
        "overall_streaming": overall,
        "continuous_agreement": continuous_agreement(signals),
        "stride_metrics_median_over_subjects": json.loads(
            numeric.groupby("comparison").median().to_json(orient="index")),
    }
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2, default=lambda o: asdict(o)))

    plot_strides(strides, names, a.out / "test_strides_per_subject.png")
    plot_trial(signals, sorted(signals[names[0]])[0], a.out / "test_trial_example.png")
    plot_peaks(strides, names, a.out / "per_stride_peak_agreement.png")
    print(json.dumps({k: summary[k] for k in ("overall_streaming", "continuous_agreement")}, indent=2))
    print(numeric.groupby("comparison").median().round(4).T.to_string())


if __name__ == "__main__":
    main()
