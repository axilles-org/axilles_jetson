"""Compare causal gait baselines and TCNs using both definitions of R².

Literature convention: square Pearson r within each test subject, then report
the unweighted mean and sample SD across subjects. Prediction R² (1-SSE/SST)
and RMSE retain sensitivity to amplitude and bias. No stride averaging, fitted
test-data scaling, phase warping, or retrospective lag correction is applied.

Example (paths relative to ML_model):
    python scripts/audit_gait_baselines.py --config runs/.../no_imu/config.yaml \
        --model 'No IMU=runs/.../no_imu' --model 'Full=runs/.../tx_all' \
        --out runs/report/5_gait_baseline_audit
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from exo.config import Config, replace
from exo.data.demographics import subject_mass
from exo.data.gait_cycles import heel_strikes_from_stance, stride_curves
from exo.data.window_index import _windows_for_trial
from exo.paths import resolve
from exo.training.evaluate import Evaluator


def scores(pred: np.ndarray, truth: np.ndarray) -> dict:
    pred, truth = np.asarray(pred, float), np.asarray(truth, float)
    if pred.shape != truth.shape or not np.isfinite([pred, truth]).all():
        raise ValueError("metrics require matching finite arrays")
    residual = np.sum((pred - truth) ** 2)
    total = np.sum((truth - truth.mean()) ** 2)
    r = float(np.corrcoef(pred, truth)[0, 1]) if pred.std() and truth.std() else float("nan")
    return {"n": len(truth), "pearson_r": r, "r2_corr": r * r,
            "r2_pred": float(1 - residual / total) if total else float("nan"),
            "rmse_nm_per_kg": float(np.sqrt(residual / len(truth)))}


def causal_template(stance: np.ndarray, template: np.ndarray, initial_period: float,
                    fs: float) -> np.ndarray:
    """Phase uses time since the latest observed heel strike and a past period."""
    grid = np.linspace(0, 1, len(template))
    pred = np.full(len(stance), template[0], dtype=float)
    last, period = None, initial_period
    for t in range(len(stance)):
        if t > 0 and stance[t - 1] == 0 and stance[t] == 1:
            if last is not None and 0.6 * fs <= t - last <= 2.2 * fs:
                period = t - last
            last = t
        if last is not None:
            pred[t] = np.interp(min((t - last) / period, 1), grid, template)
    return pred


def load_trials(cfg: Config, subjects: list[str]) -> dict:
    masses = subject_mass(cfg.paths.demographics())
    trials = {}
    for subject in subjects:
        for path in sorted(cfg.paths.processed().glob(f"{subject}_*.npz")):
            with np.load(path, allow_pickle=True) as data:
                cols = list(data["feature_columns"])
                trials[path.stem] = (
                    data["target"][:, 0].astype(float) / masses[subject],
                    data["features"][:, cols.index("stance")],
                    data["features"][:, cols.index("gon_ankle_sagittal")],
                )
    return trials


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--model", action="append", default=[], metavar="NAME=RUN_DIR")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    cfg = Config.load(args.config)
    cfg.data.split.validate()
    fs = cfg.data.ingest.target_rate_hz
    train = load_trials(cfg, cfg.data.split.train)
    test = load_trials(cfg, cfg.data.split.test)
    y, stance, angle = (np.concatenate([v[i] for v in train.values()]) for i in range(3))
    if not np.isfinite([y, stance, angle]).all():
        raise ValueError("training baseline inputs must be finite")
    means = [float(y[stance == state].mean()) for state in (0, 1)]
    design = np.stack([np.ones_like(stance), stance, angle, stance * angle], axis=1)
    coef = np.linalg.lstsq(design.astype(float), y, rcond=None)[0]
    curves = np.concatenate([stride_curves(y, heel_strikes_from_stance(st), fs)
                             for y, st, _ in train.values()])
    template = curves.mean(0)
    periods = np.concatenate([np.diff(heel_strikes_from_stance(st)) for _, st, _ in train.values()])
    initial_period = float(np.median(periods[(periods >= 0.6 * fs) & (periods <= 2.2 * fs)]))
    models = dict(spec.split("=", 1) for spec in args.model)
    predictions = {name: {} for name in models}
    for name, path in models.items():
        run = resolve(path)
        model_cfg = replace(Config.load(run / "config.yaml"), **{
            "data.stride": cfg.data.stride, "data.cache_window_index": False})
        if model_cfg.data.split != cfg.data.split or model_cfg.data.window_length != cfg.data.window_length:
            raise ValueError(f"{name}: evaluation split/window differs from baseline")
        print(f"Evaluating {name} (streaming, stride {cfg.data.stride})", flush=True)
        _, trials = Evaluator(model_cfg, str(run / "best.pt"), device=args.device).run("test")
        for trial in trials:
            if not np.allclose(trial.truth, test[trial.name][0], equal_nan=True):
                raise ValueError(f"{name}/{trial.name}: targets differ")
            predictions[name][trial.name] = trial.pred_streaming
        print(f"Finished {name}: {len(trials)} trials", flush=True)

    parts, example = {}, None
    for trial, (y, stance, angle) in test.items():
        hs = heel_strikes_from_stance(stance)
        offline = np.full(len(y), np.nan)
        for start, end in zip(hs[:-1], hs[1:]):
            if 0.6 * fs <= end - start <= 2.2 * fs:
                offline[start:end] = np.interp(np.arange(end - start) / (end - start),
                                              np.linspace(0, 1, len(template)), template)
        design = np.stack([np.ones_like(stance), stance, angle, stance * angle], axis=1)
        pred = {"Binary stance": np.where(stance == 1, means[1], means[0]),
                "Linear stance + angle": np.sum(design.astype(float) * coef, axis=1),
                "Causal previous-stride template": causal_template(stance, template, initial_period, fs),
                "Offline next-heel-strike template": offline,
                **{name: data[trial] for name, data in predictions.items()}}
        windows = _windows_for_trial(y[:, None], cfg.data.window_length, cfg.data.stride,
                                     cfg.data.min_segment_length)
        stream = np.zeros(len(y), dtype=bool)
        stream[[end for _, end in windows]] = True
        for name in models:
            if not np.isfinite(pred[name][stream]).all():
                raise ValueError(f"{name}/{trial}: missing streaming predictions")
        scopes = {"all_streaming": stream,
                  "common_complete_strides": stream & np.isfinite(offline),
                  "historical_offline": np.isfinite(y) & np.isfinite(offline)}
        for scope, mask in scopes.items():
            for name, values in pred.items():
                if (scope == "all_streaming" and name.startswith("Offline")) or (
                    scope == "historical_offline" and name in models):
                    continue
                for subject in (trial.split("_")[0], "pooled"):
                    parts.setdefault((scope, subject, name), []).append((values[mask], y[mask]))
        if example is None:
            example = (trial, y, stance, angle, pred)
    rows = []
    for (scope, subject, model), pairs in sorted(parts.items()):
        pred, truth = (np.concatenate([pair[i] for pair in pairs]) for i in (0, 1))
        rows.append({"scope": scope, "subject": subject, "model": model, **scores(pred, truth)})
    df = pd.DataFrame(rows)
    summary_rows = []
    for (scope, model), data in df[df.subject != "pooled"].groupby(["scope", "model"]):
        row = {"scope": scope, "model": model, "subjects": len(data)}
        for metric in ("r2_corr", "r2_pred", "rmse_nm_per_kg"):
            row[f"{metric}_mean"] = float(data[metric].mean())
            row[f"{metric}_sd"] = float(data[metric].std(ddof=1))
        summary_rows.append(row)
    out = resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "per_subject.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(out / "subject_summary.csv", index=False)
    (out / "provenance.json").write_text(json.dumps({
        "config": str(resolve(args.config)), "models": models,
        "train_subjects": cfg.data.split.train, "test_subjects": cfg.data.split.test,
        "streaming_step_samples": cfg.data.stride, "rate_hz": fs,
        "training_template_strides": len(curves), "initial_period_samples": initial_period,
        "r2_corr": "squared within-subject Pearson r; unweighted subject mean and sample SD",
        "r2_pred": "1-SSE/SST; both per subject and pooled",
        "offline_baseline": "uses the next heel strike; historical_offline reproduces old baseline mask",
        "source": "Molinaro et al., Science Robotics 9, eadi8852 (2024), https://doi.org/10.1126/scirobotics.adi8852",
    }, indent=2) + "\n")
    trial, y, stance, angle, pred = example
    sl = slice(1000, 1500)
    time = np.arange(len(y)) / fs
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    axes[0].fill_between(time[sl], 0, 1, where=stance[sl] > 0, color="0.9",
                         transform=axes[0].get_xaxis_transform(), label="Stance (0/1)")
    axes[0].plot(time[sl], np.degrees(angle[sl]), color="tab:purple", label="Ankle angle")
    axes[0].set(ylabel="Ankle angle (deg)", title=f"Test trial {trial}")
    axes[0].legend(loc="upper right")
    axes[1].plot(time[sl], y[sl], "k--", lw=2, label="Measured ankle moment")
    for name in ("Binary stance", "Causal previous-stride template", "Offline next-heel-strike template"):
        axes[1].plot(time[sl], pred[name][sl], label=name)
    axes[1].set(xlabel="Time (s)", ylabel="Ankle moment (N·m/kg)",
                title="Training-fitted baselines: causal timing vs retrospective alignment")
    axes[1].legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "gait_baselines.png", dpi=140)
    print(df[(df.subject == "pooled") & (df.scope == "all_streaming")].to_string(index=False))
    print(pd.DataFrame(summary_rows).query("scope == 'all_streaming'").to_string(index=False))


if __name__ == "__main__":
    main()
