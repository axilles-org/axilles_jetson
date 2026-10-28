"""Replay a recorded exo walk through trained models and compare predicted torque with GaTech.

The exo has no torque ground truth, so the check is against what walking torque looks
like: each model's stride-averaged prediction (split at FSR heel strikes) is scored
against the GaTech inverse-dynamics moment of the training subjects with waveform_metrics.

    python scripts/replay_exo_recording.py \
      --recording <exo csv: time_s, foot_ax..shank_gz, ankle_encoder_deg, *_fsr_raw> \
      --model baseline=runs/ablation/<id>/original:adapter \
      --model tx_all=runs/ablation/<id>/tx_all:raw \
      --mass 72 --height 1.75 --gender M --out runs/exo_replay/<name>

IMU mode per model: ``adapter`` maps the exo IMUs into the GaTech frame with
deploy.sensor_adapter (models trained on original GaTech data); ``raw`` feeds the exo
IMUs as recorded (models trained on GaTech data transformed into the exo frame).
The encoder offset changes every mounting, so the ankle angle is zeroed per recording
by matching its median to the GaTech training median.
"""
from __future__ import annotations

import argparse
import dataclasses
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
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

from waveform_metrics import compute_all

from exo.config import Config
from exo.data.gait_cycles import N_PHASE, heel_strikes_from_stance, stride_curves
from exo.data.raw_ingest import _decimate
from exo.deploy.export import build_deploy_module
from exo.deploy.sensor_adapter import SensorAdapter

RAW_FS, MODEL_FS = 200, 100
IMU_COLUMNS = {f"imu_{seg}_{kind}_{ax}": f"{seg}_{kind[0].lower()}{ax.lower()}"
               for seg in ("foot", "shank") for kind in ("Accel", "Gyro") for ax in "XYZ"}


def gatech_reference(cfg: Config) -> tuple[np.ndarray, float]:
    """(strides, 101) moment curves in N·m/kg and the median ankle angle, training subjects."""
    meta = pd.read_parquet(cfg.paths.demographics()).drop_duplicates("subject")
    mass = dict(zip(meta.subject, meta.weight_kg))
    curves, angles = [], []
    for subject in cfg.data.split.train:
        for f in sorted(cfg.paths.processed().glob(f"{subject}_*.npz")):
            d = np.load(f, allow_pickle=True)
            cols = list(d["feature_columns"])
            stance = d["features"][:, cols.index("stance")]
            angles.append(d["features"][:, cols.index("gon_ankle_sagittal")])
            curves.append(stance_strides(d["target"][:, 0] / mass[subject], stance))
    return np.concatenate(curves), float(np.nanmedian(np.concatenate(angles)))


def stance_strides(signal: np.ndarray, stance: np.ndarray) -> np.ndarray:
    return stride_curves(signal, heel_strikes_from_stance(stance), MODEL_FS)


def exo_features(recording: pd.DataFrame, cfg: Config, imu_mode: str,
                 gatech_angle_median: float) -> pd.DataFrame:
    adapter_cfg = cfg.deploy.sensor_adapter
    angle_deg = np.degrees(gatech_angle_median) * adapter_cfg.ankle_encoder_sign
    neutral = recording["ankle_encoder_deg"].median() - angle_deg
    adapter_cfg = dataclasses.replace(adapter_cfg, ankle_encoder_neutral_deg=float(neutral))

    raw = recording.set_index("time_s")
    adapted = SensorAdapter(adapter_cfg).transform_batch(raw)
    if imu_mode == "raw":
        for feature, column in IMU_COLUMNS.items():
            adapted[feature] = raw[column].to_numpy()

    names = cfg.data.feature_names()
    factor = RAW_FS // MODEL_FS
    continuous = [n for n in names if n != "stance"]
    out = pd.DataFrame(_decimate(adapted[continuous].to_numpy(np.float64), factor, antialias=True),
                       columns=continuous)
    out["stance"] = adapted["stance"].to_numpy()[::factor][:len(out)]
    return out[names]


@torch.inference_mode()
def predict(run_dir: Path, features: pd.DataFrame, mass: float, demo: tuple) -> np.ndarray:
    deploy, info = build_deploy_module(run_dir, mass, demo)
    x = torch.tensor(features.to_numpy(np.float32).T)
    win = info["window_length"]
    windows = x.unfold(1, win, 1).permute(1, 0, 2)                   # (N, C, win)
    z = (windows - deploy.x_mean[0].T[..., None]) / deploy.x_scale[0].T[..., None]
    preds = []
    for chunk in torch.split(z, 512):
        out = deploy.model(chunk, deploy.subject_idx.expand(len(chunk)))[:, 0, -1]
        preds.append(out * deploy.y_scale + deploy.y_mean)
    stream = np.full(len(features), np.nan)
    stream[win - 1:] = (torch.cat(preds) / deploy.subject_mass).numpy()
    return stream


def summarize(curves: np.ndarray, reference: np.ndarray) -> dict:
    mean, ref = curves.mean(0), reference.mean(0)
    m = compute_all(mean, ref)
    return {
        "strides": len(curves),
        "peak_plantarflexion_nm_per_kg": float(mean.min()),
        "peak_at_pct": int(mean.argmin()),
        "rmse_vs_gatech": m.rmse,
        "pearson_r_vs_gatech": m.pearson_r,
        "peak_timing_error_pct": float(abs(mean.argmin() - ref.argmin())),
        "stride_sd_mean": float(curves.std(0).mean()),
    }


MOMENT_LABEL = "Ankle moment (N·m/kg)\n← plantarflexion | dorsiflexion →"


def _shade_stance(ax, times: np.ndarray, stance: np.ndarray) -> None:
    ax.fill_between(times, 0, 1, where=stance > 0, color="0.92", step="mid",
                    transform=ax.get_xaxis_transform(), label="Stance (FSR)")


def plot(results: dict, reference: np.ndarray, times: np.ndarray, title: str, path: Path,
         zoom_s: float = 10.0) -> None:
    pct = np.linspace(0, 100, N_PHASE)
    first = next(iter(results.values()))
    stance, valid = first["stance"], ~np.isnan(first["stream"])
    hs_times = times[heel_strikes_from_stance(stance)]
    zoom_start = times[valid][0]

    fig = plt.figure(figsize=(15, 9))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.1])
    full, zoom, avg = fig.add_subplot(grid[0, :]), fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])

    for ax in (full, zoom):
        _shade_stance(ax, times, stance)
        ax.axhline(0, color="0.6", lw=0.8)
    for (name, r), color in zip(results.items(), plt.cm.tab10.colors):
        full.plot(times, r["stream"], color=color, lw=1, label=name)
        zoom.plot(times, r["stream"], color=color, lw=1.6, label=name)
        m, s = r["curves"].mean(0), r["curves"].std(0)
        avg.fill_between(pct, m - s, m + s, color=color, alpha=0.18)
        avg.plot(pct, m, color=color, lw=2.2, label=f"{name}, mean ± 1 SD ({len(r['curves'])} strides)")
    for t in hs_times[(hs_times >= zoom_start) & (hs_times <= zoom_start + zoom_s)]:
        zoom.axvline(t, color="0.35", lw=0.8, ls=":")

    ref_m, ref_s = reference.mean(0), reference.std(0)
    avg.fill_between(pct, ref_m - ref_s, ref_m + ref_s, color="0.8", alpha=0.6)
    avg.plot(pct, ref_m, color="0.25", lw=2, ls="--",
             label=f"GaTech measured (inverse dynamics), mean ± 1 SD ({len(reference)} strides)")
    avg.axhline(0, color="0.6", lw=0.8)

    full.set(xlabel="Time (s)", ylabel=MOMENT_LABEL, xlim=(times[0], times[-1]),
             title=f"Predicted ankle moment, full recording ({times[-1]:.0f} s; first 3 s fill the model window)")
    zoom.set(xlabel="Time (s)", ylabel=MOMENT_LABEL, xlim=(zoom_start, zoom_start + zoom_s),
             title=f"First {zoom_s:.0f} s of prediction (dotted lines: FSR heel strikes)")
    avg.set(xlabel="Gait cycle (%), 0 = FSR heel strike", ylabel=MOMENT_LABEL, xlim=(0, 100),
            title="Stride-averaged prediction vs GaTech measured moment")
    handles = {}
    for ax in (full, zoom, avg):
        for h, label in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(label.split(", mean")[0], h)
    fig.legend(handles.values(), handles.keys(), loc="lower center", ncol=2, fontsize=9,
               frameon=False)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04 + 0.025 * ((len(handles) + 1) // 2), 1, 1))
    fig.savefig(path, dpi=140)


def parse_model(spec: str) -> tuple[str, Path, str]:
    name, rest = spec.split("=", 1)
    run_dir, mode = rest.rsplit(":", 1)
    if mode not in ("raw", "adapter"):
        raise SystemExit(f"{spec}: IMU mode must be 'raw' or 'adapter'")
    return name, Path(run_dir), mode


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recording", required=True, type=Path)
    ap.add_argument("--model", action="append", required=True, metavar="NAME=RUN_DIR:MODE")
    ap.add_argument("--mass", type=float, required=True)
    ap.add_argument("--height", type=float, required=True)
    ap.add_argument("--gender", choices=["M", "F"], required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--title", default=None, help="figure title (default: recording file name)")
    a = ap.parse_args()

    recording = pd.read_csv(a.recording)
    models = [parse_model(s) for s in a.model]
    reference, angle_median = gatech_reference(Config.load(models[0][1] / "config.yaml"))

    results = {}
    for name, run_dir, mode in models:
        cfg = Config.load(run_dir / "config.yaml")
        feats = exo_features(recording, cfg, mode, angle_median)
        stream = predict(run_dir, feats, a.mass, (a.height, a.mass, a.gender))
        stance = feats["stance"].to_numpy()
        results[name] = {"stream": stream, "stance": stance,
                         "curves": stance_strides(stream, stance), "mode": mode}

    a.out.mkdir(parents=True, exist_ok=True)
    times = np.arange(len(next(iter(results.values()))["stream"])) / MODEL_FS
    plot(results, reference, times, a.title or a.recording.name, a.out / "torque_replay.png")
    summary = {
        "recording": a.recording.name,
        "gatech_reference": {"strides": len(reference),
                             "peak_plantarflexion_nm_per_kg": float(reference.mean(0).min()),
                             "peak_at_pct": int(reference.mean(0).argmin())},
        "models": {n: {"run_dir": str(dict((m[0], m[1]) for m in models)[n]), "imu_mode": r["mode"],
                       **summarize(r["curves"], reference)} for n, r in results.items()},
    }
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
