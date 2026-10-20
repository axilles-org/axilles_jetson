"""How well GaTech IMU signals match the exo's own IMUs, before and after a transform.

Mean gait cycles (heel strike to heel strike) of every foot/shank IMU channel: the exo
recording against each GaTech subject, once with the original GaTech data and once
with the transformed copy, scored with waveform_metrics.

    python scripts/compare_imu_to_exo.py \
      --exo <exo csv in the target frame: time_s, foot_ax..shank_gz, *_fsr_raw> \
      --gatech-original /path/hf_dataset --gatech-transformed /path/hf_dataset_tx_mujoco \
      --out runs/report/2_transform_verification
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

from waveform_metrics import compute_all

from exo.config import Config
from exo.data.gait_cycles import (
    N_PHASE,
    heel_strikes_from_phase,
    heel_strikes_from_stance,
    stride_curves,
)
from exo.deploy.sensor_adapter import SensorAdapter

G = 9.80665
FS = 200
SEGMENTS, SENSORS, AXES = ("foot", "shank"), ("Accel", "Gyro"), "XYZ"
CHANNELS = [f"{seg}_{s}_{ax}" for seg in SEGMENTS for s in SENSORS for ax in AXES]
EXO_COLUMNS = {c: f"{c.split('_')[0]}_{c.split('_')[1][0].lower()}{c[-1].lower()}" for c in CHANNELS}
UNITS = {"Accel": "m/s²", "Gyro": "rad/s"}
METRICS = ("pearson_r", "rmse", "nrmse", "peak_timing_error_pct", "cmc")


def exo_strides(path: Path, cfg: Config) -> np.ndarray:
    df = pd.read_csv(path)
    stance = SensorAdapter(cfg.deploy.sensor_adapter).transform_batch(df.set_index("time_s"))["stance"]
    signal = df[[EXO_COLUMNS[c] for c in CHANNELS]].to_numpy(float)
    return stride_curves(signal, heel_strikes_from_stance(stance.to_numpy()), FS)


def gatech_strides(root: Path, subject: str, trials: list[str], accel_in_g: bool) -> np.ndarray:
    out = []
    for trial in trials:
        base = root / "subjects" / subject / trial
        imu = pd.read_parquet(f"{base}__imu.parquet")
        gc = pd.read_parquet(f"{base}__gcRight.parquet")
        phase = np.interp(imu["time_s"], gc["time_s"], gc["HeelStrike"])
        signal = imu[CHANNELS].to_numpy(float)
        if accel_in_g:
            signal[:, [i for i, c in enumerate(CHANNELS) if "Accel" in c]] *= G
        out.append(stride_curves(signal, heel_strikes_from_phase(phase), FS))
    return np.concatenate(out)


def score(exo: np.ndarray, gatech: np.ndarray) -> list[dict]:
    rows = []
    for i, ch in enumerate(CHANNELS):
        m = compute_all(exo[..., i].mean(0), gatech[..., i].mean(0),
                        exo_strides=exo[..., i], gatech_strides=gatech[..., i])
        rows.append({"channel": ch, **{k: getattr(m, k) for k in METRICS}})
    return rows


def plot(exo: np.ndarray, pooled: dict[str, np.ndarray], exo_name: str, path: Path) -> None:
    pct = np.linspace(0, 100, N_PHASE)
    fig, axes = plt.subplots(4, 3, figsize=(15, 13), sharex=True)
    styles = {"original": dict(color="0.45", ls="--"), "transformed": dict(color="tab:orange", ls="-")}
    for i, ch in enumerate(CHANNELS):
        ax = axes.flat[i]
        for stage, data in pooled.items():
            m, s = data[..., i].mean(0), data[..., i].std(0)
            if stage == "transformed":
                ax.fill_between(pct, m - s, m + s, color="tab:orange", alpha=0.15)
            ax.plot(pct, m, lw=2, label=f"GaTech {stage}", **styles[stage])
        m, s = exo[..., i].mean(0), exo[..., i].std(0)
        ax.fill_between(pct, m - s, m + s, color="tab:blue", alpha=0.15)
        ax.plot(pct, m, color="tab:blue", lw=2.2, label="Exo (measured)")
        seg, sensor, axis = ch.split("_")
        ax.set_title(f"{seg} {sensor.lower()} {axis}")
        ax.set_ylabel(UNITS[sensor])
    for ax in axes[-1]:
        ax.set_xlabel("Gait cycle (%), 0 = heel strike")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(f"IMU mean gait cycle: exo ({exo_name}) vs GaTech before and after the transform\n"
                 "mean ± 1 SD; exo frame = exo IMU raw sensor axes", fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=130)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exo", required=True, type=Path)
    ap.add_argument("--gatech-original", required=True, type=Path)
    ap.add_argument("--gatech-transformed", required=True, type=Path)
    ap.add_argument("--max-speed", type=float, default=0.95,
                    help="GaTech treadmill trials at or below this speed (m/s); the exo walk is slow")
    ap.add_argument("--config", default="configs/default.yaml", help="for the FSR stance thresholds")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    exo = exo_strides(a.exo, Config.load(a.config))
    meta = pd.read_parquet(a.gatech_original / "metadata.parquet")
    meta = meta[meta.speed_mean_mps <= a.max_speed]
    trials = meta.groupby("subject")["trial"].apply(list).to_dict()

    rows, pooled = [], {"original": [], "transformed": []}
    for subject, subject_trials in sorted(trials.items()):
        for stage, root, in_g in (("original", a.gatech_original, True),
                                  ("transformed", a.gatech_transformed, False)):
            gt = gatech_strides(root, subject, subject_trials, accel_in_g=in_g)
            pooled[stage].append(gt)
            rows += [{"subject": subject, "stage": stage, **r} for r in score(exo, gt)]
    pooled = {k: np.concatenate(v) for k, v in pooled.items()}

    df = pd.DataFrame(rows)
    a.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out / "per_subject_channel.csv", index=False)
    table = df.groupby(["channel", "stage"])[list(METRICS)].median().unstack("stage")
    table.columns = [f"{m}_{s}" for m, s in table.columns]
    table = table.loc[CHANNELS]
    table.to_csv(a.out / "summary_median_over_subjects.csv")
    (a.out / "meta.json").write_text(json.dumps({
        "exo_recording": a.exo.name, "exo_strides": len(exo),
        "gatech_subjects": sorted(trials), "gatech_strides": {k: len(v) for k, v in pooled.items()},
        "max_speed_mps": a.max_speed}, indent=2))
    plot(exo, pooled, a.exo.stem, a.out / "imu_mean_cycles.png")
    pd.set_option("display.width", 220)
    print(f"exo strides {len(exo)}, GaTech strides {len(pooled['original'])} from {len(trials)} subjects")
    print(table.round(3).to_string())


if __name__ == "__main__":
    main()
