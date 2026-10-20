"""Apply a committed transform JSON to the Hugging Face Parquet copy of the GaTech dataset.

Same math as imu_locator.exo_transform.Transform.apply, without the MuJoCo import:

    gyro_exo = R gyro_gt
    acc_exo  = R acc_gt + alpha x d + omega x (omega x d)     (omega low-passed at fc_lever)

Output mirrors the input layout (``metadata.parquet``, ``subjects/<AB>/<trial>__*.parquet``):
``__imu.parquet`` holds ``time_s`` and ``{foot,shank}_{Accel,Gyro}_{X,Y,Z}`` in the target
frame (m/s^2, rad/s); every other stream is symlinked unchanged.

  python scripts/transform_hf_dataset.py \
    --hf-root /path/to/hf_dataset \
    --transform 3_transformation_matrices/imu_raw_frame/transform_version1_average.json \
    --out /path/to/hf_dataset_tx_mujoco
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

G = 9.80665
SEGMENTS = ("foot", "shank")


def lowpass(x: np.ndarray, fs: float, fc: float, order: int = 4) -> np.ndarray:
    b, a = butter(order, fc / (0.5 * fs))
    return filtfilt(b, a, x, axis=0)


def apply_transform(acc_g: np.ndarray, gyro: np.ndarray, fs: float, seg: dict,
                    fc_lever: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    R = np.asarray(seg["R_gt_sensor_to_exo"], float)
    d = np.asarray(seg["lever_arm_d_m"], float)
    w = gyro @ R.T
    a = (acc_g * G * seg.get("acc_sign", 1)) @ R.T
    wf = lowpass(w, fs, fc_lever)
    dw = np.gradient(wf, 1.0 / fs, axis=0)
    return a + np.cross(dw, d) + np.cross(wf, np.cross(wf, d)), w


def transform_trial(src: Path, dst: Path, transform: dict) -> None:
    df = pd.read_parquet(src)
    t = df["time_s"].to_numpy(float)
    fs = 1.0 / np.median(np.diff(t))
    cols = {"time_s": t}
    for seg in SEGMENTS:
        acc = df[[f"{seg}_Accel_{ax}" for ax in "XYZ"]].to_numpy(float)
        gyro = df[[f"{seg}_Gyro_{ax}" for ax in "XYZ"]].to_numpy(float)
        a, w = apply_transform(acc, gyro, fs, transform[seg])
        for i, ax in enumerate("XYZ"):
            cols[f"{seg}_Accel_{ax}"] = a[:, i]
            cols[f"{seg}_Gyro_{ax}"] = w[:, i]
    pd.DataFrame(cols).to_parquet(dst, index=False)


def link(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        shutil.copy2(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-root", required=True, type=Path)
    ap.add_argument("--transform", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    transform = json.loads(a.transform.read_text())
    a.out.mkdir(parents=True, exist_ok=True)
    link(a.hf_root / "metadata.parquet", a.out / "metadata.parquet")

    n = 0
    for subj_dir in sorted((a.hf_root / "subjects").iterdir()):
        dst_dir = a.out / "subjects" / subj_dir.name
        dst_dir.mkdir(parents=True, exist_ok=True)
        for src in sorted(subj_dir.glob("*.parquet")):
            if src.name.endswith("__imu.parquet"):
                transform_trial(src, dst_dir / src.name, transform)
                n += 1
            else:
                link(src, dst_dir / src.name)
    print(f"{n} trials -> {a.out}")


if __name__ == "__main__":
    main()
