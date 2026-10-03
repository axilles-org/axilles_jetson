"""Estimate SensorAdapter parameters from a short static-hold exo recording.

Static hold  -> per-IMU gravity direction  -> rotation aligning the exo frame's
"up" with the training convention's up axis, plus the accel unit scale.
Known ankle pose -> encoder neutral offset.
FSR baseline -> contact-threshold floor.

This fixes only the gravity-alignable 2 rotational DOF and the units. For the
remaining yaw DOF, the IMU lever arms, the encoder sign, and calibrated FSR
thresholds, run ``scripts/fit_adapter.py`` on a walking trial afterwards (it can
take this file's output as a starting point). Writes a ``sensor_adapter`` block.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Training-convention "up" for a foot/shank IMU at rest: gravity reads +Z.
_TRAINING_UP = np.array([0.0, 0.0, 1.0])


def _rotation_aligning(measured_up: np.ndarray, target_up: np.ndarray) -> np.ndarray:
    a = measured_up / np.linalg.norm(measured_up)
    b = target_up / np.linalg.norm(target_up)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-8:
        return np.eye(3) if c > 0 else -np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--static", required=True, help="CSV of a still, feet-flat hold")
    ap.add_argument("--neutral-ankle-deg", type=float, default=None,
                    help="encoder reading at the neutral ankle pose")
    ap.add_argument("--out", default="sensor_adapter.yaml")
    args = ap.parse_args()

    df = pd.read_csv(args.static)
    foot_up = df[["foot_ax", "foot_ay", "foot_az"]].mean().to_numpy()
    shank_up = df[["shank_ax", "shank_ay", "shank_az"]].mean().to_numpy()

    foot_R = _rotation_aligning(foot_up, _TRAINING_UP)
    shank_R = _rotation_aligning(shank_up, _TRAINING_UP)

    g = 9.80665
    accel_scale = float(g / np.linalg.norm(foot_up)) if np.linalg.norm(foot_up) > 1e-6 else 1.0

    heel_baseline = float(df["heel_fsr_raw"].median()) if "heel_fsr_raw" in df else 0.0
    toe_baseline = float(df["toe_fsr_raw"].median()) if "toe_fsr_raw" in df else 0.0

    shank_up_mag = float(np.linalg.norm(shank_up))
    shank_accel_scale = g / shank_up_mag if shank_up_mag > 1e-6 else 1.0

    adapter = {
        "sensor_adapter": {
            "foot_rotation": foot_R.round(6).tolist(),
            "shank_rotation": shank_R.round(6).tolist(),
            "foot_accel_scale": round(accel_scale, 6),
            "foot_gyro_scale": 1.0,
            "shank_accel_scale": round(shank_accel_scale, 6),
            "shank_gyro_scale": 1.0,
            "foot_lever_arm_m": [0.0, 0.0, 0.0],
            "shank_lever_arm_m": [0.0, 0.0, 0.0],
            "ankle_encoder_neutral_deg": args.neutral_ankle_deg
            if args.neutral_ankle_deg is not None
            else round(float(df["ankle_encoder_deg"].mean()), 3),
            "ankle_encoder_sign": 1.0,
            "heel_fsr_threshold": round(heel_baseline + 1500.0, 1),
            "toe_fsr_threshold": round(toe_baseline + 1500.0, 1),
            "fsr_debounce_s": 0.10,
        }
    }
    with open(args.out, "w") as f:
        yaml.safe_dump(adapter, f, sort_keys=False)
    print(yaml.safe_dump(adapter, sort_keys=False))
    print(f"written -> {args.out}")


if __name__ == "__main__":
    main()
