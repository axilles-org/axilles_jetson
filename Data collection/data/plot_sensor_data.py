#!/usr/bin/env python3
"""
Plot grouped sensor signals from a data_collection CSV.

Usage:
1) Set FILE_NAME below (base name without .csv).
2) Run:
       python3 "Data collection/data/plot_sensor_data.py"
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


# Paste the file name you want to plot (WITHOUT .csv extension).
# Example: data_collection_20260404_214949_100Hz_2kmph
FILE_NAME = "data_collection_20260404_215743_tightshoes_200Hz_2kmph"


def _to_float(value: str):
    if value is None:
        return None
    value = value.strip()
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _read_csv(csv_path: Path):
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"CSV has no rows: {csv_path}")

    columns = {
        "time": "timestamp_s",
        "foot_gyro": ["foot_gx", "foot_gy", "foot_gz"],
        "foot_accel": ["foot_ax", "foot_ay", "foot_az"],
        "shank_gyro": ["shank_gx", "shank_gy", "shank_gz"],
        "shank_accel": ["shank_ax", "shank_ay", "shank_az"],
        "encoder": ["ankle_encoder_deg"],
        "fsr": ["toe_fsr_raw", "heel_fsr_raw"],
    }

    for key in [
        columns["time"],
        *columns["foot_gyro"],
        *columns["foot_accel"],
        *columns["shank_gyro"],
        *columns["shank_accel"],
        *columns["encoder"],
        *columns["fsr"],
    ]:
        if key not in reader.fieldnames:
            raise KeyError(f"Missing required column '{key}' in {csv_path.name}")

    data = {
        "time": [],
        "foot_gyro": {"x": [], "y": [], "z": []},
        "foot_accel": {"x": [], "y": [], "z": []},
        "shank_gyro": {"x": [], "y": [], "z": []},
        "shank_accel": {"x": [], "y": [], "z": []},
        "encoder": [],
        "fsr": {"toe": [], "heel": []},
    }

    for row in rows:
        data["time"].append(_to_float(row[columns["time"]]))

        data["foot_gyro"]["x"].append(_to_float(row[columns["foot_gyro"][0]]))
        data["foot_gyro"]["y"].append(_to_float(row[columns["foot_gyro"][1]]))
        data["foot_gyro"]["z"].append(_to_float(row[columns["foot_gyro"][2]]))

        data["foot_accel"]["x"].append(_to_float(row[columns["foot_accel"][0]]))
        data["foot_accel"]["y"].append(_to_float(row[columns["foot_accel"][1]]))
        data["foot_accel"]["z"].append(_to_float(row[columns["foot_accel"][2]]))

        data["shank_gyro"]["x"].append(_to_float(row[columns["shank_gyro"][0]]))
        data["shank_gyro"]["y"].append(_to_float(row[columns["shank_gyro"][1]]))
        data["shank_gyro"]["z"].append(_to_float(row[columns["shank_gyro"][2]]))

        data["shank_accel"]["x"].append(_to_float(row[columns["shank_accel"][0]]))
        data["shank_accel"]["y"].append(_to_float(row[columns["shank_accel"][1]]))
        data["shank_accel"]["z"].append(_to_float(row[columns["shank_accel"][2]]))

        data["encoder"].append(_to_float(row[columns["encoder"][0]]))

        data["fsr"]["toe"].append(_to_float(row[columns["fsr"][0]]))
        data["fsr"]["heel"].append(_to_float(row[columns["fsr"][1]]))

    return data


def _plot(data, file_name: str):
    t = data["time"]

    fig, axes = plt.subplots(6, 1, figsize=(14, 16), sharex=True)
    fig.suptitle(f"Sensor Data Plot: {file_name}", fontsize=14)

    axes[0].plot(t, data["foot_gyro"]["x"], label="gx")
    axes[0].plot(t, data["foot_gyro"]["y"], label="gy")
    axes[0].plot(t, data["foot_gyro"]["z"], label="gz")
    axes[0].set_ylabel("Foot Gyro")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")

    axes[1].plot(t, data["foot_accel"]["x"], label="ax")
    axes[1].plot(t, data["foot_accel"]["y"], label="ay")
    axes[1].plot(t, data["foot_accel"]["z"], label="az")
    axes[1].set_ylabel("Foot Accel")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right")

    axes[2].plot(t, data["shank_gyro"]["x"], label="gx")
    axes[2].plot(t, data["shank_gyro"]["y"], label="gy")
    axes[2].plot(t, data["shank_gyro"]["z"], label="gz")
    axes[2].set_ylabel("Shank Gyro")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="upper right")

    axes[3].plot(t, data["shank_accel"]["x"], label="ax")
    axes[3].plot(t, data["shank_accel"]["y"], label="ay")
    axes[3].plot(t, data["shank_accel"]["z"], label="az")
    axes[3].set_ylabel("Shank Accel")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc="upper right")

    axes[4].plot(t, data["encoder"], label="ankle_encoder_deg", color="tab:orange")
    axes[4].set_ylabel("Encoder (deg)")
    axes[4].grid(True, alpha=0.3)
    axes[4].legend(loc="upper right")

    axes[5].plot(t, data["fsr"]["toe"], label="toe_fsr_raw", color="tab:red")
    axes[5].plot(t, data["fsr"]["heel"], label="heel_fsr_raw", color="tab:green")
    axes[5].set_ylabel("FSR")
    axes[5].set_xlabel("Time (s)")
    axes[5].grid(True, alpha=0.3)
    axes[5].legend(loc="upper right")

    plt.tight_layout(rect=[0, 0.02, 1, 0.98])
    plt.show()


def main():
    data_dir = Path(__file__).resolve().parent
    csv_path = data_dir / f"{FILE_NAME}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Could not find: {csv_path}\n"
            "Set FILE_NAME to a valid CSV base name in this folder."
        )

    data = _read_csv(csv_path)
    _plot(data, FILE_NAME)


if __name__ == "__main__":
    main()
