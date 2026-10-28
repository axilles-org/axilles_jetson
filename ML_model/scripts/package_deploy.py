"""Package a trained run into a self-contained deploy folder for the Jetson.

    python scripts/package_deploy.py --run runs/ablation/20261009_110715/tx_all --out runs/deploy

The folder holds ``deploy.pt`` plus everything needed to re-export it on the Jetson
for a new wearer (scalers, demographics of the training subjects) without the
dataset. The model was trained on GaTech data transformed into the exo IMU frame,
so the sensor adapter passes the exo IMUs through unchanged. ``ankle_reference.json``
holds the GaTech walking median ankle angle, used to zero the encoder live.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from exo.config import Config, replace
from exo.deploy.export import export

COPIED = ("scaler_bundle.pkl", "model_meta.json", "test_metrics.json")
EXO_FRAME_IMU = {
    "foot_rotation": np.eye(3).tolist(), "shank_rotation": np.eye(3).tolist(),
    "foot_accel_scale": 1.0, "foot_gyro_scale": 1.0,
    "shank_accel_scale": 1.0, "shank_gyro_scale": 1.0,
    "foot_lever_arm_m": [0.0, 0.0, 0.0], "shank_lever_arm_m": [0.0, 0.0, 0.0],
}


def gatech_ankle_median_deg(cfg: Config) -> float:
    angles = []
    for subject in cfg.data.split.train:
        for f in sorted(cfg.paths.processed().glob(f"{subject}_*.npz")):
            d = np.load(f, allow_pickle=True)
            angles.append(d["features"][:, list(d["feature_columns"]).index("gon_ankle_sagittal")])
    return float(np.degrees(np.nanmedian(np.concatenate(angles))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--out", default="runs/deploy", type=Path)
    ap.add_argument("--name", default="TCN exo-frame (Amily transform, tx_all)")
    a = ap.parse_args()

    cfg = Config.load(a.run / "config.yaml")
    a.out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(a.run / "best.pt", a.out / "deploy.pt")
    for name in COPIED:
        shutil.copy2(a.run / name, a.out / name)

    demo = pd.read_parquet(cfg.paths.demographics()).drop_duplicates("subject")
    demo = demo[demo.subject.isin(cfg.data.split.train)][["subject", "height_m", "weight_kg", "gender"]]
    demo.to_parquet(a.out / "demographics.parquet", index=False)

    reference = {"gatech_walking_median_deg": gatech_ankle_median_deg(cfg), "source_run": str(a.run)}
    (a.out / "ankle_reference.json").write_text(json.dumps(reference, indent=2))

    adapter = {**cfg.to_dict()["deploy"]["sensor_adapter"], **EXO_FRAME_IMU}
    replace(cfg, **{"deploy.sensor_adapter": adapter, "deploy.backend": "onnx"}).save(a.out)
    (a.out / "display_name.txt").write_text(a.name + "\n")

    artifacts = export(a.out, 72.0, subject_demo=(1.75, 72.0, "M"), onnx=True, checkpoint="deploy.pt")
    print(f"packaged {a.run} -> {a.out}")
    print(f"ankle reference: {reference['gatech_walking_median_deg']:.2f} deg")
    print(f"onnx: {artifacts.onnx}  (placeholder wearer; jetson_deploy.py re-exports per wearer)")


if __name__ == "__main__":
    main()
