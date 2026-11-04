"""Training-only ankle reference used by automatic encoder calibration."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..config import Config


def gatech_ankle_median_deg(cfg: Config) -> float:
    angles = []
    for subject in cfg.data.split.train:
        files = sorted(cfg.paths.processed().glob(f"{subject}_*.npz"))
        if not files:
            raise FileNotFoundError(f"No processed training trials for {subject}")
        for path in files:
            with np.load(path, allow_pickle=True) as data:
                column = list(data["feature_columns"]).index("gon_ankle_sagittal")
                values = data["features"][:, column]
                angles.append(values[np.isfinite(values)])
    if not angles or not any(a.size for a in angles):
        raise ValueError("No finite training ankle angles for automatic calibration")
    return float(np.degrees(np.median(np.concatenate(angles))))


def load_ankle_reference(run_dir: Path) -> float:
    path = run_dir / "ankle_reference.json"
    try:
        value = json.loads(path.read_text())["gatech_walking_median_deg"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError("reference must be a finite number in degrees")
        return float(value)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(
            f"Automatic ankle calibration requires a valid {path}. "
            "Update the model package to include its training ankle reference."
        ) from exc
