"""Subject demographics, read from the dataset's ``metadata.parquet``."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def _table(metadata_path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(metadata_path)
    cols = ["subject", "gender", "height_m", "weight_kg"]
    return df[cols].drop_duplicates("subject").set_index("subject")


def subject_mass(metadata_path: str | Path) -> dict[str, float]:
    return _table(metadata_path)["weight_kg"].astype(float).to_dict()


def demographics(metadata_path: str | Path) -> dict[str, dict]:
    """{subject: {"gender", "height_m", "weight_kg"}}."""
    return _table(metadata_path).to_dict("index")
