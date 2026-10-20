"""Hugging Face dataset hosting and retrieval.

The canonical dataset is an HF dataset repo laid out as::

    metadata.parquet                     one row per (subject, trial)
    subjects/<subject>/<trial>__<sensor>.parquet

Each sensor file holds ``time_s`` plus that sensor's channels at its native rate,
so the full-resolution signal is preserved and any target rate can be derived
locally.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

SENSORS = ("imu", "id", "fp", "gon", "gcRight", "conditions")

_METADATA_FILE = "metadata.parquet"


@dataclass(frozen=True)
class HubConfig:
    repo_id: str
    revision: str = "main"


def dataset_path(subject: str, trial: str, sensor: str) -> str:
    return f"subjects/{subject}/{trial}__{sensor}.parquet"


def fetch(cfg: HubConfig, dest: str | Path, allow_patterns: list[str] | None = None) -> Path:
    """Download the dataset (or a subset) to ``dest`` and return the local path."""
    from huggingface_hub import snapshot_download

    local = snapshot_download(
        repo_id=cfg.repo_id,
        repo_type="dataset",
        revision=cfg.revision,
        local_dir=str(dest),
        allow_patterns=allow_patterns,
    )
    return Path(local)


def load_metadata(root: str | Path) -> pd.DataFrame:
    return pd.read_parquet(Path(root) / _METADATA_FILE)


def read_sensor(root: str | Path, subject: str, trial: str, sensor: str) -> pd.DataFrame | None:
    path = Path(root) / dataset_path(subject, trial, sensor)
    return pd.read_parquet(path) if path.exists() else None
