"""Build GaTech dataset variants that mix original and exo-frame-transformed IMU channels.

A variant names which IMU channel groups (``foot_Accel``, ``shank_Gyro``, ...) are
taken from the transformed dataset; every other IMU group and every non-IMU stream
come from the original. Non-IMU files are symlinked, so a variant costs only its
rewritten ``__imu.parquet`` files on disk.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

IMU_GROUPS = ("foot_Accel", "foot_Gyro", "shank_Accel", "shank_Gyro")
_AXES = ("X", "Y", "Z")


@dataclass(frozen=True)
class Variant:
    name: str
    tx_groups: frozenset[str] = frozenset()
    drop_groups: frozenset[str] = frozenset()

    @property
    def data_key(self) -> str:
        """Variants with the same transformed groups share one dataset and cache."""
        return "tx_" + "_".join(sorted(self.tx_groups)) if self.tx_groups else "original"


def _columns(group: str) -> list[str]:
    return [f"{group}_{ax}" for ax in _AXES]


def _link(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(src.resolve(), dst)
    except OSError:
        shutil.copy2(src, dst)


def build_dataset(original_root: Path, tx_root: Path, out_root: Path,
                  tx_groups: frozenset[str]) -> Path:
    """Write a dataset root whose IMU groups in ``tx_groups`` come from ``tx_root``."""
    unknown = set(tx_groups) - set(IMU_GROUPS)
    if unknown:
        raise ValueError(f"unknown IMU groups: {sorted(unknown)}")
    if not tx_groups:
        return original_root
    if set(tx_groups) == set(IMU_GROUPS):
        return tx_root

    out_root.mkdir(parents=True, exist_ok=True)
    _link(original_root / "metadata.parquet", out_root / "metadata.parquet")

    for src_dir in sorted((original_root / "subjects").iterdir()):
        dst_dir = out_root / "subjects" / src_dir.name
        dst_dir.mkdir(parents=True, exist_ok=True)
        for src in src_dir.glob("*.parquet"):
            if not src.name.endswith("__imu.parquet"):
                _link(src, dst_dir / src.name)
                continue
            dst = dst_dir / src.name
            tx_file = tx_root / "subjects" / src_dir.name / src.name
            if dst.exists() or not tx_file.exists():
                continue
            _write_mixed_imu(src, tx_file, dst, tx_groups)
    return out_root


def _write_mixed_imu(original: Path, transformed: Path, dst: Path,
                     tx_groups: frozenset[str]) -> None:
    orig = pd.read_parquet(original)
    tx_cols = [c for g in sorted(tx_groups) for c in _columns(g)]
    tx = pd.read_parquet(transformed, columns=tx_cols)
    if len(tx) != len(orig):
        raise ValueError(f"{original.name}: length mismatch {len(orig)} vs {len(tx)}")
    orig[tx_cols] = tx.to_numpy()
    orig.to_parquet(dst, index=False)


def drop_features(feature_names: list[str], drop_groups: frozenset[str]) -> list[str]:
    """Remove features: an IMU group drops its three ``imu_<group>_<axis>`` columns,
    any other name is dropped as a feature name (e.g. ``stance``)."""
    dropped = set()
    for g in drop_groups:
        dropped |= {f"imu_{c}" for c in _columns(g)} if g in IMU_GROUPS else {g}
    unknown = dropped - set(feature_names)
    if unknown:
        raise ValueError(f"not in the feature set: {sorted(unknown)}")
    kept = [f for f in feature_names if f not in dropped]
    if not kept:
        raise ValueError("every feature dropped")
    return kept
