"""Read one trial and return its sensor streams aligned at the target rate.

Streams are anti-alias decimated to ``target_rate_hz``. A binary ``stance`` channel
is derived from the ``gcRight`` gait-phase signals. Output is in physical units;
z-scoring happens later in :mod:`exo.data.scalers`.

Source data is the Parquet layout produced by ``scripts/csv_to_parquet.py`` /
hosted on Hugging Face: ``subjects/<subject>/<trial>__<sensor>.parquet``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import decimate

from ..config import IngestConfig

_IMU_SEGMENTS = ("foot", "shank", "thigh")
_IMU_AXES = ("Accel_X", "Accel_Y", "Accel_Z", "Gyro_X", "Gyro_Y", "Gyro_Z")
_FP_COLS = ("Treadmill_R_vy", "Treadmill_R_px", "Treadmill_R_pz")
_GON_COLS = ("ankle_sagittal", "knee_sagittal", "hip_sagittal")


@dataclass
class Trial:
    """One trial's features and target, aligned at the target rate."""

    subject: str
    name: str
    features: pd.DataFrame  # physical units, includes a binary ``stance`` column
    target: pd.DataFrame    # ankle moment in N·m, NaN outside valid inverse-dynamics

    @property
    def n_samples(self) -> int:
        return len(self.features)


def _decimate(x: np.ndarray, factor: int, antialias: bool) -> np.ndarray:
    if factor == 1:
        return x
    if not antialias:
        return x[::factor]
    if x.ndim == 1:
        return decimate(x, factor, ftype="iir", zero_phase=True)
    return np.stack(
        [decimate(x[:, c], factor, ftype="iir", zero_phase=True) for c in range(x.shape[1])],
        axis=1,
    )


def _stance_from_gait_phase(heel_strike: np.ndarray, toe_off: np.ndarray) -> np.ndarray:
    """Binary stance from the gcRight gait-phase signals.

    Both inputs are percent-of-cycle (0..100) that ramp then wrap; a wrap marks
    the event. Stance runs from each heel strike to the next toe off.
    """
    hs = np.where(np.diff(heel_strike) < -50.0)[0] + 1
    to = np.where(np.diff(toe_off) < -50.0)[0] + 1
    stance = np.zeros(len(heel_strike), dtype=np.float32)
    for start in hs:
        after = to[to > start]
        end = int(after[0]) if len(after) else len(stance)
        stance[start:end] = 1.0
    return stance


class RawTrialReader:
    """Reads and aligns trials from the Parquet dataset layout."""

    def __init__(self, dataset_root: str | Path, cfg: IngestConfig):
        self.root = Path(dataset_root)
        self.cfg = cfg
        self._metadata: pd.DataFrame | None = None

    @property
    def metadata(self) -> pd.DataFrame:
        if self._metadata is None:
            self._metadata = pd.read_parquet(self.root / "metadata.parquet")
        return self._metadata

    def subjects(self) -> list[str]:
        return sorted(self.metadata["subject"].unique())

    def trials(self, subject: str) -> list[str]:
        rows = self.metadata[self.metadata["subject"] == subject]
        return sorted(rows["trial"].unique())

    def read(self, subject: str, trial: str) -> Trial | None:
        imu_df = self._sensor(subject, trial, "imu")
        id_df = self._sensor(subject, trial, "id")
        if imu_df is None or id_df is None:
            return None

        factors = {s: self.cfg.native_rates_hz[s] // self.cfg.target_rate_hz
                   for s in ("imu", "id", "fp", "gon")}

        imu = self._decimate_imu(imu_df, factors["imu"])
        target = self._decimate_target(id_df, factors["id"])
        fp = self._decimate_optional(self._sensor(subject, trial, "fp"), _FP_COLS,
                                     factors["fp"], "fp_")
        gon = self._decimate_optional(self._sensor(subject, trial, "gon"), _GON_COLS,
                                      factors["gon"], "gon_", to_radians=self.cfg.gon_to_radians)
        stance = self._decimate_stance(self._sensor(subject, trial, "gcRight"), factors["id"])

        frames = {**imu, **fp, **gon}
        n = min([len(v) for v in frames.values()] + [len(target)])
        if stance is None:
            stance = self._stance_from_force(frames, subject, trial)
        n = min(n, len(stance))

        features = pd.DataFrame({k: v[:n] for k, v in frames.items()})
        features["stance"] = stance[:n]
        return Trial(
            subject=subject,
            name=f"{subject}_{trial}",
            features=features,
            target=pd.DataFrame({self.cfg.target_column: target[:n]}),
        )

    # -- io ---------------------------------------------------------
    def _sensor(self, subject: str, trial: str, sensor: str) -> pd.DataFrame | None:
        path = self.root / "subjects" / subject / f"{trial}__{sensor}.parquet"
        return pd.read_parquet(path) if path.exists() else None

    # -- per-stream decimation -----------------------------------
    def _decimate_imu(self, df: pd.DataFrame, factor: int) -> dict[str, np.ndarray]:
        dropped = set(self.cfg.drop_imu_segments)
        cols, names = [], []
        for seg in _IMU_SEGMENTS:
            if seg in dropped:
                continue
            for ax in _IMU_AXES:
                col = f"{seg}_{ax}"
                if col in df.columns:
                    cols.append(col)
                    names.append(f"imu_{seg}_{ax}")
        arr = _decimate(df[cols].to_numpy(np.float64), factor, self.cfg.antialias)
        return {name: arr[:, i] for i, name in enumerate(names)}

    def _decimate_target(self, df: pd.DataFrame, factor: int) -> np.ndarray:
        raw = df[self.cfg.target_column].to_numpy(np.float64)
        nan_mask = np.isnan(raw)
        values = _decimate(np.where(nan_mask, 0.0, raw), factor, self.cfg.antialias)
        mask = _decimate(nan_mask.astype(np.float64), factor, antialias=False) > 0.5
        return np.where(mask, np.nan, values)

    def _decimate_optional(self, df: pd.DataFrame | None, wanted: tuple[str, ...],
                           factor: int, prefix: str, to_radians: bool = False) -> dict[str, np.ndarray]:
        if df is None:
            return {}
        cols = [c for c in wanted if c in df.columns]
        arr = _decimate(df[cols].to_numpy(np.float64), factor, self.cfg.antialias)
        if to_radians:
            arr = np.radians(arr)
        return {f"{prefix}{c}": arr[:, i] for i, c in enumerate(cols)}

    def _decimate_stance(self, df: pd.DataFrame | None, factor: int) -> np.ndarray | None:
        if self.cfg.stance_source != "gcRight" or df is None:
            return None
        stance = _stance_from_gait_phase(
            df["HeelStrike"].to_numpy(np.float64), df["ToeOff"].to_numpy(np.float64))
        return (_decimate(stance, factor, antialias=False) > 0.5).astype(np.float32)

    def _stance_from_force(self, frames: dict[str, np.ndarray], subject: str,
                           trial: str) -> np.ndarray:
        vgrf = frames.get("fp_Treadmill_R_vy")
        if vgrf is None:
            raise ValueError(
                f"{subject}/{trial}: stance_source={self.cfg.stance_source} but no "
                f"gcRight and no fp_Treadmill_R_vy available"
            )
        return (vgrf > self.cfg.force_threshold_n).astype(np.float32)
