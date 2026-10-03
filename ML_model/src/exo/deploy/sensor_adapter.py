"""Map the exo's raw sensor frame onto the training (GaTech) convention.

Per IMU: ``gyro = R . (s_g . gyro_exo)`` and
``accel = s_a . R . accel_exo - (w x (w x r) + wdot x r)`` — the accelerometer
lever-arm term needs ``wdot``, so the adapter keeps a one-sample history and is
stateful. Encoder deg -> rad; FSR counts -> debounced binary stance. Parameters
come from ``SensorAdapterConfig`` (scripts/fit_adapter.py). See
docs/SENSOR_ADAPTER.md.
"""
from __future__ import annotations

import numpy as np

from ..config import SensorAdapterConfig

_FOOT_ACCEL = ("foot_ax", "foot_ay", "foot_az")
_FOOT_GYRO = ("foot_gx", "foot_gy", "foot_gz")
_SHANK_ACCEL = ("shank_ax", "shank_ay", "shank_az")
_SHANK_GYRO = ("shank_gx", "shank_gy", "shank_gz")


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]],
                     [v[2], 0.0, -v[0]],
                     [-v[1], v[0], 0.0]])


class _ImuChannel:
    """One IMU's rotation + scales + lever-arm correction, with wdot history."""

    def __init__(self, rotation, accel_scale: float, gyro_scale: float,
                 lever_arm_m):
        self.R = np.asarray(rotation, dtype=np.float64)
        self.s_a = float(accel_scale)
        self.s_g = float(gyro_scale)
        self.r = np.asarray(lever_arm_m, dtype=np.float64)
        self._w_prev: np.ndarray | None = None

    def reset(self) -> None:
        self._w_prev = None

    def transform(self, accel_exo: np.ndarray, gyro_exo: np.ndarray,
                  dt: float) -> tuple[np.ndarray, np.ndarray]:
        w = self.R @ (self.s_g * gyro_exo)                 # rad/s, GaTech frame
        a = self.s_a * (self.R @ accel_exo)                # g, GaTech frame

        if np.any(self.r):
            if self._w_prev is None or dt <= 0.0:
                wdot = np.zeros(3)
            else:
                wdot = (w - self._w_prev) / dt
            a = a - (_skew(w) @ (_skew(w) @ self.r) + _skew(wdot) @ self.r)
        self._w_prev = w
        return a, w


class SensorAdapter:
    def __init__(self, cfg: SensorAdapterConfig):
        self.cfg = cfg
        self.foot = _ImuChannel(cfg.foot_rotation, cfg.foot_accel_scale,
                                cfg.foot_gyro_scale, cfg.foot_lever_arm_m)
        self.shank = _ImuChannel(cfg.shank_rotation, cfg.shank_accel_scale,
                                 cfg.shank_gyro_scale, cfg.shank_lever_arm_m)
        self._stance = 0.0
        self._stance_age = 0.0

    def reset(self) -> None:
        self.foot.reset()
        self.shank.reset()
        self._stance = 0.0
        self._stance_age = 0.0

    def transform_frame(self, raw: dict[str, float], dt: float = 0.01
                        ) -> dict[str, float]:
        """Convert one raw sample dict into GaTech-convention feature values.

        Output keys match the ingest feature names: ``imu_foot_Accel_X`` ...,
        ``imu_shank_Gyro_Z``, ``gon_ankle_sagittal`` (radians), ``stance`` (0/1).
        """
        c = self.cfg
        out: dict[str, float] = {}

        fa, fg = self.foot.transform(_vec(raw, _FOOT_ACCEL), _vec(raw, _FOOT_GYRO), dt)
        sa, sg = self.shank.transform(_vec(raw, _SHANK_ACCEL), _vec(raw, _SHANK_GYRO), dt)
        for axis, v in zip("XYZ", fa):
            out[f"imu_foot_Accel_{axis}"] = float(v)
        for axis, v in zip("XYZ", fg):
            out[f"imu_foot_Gyro_{axis}"] = float(v)
        for axis, v in zip("XYZ", sa):
            out[f"imu_shank_Accel_{axis}"] = float(v)
        for axis, v in zip("XYZ", sg):
            out[f"imu_shank_Gyro_{axis}"] = float(v)

        angle_deg = raw["ankle_encoder_deg"] - c.ankle_encoder_neutral_deg
        out["gon_ankle_sagittal"] = float(np.radians(angle_deg) * c.ankle_encoder_sign)

        out["stance"] = self._stance_flag(
            raw.get("heel_fsr_raw", 0.0), raw.get("toe_fsr_raw", 0.0), dt)
        return out

    def _stance_flag(self, heel: float, toe: float, dt: float) -> float:
        c = self.cfg
        raw = 1.0 if (heel > c.heel_fsr_threshold or toe > c.toe_fsr_threshold) else 0.0
        if raw != self._stance:
            self._stance_age += dt
            if self._stance_age >= c.fsr_debounce_s:
                self._stance = raw
                self._stance_age = 0.0
        else:
            self._stance_age = 0.0
        return self._stance

    def transform_batch(self, df):
        """Vectorised transform of a DataFrame of raw samples (uniform dt)."""
        import pandas as pd

        self.reset()
        dt = float(np.median(np.diff(df.index))) if len(df) > 1 else 0.01
        rows = [self.transform_frame(
            row._asdict() if hasattr(row, "_asdict") else dict(row), dt)
            for _, row in df.iterrows()]
        return pd.DataFrame(rows)


def _vec(raw: dict[str, float], keys) -> np.ndarray:
    return np.array([raw[k] for k in keys], dtype=np.float64)
