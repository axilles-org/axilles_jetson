"""Jetson sensor I/O and Teleplot.

``JetsonSensors`` needs the ``axilles_jetson`` repo + Jetson hardware (imported
lazily). ``ReplaySensors`` feeds a recorded CSV. ``Teleplot`` is pure UDP.
"""
from __future__ import annotations

import socket
import sys
from pathlib import Path

# raw-frame keys the SensorAdapter expects (data_collection CSV schema)
FRAME_KEYS = [
    "foot_ax", "foot_ay", "foot_az", "foot_gx", "foot_gy", "foot_gz",
    "shank_ax", "shank_ay", "shank_az", "shank_gx", "shank_gy", "shank_gz",
    "ankle_encoder_deg", "heel_fsr_raw", "toe_fsr_raw",
]


class Teleplot:
    """Minimal Teleplot UDP sender (same protocol as the TBE repo)."""

    def __init__(self, host: str = "127.0.0.1", port: int = 47269, enabled: bool = True):
        self.addr = (host, port)
        self.enabled = enabled
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if enabled else None

    def send(self, name: str, value: float, t_ms: int) -> None:
        if not self.enabled:
            return
        try:
            self.sock.sendto(f"{name}:{t_ms}:{value}|g".encode(), self.addr)
        except OSError:
            pass

    def close(self) -> None:
        if self.sock is not None:
            self.sock.close()


class JetsonSensors:
    """One exo raw-frame dict per read, from the team's hardware modules.

    Foot IMU = BNO085 IMU-A (0x4A), shank IMU = IMU-B (0x4B); FSR + encoder from
    ``SensorData``. BNO085 accel is m/s^2 and gyro rad/s — passed through; the
    SensorAdapter's per-IMU accel_scale converts m/s^2 -> g.
    """

    def __init__(self, axilles_root: str):
        root = Path(axilles_root).expanduser()
        sys.path.insert(0, str(root / "Sensor_Tests" / "IMUs" / "BNO085"))
        sys.path.insert(0, str(root / "TBE_controller"))

        from bno085_live_dual_fast import FastDualIMUReader
        from data_obtainer import SensorData
        from utilities import Logger

        self._log = Logger()
        self._imus = FastDualIMUReader()
        self._fsr_enc = SensorData(self._log)
        self._last = {k: 0.0 for k in FRAME_KEYS}

    def read(self) -> dict[str, float]:
        sa, sb = self._imus.read()
        self._fsr_enc.readSensors()

        f = self._last
        if sa is not None:
            f["foot_ax"], f["foot_ay"], f["foot_az"] = sa["accel"]
            f["foot_gx"], f["foot_gy"], f["foot_gz"] = sa["gyro"]
        if sb is not None:
            f["shank_ax"], f["shank_ay"], f["shank_az"] = sb["accel"]
            f["shank_gx"], f["shank_gy"], f["shank_gz"] = sb["gyro"]

        f["ankle_encoder_deg"] = float(self._fsr_enc.encoder_data)
        f["heel_fsr_raw"] = float(self._fsr_enc.filtered_heel_fsr)
        f["toe_fsr_raw"] = float(self._fsr_enc.filtered_toe_fsr)
        return dict(f)

    def shutdown(self) -> None:
        # SensorData.shutdown() also stops its motor + closes CAN; harmless here.
        try:
            self._fsr_enc.shutdown()
        except Exception:
            pass


class ReplaySensors:
    """Feeds a recorded ``data_collection_*.csv`` frame-by-frame — for a bench
    dry-run of the powered runner (with MotorInterface in dry_run mode)."""

    def __init__(self, csv_path: str, rate_hz: float):
        import numpy as np
        import pandas as pd

        df = pd.read_csv(csv_path).sort_values("timestamp_s")
        df = df.interpolate().ffill().bfill()
        t = df["timestamp_s"].to_numpy()
        t = t - t[0]
        grid = np.arange(0.0, t[-1], 1.0 / rate_hz)
        self._frames = [
            {k: float(np.interp(grid, t, df[k].to_numpy())[i]) for k in FRAME_KEYS}
            for i in range(len(grid))
        ]
        self._i = 0

    def read(self) -> dict[str, float]:
        f = self._frames[min(self._i, len(self._frames) - 1)]
        self._i += 1
        return dict(f)

    @property
    def exhausted(self) -> bool:
        return self._i >= len(self._frames)

    def shutdown(self) -> None:
        pass
