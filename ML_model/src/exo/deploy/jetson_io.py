"""Jetson sensor I/O and Teleplot.

``JetsonSensors`` needs the ``axilles_jetson`` repo + Jetson hardware (imported
lazily). ``ReplaySensors`` feeds a recorded CSV. ``Teleplot`` is pure UDP.
"""
from __future__ import annotations

import bisect
import socket
import sys
import threading
import time
from collections import deque
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


class SensorFault(RuntimeError):
    """A sensor stopped updating, or the sensor thread failed."""


class JetsonSensors:
    """Exo sensors, read continuously in a background thread.

    Foot IMU = BNO085 IMU-A (0x4A), shank IMU = IMU-B (0x4B), both on
    ``i2c_bus`` (bus 1 on the exo, same as TBE_controller); FSR + encoder from
    ``SensorData``. BNO085 accel is m/s^2 and gyro rad/s — passed through; the
    SensorAdapter's per-IMU accel_scale converts m/s^2 -> g.

    The IMUs are read with Pawan's lean reader (``pawan/bno085_lean.py``, one I2C
    read per packet) and report accel and gyro every ``imu_interval_ms``; the old
    reader left the library's 50 ms default, so the model got 20 Hz gyro. An IMU
    read error ends the run rather than reconnecting. The thread keeps time-stamped values,
    so the control loop never waits on I2C, and ``frame_at`` gives the model
    evenly spaced frames. Because a hung sensor no longer stalls the loop,
    ``read`` and ``frame_at`` raise ``SensorFault`` once any sensor is older than
    ``stale_s``, or if the thread failed.
    """

    HISTORY_S = 0.5             # frames kept for interpolation
    STARTUP_S = 2.0             # grace for the first reading of each sensor
    MAX_POLLS = 4               # packets drained from each IMU per pass

    def __init__(self, axilles_root: str, i2c_bus: int = 1,
                 imu_interval_ms: float = 10.0, stale_s: float = 0.1):
        root = Path(axilles_root).expanduser()
        if not (root / "TBE_controller" / "data_obtainer.py").is_file():
            raise FileNotFoundError(
                f"--axilles {root} is not an axilles_jetson checkout "
                "(expected TBE_controller/ and Sensor_Tests/IMUs/BNO085/)")
        sys.path.insert(0, str(root / "pawan"))
        sys.path.insert(0, str(root / "TBE_controller"))

        from bno085_lean import ACCEL, GYRO, LeanBNO085
        from data_obtainer import SensorData
        from utilities import Logger

        self._log = Logger()
        report_hz = 1000.0 / imu_interval_ms
        self._imus = [(prefix, LeanBNO085(i2c_bus, address, report_hz, (ACCEL, GYRO),
                                          label=f"IMU-{prefix}"))
                      for prefix, address in (("foot", 0x4A), ("shank", 0x4B))]
        # sensors only: no CAN, no Teleplot packets on every read
        self._fsr_enc = SensorData(self._log, use_can=False, use_teleplot=False)

        self.stale_s = stale_s
        self._frame = {k: 0.0 for k in FRAME_KEYS}
        self._updated = {"foot IMU": None, "shank IMU": None, "FSR/encoder": None}
        self._history: deque[tuple[float, dict[str, float]]] = deque()
        self._lock = threading.Lock()
        self._error: BaseException | None = None
        self._stop = threading.Event()
        self._started = time.monotonic()
        self._thread = threading.Thread(target=self._run, name="exo-sensors", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                fresh = []
                for prefix, imu in self._imus:
                    got = 0
                    for _ in range(self.MAX_POLLS):
                        n = imu.poll()
                        if not n:
                            break
                        got += n
                    if got and imu.accel is not None and imu.gyro is not None:
                        fresh.append((prefix, imu.accel, imu.gyro))
                self._fsr_enc.readSensors()
                now = time.monotonic()
                with self._lock:
                    f = self._frame
                    for prefix, accel, gyro in fresh:
                        f[f"{prefix}_ax"], f[f"{prefix}_ay"], f[f"{prefix}_az"] = accel
                        f[f"{prefix}_gx"], f[f"{prefix}_gy"], f[f"{prefix}_gz"] = gyro
                        self._updated[f"{prefix} IMU"] = now
                    f["ankle_encoder_deg"] = float(self._fsr_enc.encoder_data)
                    f["heel_fsr_raw"] = float(self._fsr_enc.filtered_heel_fsr)
                    f["toe_fsr_raw"] = float(self._fsr_enc.filtered_toe_fsr)
                    self._updated["FSR/encoder"] = now
                    self._history.append((now, dict(f)))
                    while now - self._history[0][0] > self.HISTORY_S:
                        self._history.popleft()
        except Exception as exc:                           # noqa: BLE001
            self._error = exc

    def _check(self) -> None:
        if self._error is not None:
            raise SensorFault(f"sensor thread stopped: {self._error!r}") from self._error
        now = time.monotonic()
        for name, seen in self._updated.items():
            if seen is None:
                if now - self._started > self.STARTUP_S:
                    raise SensorFault(f"no reading from the {name} after {self.STARTUP_S:.0f} s")
            elif now - seen > self.stale_s:
                raise SensorFault(f"{name} has not updated for {now - seen:.2f} s")

    def age_ms(self) -> float:
        """Age of the oldest sensor's latest reading."""
        now = time.monotonic()
        with self._lock:
            return max((now - seen) * 1000.0 if seen is not None else 0.0
                       for seen in self._updated.values())

    def read(self) -> dict[str, float]:
        """The latest reading of every sensor."""
        self._check()
        with self._lock:
            return dict(self._frame)

    def frame_at(self, t: float) -> dict[str, float]:
        """The frame at ``time.monotonic()`` time ``t``, interpolated between the
        readings either side (the latest reading if ``t`` is newer)."""
        self._check()
        with self._lock:
            history = list(self._history)
        if not history:
            return self.read()
        i = bisect.bisect_left([h[0] for h in history], t)
        if i == 0:
            return dict(history[0][1])
        if i == len(history):
            return dict(history[-1][1])
        (t0, f0), (t1, f1) = history[i - 1], history[i]
        w = (t - t0) / (t1 - t0) if t1 > t0 else 1.0
        return {k: f0[k] + w * (f1[k] - f0[k]) for k in FRAME_KEYS}

    def shutdown(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        try:
            self._fsr_enc.shutdown()
        except Exception:
            pass
        for _, imu in self._imus:
            try:
                imu.close()
            except Exception:
                pass


class ReplaySensors:
    """Feeds a recorded ``data_collection_*.csv`` frame-by-frame — for a bench
    dry-run of the powered runner (with MotorInterface in dry_run mode)."""

    def __init__(self, csv_path: str, rate_hz: float):
        import numpy as np
        import pandas as pd

        df = pd.read_csv(csv_path).rename(columns={"time_s": "timestamp_s"}).sort_values("timestamp_s")
        df = df.interpolate().ffill().bfill()
        t = df["timestamp_s"].to_numpy()
        t = t - t[0]
        grid = np.arange(0.0, t[-1], 1.0 / rate_hz)
        columns = {k: np.interp(grid, t, df[k].to_numpy()) for k in FRAME_KEYS}
        self._frames = [{k: float(columns[k][i]) for k in FRAME_KEYS} for i in range(len(grid))]
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
