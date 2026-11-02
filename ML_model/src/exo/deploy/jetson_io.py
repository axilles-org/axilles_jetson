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


class TimedADS1115:
    """Toe and heel FSRs on the ADS1115, read as Pawan's
    ``pawan/all_sensors_rate_test.py`` does: single-shot at 860 SPS, AIN0 (toe)
    then AIN1 (heel). Each conversion is timed rather than polled via the ready
    bit, and ``step`` never waits, so the IMUs use the bus during conversions.
    Values are raw counts, like data_collection.py records."""

    ADDR = 0x48
    REG_CONV = 0x00
    REG_CFG = 0x01
    CFG = ([0xC3, 0xE3], [0xD3, 0xE3])     # ±4.096 V, 860 SPS, comparator off
    CONV_S = 0.0013                         # 1/860 s = 1.16 ms, +10% clock tolerance

    def __init__(self, bus, sample_hz: float):
        self._bus = bus
        self._period = 1.0 / sample_hz
        self._next = 0.0
        self._ch: int | None = None         # channel converting, None when idle
        self._t_trigger = 0.0
        self._raw = [0, 0]

    def step(self, now: float) -> tuple[int, int] | None:
        """Start a sample if one is due, or read a finished conversion and start
        the next channel. Returns (toe, heel) when both channels are new."""
        if self._ch is None:
            if now >= self._next:
                self._next = max(self._next + self._period, now)
                self._trigger(0)
            return None
        if now - self._t_trigger < self.CONV_S:
            return None
        d = self._bus.read_i2c_block_data(self.ADDR, self.REG_CONV, 2)
        raw = (d[0] << 8) | d[1]
        self._raw[self._ch] = raw - 0x10000 if raw >= 0x8000 else raw
        if self._ch == 0:
            self._trigger(1)
            return None
        self._ch = None
        return self._raw[0], self._raw[1]

    def _trigger(self, ch: int) -> None:
        self._bus.write_i2c_block_data(self.ADDR, self.REG_CFG, self.CFG[ch])
        self._ch = ch
        self._t_trigger = time.perf_counter()


class JetsonSensors:
    """One exo raw-frame dict per read, from the team's hardware modules.

    Foot IMU = BNO085 IMU-A (0x4A), shank IMU = IMU-B (0x4B), both on
    ``i2c_bus`` (bus 1 on the exo, same as TBE_controller); FSRs from
    ``TimedADS1115`` at ``fsr_hz``, encoder from ``SensorData``. BNO085 accel is
    m/s^2 and gyro rad/s — passed through; the SensorAdapter's per-IMU
    accel_scale converts m/s^2 -> g. The ADC is stepped after every IMU poll, so
    a finished conversion waits at most one IMU read.

    The IMUs are read with Pawan's lean reader (``pawan/bno085_lean.py``, one I2C
    read per packet) and report accel and gyro every ``imu_interval_ms``; the old
    reader left the library's 50 ms default, so the model got 20 Hz gyro. The
    thread keeps time-stamped values, so the control loop never waits on I2C, and
    ``frame_at`` gives the model evenly spaced frames.

    Sensor problems do not end the run. A failed read (I2C error, corrupt BNO085
    packet) is counted, printed at most once a second per sensor, and skipped; a
    sensor older than ``stale_s`` is printed when it goes stale and when it
    recovers, and its last value is kept. ``read`` and ``frame_at`` raise
    ``SensorFault`` only if the sensor thread itself died, since every value
    would then stay frozen.
    """

    HISTORY_S = 0.5             # frames kept for interpolation
    STARTUP_S = 2.0             # grace for the first reading of each sensor
    MAX_POLLS = 4               # packets drained from each IMU per pass
    # what a bad bus read raises: OSError from I2C, KeyError/RuntimeError/
    # IndexError from the BNO085 parser on a corrupt packet
    READ_ERRORS = (OSError, KeyError, RuntimeError, IndexError)

    def __init__(self, axilles_root: str, i2c_bus: int = 1,
                 imu_interval_ms: float = 10.0, stale_s: float = 0.1,
                 fsr_hz: float = 200.0):
        root = Path(axilles_root).expanduser()
        if not (root / "TBE_controller" / "data_obtainer.py").is_file():
            raise FileNotFoundError(
                f"--axilles {root} is not an axilles_jetson checkout "
                "(expected TBE_controller/ and pawan/)")
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
        # encoder only: no CAN, no Teleplot packets on every read
        self._enc = SensorData(self._log, use_can=False, use_teleplot=False)
        self._fsr = TimedADS1115(self._enc.bus, fsr_hz)

        self.stale_s = stale_s
        self._frame = {k: 0.0 for k in FRAME_KEYS}
        self._updated = {"foot IMU": None, "shank IMU": None, "FSR": None, "encoder": None}
        self._history: deque[tuple[float, dict[str, float]]] = deque()
        self._lock = threading.Lock()
        self._error: BaseException | None = None
        self.read_errors: dict[str, int] = {}
        self._error_printed: dict[str, float] = {}
        self._stale: set[str] = set()
        self._stop = threading.Event()
        self._started = time.monotonic()
        self._thread = threading.Thread(target=self._run, name="exo-sensors", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                fresh = []
                fsr = self._step_fsr()
                for prefix, imu in self._imus:
                    got = 0
                    for _ in range(self.MAX_POLLS):
                        try:
                            n = imu.poll()
                        except self.READ_ERRORS as exc:
                            self._read_error(f"{prefix} IMU", exc)
                            n = 0
                        fsr = self._step_fsr() or fsr
                        if not n:
                            break
                        got += n
                    if got and imu.accel is not None and imu.gyro is not None:
                        fresh.append((prefix, imu.accel, imu.gyro))
                self._enc.readEncoder()
                now = time.monotonic()
                with self._lock:
                    f = self._frame
                    for prefix, accel, gyro in fresh:
                        f[f"{prefix}_ax"], f[f"{prefix}_ay"], f[f"{prefix}_az"] = accel
                        f[f"{prefix}_gx"], f[f"{prefix}_gy"], f[f"{prefix}_gz"] = gyro
                        self._updated[f"{prefix} IMU"] = now
                    if fsr is not None:
                        f["toe_fsr_raw"], f["heel_fsr_raw"] = float(fsr[0]), float(fsr[1])
                        self._updated["FSR"] = now
                    f["ankle_encoder_deg"] = float(self._enc.encoder_data)
                    self._updated["encoder"] = now
                    self._history.append((now, dict(f)))
                    while now - self._history[0][0] > self.HISTORY_S:
                        self._history.popleft()
        except Exception as exc:                           # noqa: BLE001
            self._error = exc

    def _step_fsr(self) -> tuple[int, int] | None:
        try:
            return self._fsr.step(time.perf_counter())
        except OSError as exc:
            self._read_error("FSR", exc)
            return None

    def _read_error(self, name: str, exc: BaseException) -> None:
        """Count a failed read and carry on; printed at most once a second per sensor."""
        self.read_errors[name] = self.read_errors.get(name, 0) + 1
        now = time.monotonic()
        if now - self._error_printed.get(name, float("-inf")) >= 1.0:
            self._error_printed[name] = now
            print(f"!! {name} read error #{self.read_errors[name]}: {exc!r} (run continues)")

    def _check(self) -> None:
        if self._error is not None:
            raise SensorFault(f"sensor thread stopped: {self._error!r}") from self._error
        now = time.monotonic()
        for name, seen in self._updated.items():
            age = now - (self._started if seen is None else seen)
            stale = age > (self.STARTUP_S if seen is None else self.stale_s)
            if stale and name not in self._stale:
                self._stale.add(name)
                print(f"!! {name} has not updated for {age:.2f} s (run continues)")
            elif not stale and name in self._stale:
                self._stale.discard(name)
                print(f"{name} updating again")

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
            self._enc.shutdown()
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
