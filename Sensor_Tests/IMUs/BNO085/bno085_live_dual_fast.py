#!/usr/bin/env python3
"""
Dual BNO085 FAST reader — Bus 7, Addresses 0x4A (IMU-A) and 0x4B (IMU-B).
Target: 500 Hz aggregate (≥250 Hz per sensor).

── Why this is faster than bno085_live_dual.py ──────────────────────────────
The standard IMUReader.read() reads three properties:
    bno.acceleration    →  calls _process_available_packets()  (1 I2C read)
    bno.gyro            →  calls _process_available_packets()  (1 I2C read)
    bno.game_quaternion →  calls _process_available_packets()  (1 I2C read)
= 3 I2C reads per sensor × 2 sensors = 6 I2C reads per output sample.

FastIMUReader.read() calls _process_available_packets() ONCE then reads the
library's private cached attributes directly:
    bno._process_available_packets()   (1 I2C read — populates all caches)
    bno._acceleration                  (cache lookup, zero I2C)
    bno._gyro                          (cache lookup, zero I2C)
    bno._game_quaternion               (cache lookup, zero I2C)
= 1 I2C read per sensor × 2 sensors = 2 I2C reads per output sample → 3×.

Display is decoupled: Hz counter updates every 0.5 s via a rolling window;
the data loop itself never sleeps.

──────────────────────────────────────────────────────────────────────────────
STANDALONE:
    python3 bno085_live_dual_fast.py

IMPORT INTO OTHER SCRIPTS:
    from bno085_live_dual_fast import FastDualIMUReader, quat_to_euler

    with FastDualIMUReader() as dual:
        for sa, sb in dual.stream():
            # sa / sb: dict{"quat","accel","gyro"} or None
            if sa:
                qi, qj, qk, qr = sa["quat"]
            if sb:
                qi, qj, qk, qr = sb["quat"]
──────────────────────────────────────────────────────────────────────────────
"""

import sys
import time
import warnings
from collections import deque

warnings.filterwarnings("ignore", category=RuntimeWarning,
                        message="I2C frequency is not settable")

from adafruit_extended_bus import ExtendedI2C as I2C
from adafruit_bno08x.i2c import BNO08X_I2C
from adafruit_bno08x import (
    BNO_REPORT_ACCELEROMETER,
    BNO_REPORT_GYROSCOPE,
    BNO_REPORT_GAME_ROTATION_VECTOR,
)
from bno085_live import quat_to_euler

# Report ID keys used to read directly from _readings cache
_ACC  = BNO_REPORT_ACCELEROMETER
_GYRO = BNO_REPORT_GYROSCOPE
_QUAT = BNO_REPORT_GAME_ROTATION_VECTOR

# ── Hardware ──────────────────────────────────────────────────────────────────
I2C_BUS         = 7
ADDR_A          = 0x4A
ADDR_B          = 0x4B
RECONNECT_DELAY = 1.0
BOOT_DELAY      = 0.8
FEATURE_RETRIES = 5


# ── FastIMUReader ─────────────────────────────────────────────────────────────
class FastIMUReader:
    """
    Single-sensor reader optimised for speed.
    One _process_available_packets() call per read instead of three,
    reducing I2C traffic by 3× versus the standard IMUReader.
    """

    def __init__(self, bus: int, address: int, label: str = "IMU"):
        self._bus     = bus
        self._address = address
        self._label   = label
        self._i2c     = None
        self._bno     = None
        self._connect()

    # ── init / reconnect ──────────────────────────────────────────────────────
    def _connect(self):
        while True:
            try:
                if self._i2c is not None:
                    try:
                        self._i2c.deinit()
                    except Exception:
                        pass
                    time.sleep(0.3)

                self._i2c = I2C(self._bus)
                self._bno = BNO08X_I2C(self._i2c, address=self._address)
                time.sleep(BOOT_DELAY)

                for feature in (BNO_REPORT_GAME_ROTATION_VECTOR,
                                BNO_REPORT_ACCELEROMETER,
                                BNO_REPORT_GYROSCOPE):
                    for attempt in range(FEATURE_RETRIES):
                        try:
                            self._bno.enable_feature(feature)
                            break
                        except Exception:
                            if attempt == FEATURE_RETRIES - 1:
                                raise
                            time.sleep(0.2)
                return

            except KeyboardInterrupt:
                raise
            except Exception as exc:
                print(f"[{self._label}] Connect failed ({exc}), "
                      f"retrying in {RECONNECT_DELAY}s...", flush=True)
                time.sleep(RECONNECT_DELAY)

    def _reconnect(self):
        print(f"\n[{self._label}] Reconnecting...", flush=True)
        self._connect()
        print(f"[{self._label}] Reconnected.\n", flush=True)

    # ── context manager ───────────────────────────────────────────────────────
    def __enter__(self):
        return self

    def __exit__(self, *_):
        try:
            self._i2c.deinit()
        except Exception:
            pass

    # ── fast read ─────────────────────────────────────────────────────────────
    def read(self):
        """
        Single I2C read per call.
        Calls _process_available_packets() ONCE to drain all pending SHTP
        packets into the _readings cache, then reads accel/gyro/quat from
        that cache directly — no further I2C transactions.
        Returns dict{"quat","accel","gyro"} or None if no new data.
        """
        try:
            # One I2C burst drains all pending packets → fills _readings dict
            self._bno._process_available_packets()
            accel = self._bno._readings.get(_ACC)
            gyro  = self._bno._readings.get(_GYRO)
            quat  = self._bno._readings.get(_QUAT)
        except (OSError, RuntimeError, AttributeError, KeyError) as exc:
            print(f"\n[{self._label}] Read error ({type(exc).__name__}: {exc}), "
                  f"reconnecting...", flush=True)
            self._reconnect()
            return None

        if accel is None or gyro is None or quat is None:
            return None
        return {"quat": quat, "accel": accel, "gyro": gyro}


# ── FastDualIMUReader ─────────────────────────────────────────────────────────
class FastDualIMUReader:
    """
    Two FastIMUReaders polled sequentially — 2 I2C reads per cycle total.

    Example
    -------
    with FastDualIMUReader() as dual:
        for sa, sb in dual.stream():
            if sa:
                qi, qj, qk, qr = sa["quat"]
    """

    def __init__(self,
                 bus:    int = I2C_BUS,
                 addr_a: int = ADDR_A,
                 addr_b: int = ADDR_B):
        print(f"[Dual-Fast] Connecting IMU-A (bus {bus}, 0x{addr_a:02X})...")
        self._imu_a = FastIMUReader(bus=bus, address=addr_a, label="IMU-A")
        print(f"[Dual-Fast] IMU-A connected.")
        print(f"[Dual-Fast] Connecting IMU-B (bus {bus}, 0x{addr_b:02X})...")
        self._imu_b = FastIMUReader(bus=bus, address=addr_b, label="IMU-B")
        print(f"[Dual-Fast] IMU-B connected.")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        for imu in (self._imu_a, self._imu_b):
            try:
                imu._i2c.deinit()
            except Exception:
                pass

    def read(self):
        """Returns (sample_a, sample_b) — each dict or None."""
        return self._imu_a.read(), self._imu_b.read()

    def stream(self):
        """
        Infinite generator — no sleep, no rate cap.
        Yields (sample_a, sample_b) whenever at least one sensor has data.
        """
        while True:
            sa, sb = self.read()
            if sa is not None or sb is not None:
                yield sa, sb


# ── Standalone entry point ────────────────────────────────────────────────────
def main():
    try:
        dual = FastDualIMUReader()
    except KeyboardInterrupt:
        print("\nAborted during init.")
        return

    print("\nStreaming — display updates every 0.25s, data captured at max rate.\n")
    print(f"  {'Hz-A':>6} {'Hz-B':>6}  |  "
          f"{'Roll-A':>8}{'Pitch-A':>9}{'Yaw-A':>8}  |  "
          f"{'Roll-B':>8}{'Pitch-B':>9}{'Yaw-B':>8}")
    print("-" * 90)

    # Rolling Hz windows — track timestamps of last N samples per sensor
    WIN   = 200
    ts_a  = deque(maxlen=WIN)
    ts_b  = deque(maxlen=WIN)

    last_display = time.perf_counter()
    DISPLAY_INTERVAL = 0.25   # seconds between screen refreshes

    # Last valid data for display (keep showing even if one sensor misses)
    last_a: dict | None = None
    last_b: dict | None = None

    count_a = count_b = 0
    t_start = time.perf_counter()

    try:
        with dual:
            for sa, sb in dual.stream():
                now = time.perf_counter()

                if sa is not None:
                    ts_a.append(now)
                    last_a = sa
                    count_a += 1
                if sb is not None:
                    ts_b.append(now)
                    last_b = sb
                    count_b += 1

                # Refresh display at a fixed visual rate — keeps the loop fast
                if now - last_display >= DISPLAY_INTERVAL:
                    last_display = now

                    def hz(ts):
                        if len(ts) < 2:
                            return 0.0
                        span = ts[-1] - ts[0]
                        return (len(ts) - 1) / span if span > 0 else 0.0

                    def fmt_rpy(s):
                        if s is None:
                            return f"{'---':>8}{'---':>9}{'---':>8}"
                        qi, qj, qk, qr = s["quat"]
                        r, p, y = quat_to_euler(qi, qj, qk, qr)
                        return f"{r:8.2f}{p:9.2f}{y:8.2f}"

                    ha, hb = hz(ts_a), hz(ts_b)
                    sys.stdout.write(
                        f"\r  {ha:6.1f} {hb:6.1f}  |  "
                        f"{fmt_rpy(last_a)}  |  "
                        f"{fmt_rpy(last_b)}   "
                    )
                    sys.stdout.flush()

    except KeyboardInterrupt:
        elapsed = time.perf_counter() - t_start
        total   = count_a + count_b
        print(f"\n\nStopped after {elapsed:.1f}s")
        print(f"  IMU-A: {count_a} samples  ({count_a/elapsed:.1f} Hz)")
        print(f"  IMU-B: {count_b} samples  ({count_b/elapsed:.1f} Hz)")
        print(f"  Total: {total} samples  ({total/elapsed:.1f} Hz combined)")


if __name__ == "__main__":
    main()
