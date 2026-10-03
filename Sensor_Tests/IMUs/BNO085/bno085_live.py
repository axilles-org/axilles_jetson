#!/usr/bin/env python3
"""
BNO085 IMU baseline reader — Bus 7, Address 0x4A.

Provides Game Rotation Vector (quaternion), Accelerometer, and Gyroscope
at the fastest rate the sensor can deliver (no sleep, hardware-limited).
Auto-reconnects on any I2C / SHTP error so the stream never dies.

──────────────────────────────────────────────────────────────────────────────
STANDALONE:
    python3 bno085_live.py

IMPORT INTO OTHER SCRIPTS:
    from bno085_live import IMUReader, quat_to_euler

    with IMUReader() as imu:
        for sample in imu.stream():
            qi, qj, qk, qr = sample["quat"]   # quaternion (i, j, k, real)
            ax, ay, az      = sample["accel"]  # m/s²
            gx, gy, gz      = sample["gyro"]   # rad/s

    # Or single non-blocking read (returns None if no new data):
    with IMUReader() as imu:
        sample = imu.read()
──────────────────────────────────────────────────────────────────────────────
Why ExtendedI2C and not busio.I2C?
  busio.I2C uses board-pin names (SCL_1/SDA_1) that don't resolve correctly
  on Jetson Orin Nano.  ExtendedI2C(7) directly opens /dev/i2c-7 — reliable
  on any Linux system.
"""

import math
import time
import warnings

# Suppress the harmless "I2C frequency is not settable" Blinka warning
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        message="I2C frequency is not settable")

from adafruit_extended_bus import ExtendedI2C as I2C
from adafruit_bno08x.i2c import BNO08X_I2C
from adafruit_bno08x import (
    BNO_REPORT_ACCELEROMETER,
    BNO_REPORT_GYROSCOPE,
    BNO_REPORT_GAME_ROTATION_VECTOR,
)

# ── Hardware ──────────────────────────────────────────────────────────────────
I2C_BUS         = 7
I2C_ADDR        = 0x4A
RECONNECT_DELAY = 1.0   # seconds to wait between full reconnect attempts
BOOT_DELAY      = 0.8   # seconds to wait after BNO08X_I2C init before enabling features
FEATURE_RETRIES = 5     # per-feature enable retry attempts


# ── Math helper ───────────────────────────────────────────────────────────────
def quat_to_euler(qi, qj, qk, qr):
    """Convert quaternion (i, j, k, real) → (roll, pitch, yaw) in degrees."""
    sinr = 2.0 * (qr * qi + qj * qk)
    cosr = 1.0 - 2.0 * (qi * qi + qj * qj)
    roll  = math.degrees(math.atan2(sinr, cosr))

    sinp  = max(-1.0, min(1.0, 2.0 * (qr * qj - qk * qi)))
    pitch = math.degrees(math.asin(sinp))

    siny = 2.0 * (qr * qk + qi * qj)
    cosy = 1.0 - 2.0 * (qj * qj + qk * qk)
    yaw  = math.degrees(math.atan2(siny, cosy))

    return roll, pitch, yaw


# ── IMUReader class (importable) ──────────────────────────────────────────────
class IMUReader:
    """
    Context-manager wrapper around BNO085.
    Automatically reconnects on I2C/SHTP errors — the stream never stops
    unless you press Ctrl+C.

    Example
    -------
    with IMUReader() as imu:
        for sample in imu.stream():
            qi, qj, qk, qr = sample["quat"]
    """

    def __init__(self, bus: int = I2C_BUS, address: int = I2C_ADDR):
        self._bus     = bus
        self._address = address
        self._i2c     = None
        self._bno     = None
        self._connect()

    # ── internal helpers ──────────────────────────────────────────────────────
    def _connect(self):
        """Open I2C bus and initialise sensor. Retries until successful."""
        while True:
            try:
                # Clean up any previous bus handle
                if self._i2c is not None:
                    try:
                        self._i2c.deinit()
                    except Exception:
                        pass
                    time.sleep(0.3)  # let the bus settle before reopening

                self._i2c = I2C(self._bus)
                self._bno = BNO08X_I2C(self._i2c, address=self._address)

                # Wait for sensor firmware to finish booting before enabling features.
                # Skipping this causes "Was not able to enable feature" (RuntimeError).
                time.sleep(BOOT_DELAY)

                # Enable each feature with individual retry + backoff
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

                return  # success

            except KeyboardInterrupt:
                raise  # let Ctrl+C propagate immediately — never swallow it
            except Exception as exc:
                print(f"[IMU] Connect failed ({exc}), retrying in "
                      f"{RECONNECT_DELAY}s...", flush=True)
                time.sleep(RECONNECT_DELAY)

    def _reconnect(self):
        print(f"\n[IMU] Reconnecting...", flush=True)
        self._connect()
        print("[IMU] Reconnected.\n", flush=True)

    # ── public API ────────────────────────────────────────────────────────────
    def __enter__(self):
        return self

    def __exit__(self, *_):
        try:
            self._i2c.deinit()
        except Exception:
            pass

    def read(self):
        """
        Non-blocking single read.
        Returns a dict {"quat", "accel", "gyro"} or None if no new data yet.
        On I2C/SHTP error, reconnects automatically and returns None.
        """
        try:
            accel = self._bno.acceleration    # (x, y, z) m/s²
            gyro  = self._bno.gyro            # (x, y, z) rad/s
            quat  = self._bno.game_quaternion # (i, j, k, real)
        except (OSError, RuntimeError, AttributeError, KeyError) as exc:
            # KeyError: 0 happens when the sensor spontaneously reboots and
            # sends a raw SHTP command-channel packet (report_id=0) that the
            # Adafruit library's _report_length() doesn't recognise.
            print(f"\n[IMU] Read error ({type(exc).__name__}: {exc}), reconnecting...",
                  flush=True)
            self._reconnect()
            return None

        if accel is None or gyro is None or quat is None:
            return None
        return {"quat": quat, "accel": accel, "gyro": gyro}

    def stream(self):
        """
        Infinite generator — yields every valid sample as fast as the sensor
        delivers it (no sleep, hardware-rate limited).
        Survives I2C glitches and SHTP resets via auto-reconnect.
        """
        while True:
            sample = self.read()
            if sample is not None:
                yield sample


# ── Standalone entry point ────────────────────────────────────────────────────
def main():
    print("Connecting to BNO085 (I2C bus 7, 0x4A)...")
    try:
        imu = IMUReader()
    except KeyboardInterrupt:
        print("\nAborted during init.")
        return

    print("Connected. Streaming at max rate — Ctrl+C to stop.\n")
    print(f"{'Roll':>9}{'Pitch':>9}{'Yaw':>9}  |  "
          f"{'Ax':>8}{'Ay':>8}{'Az':>8}  |  "
          f"{'Gx':>9}{'Gy':>9}{'Gz':>9}")
    print("-" * 100)

    count   = 0
    t_start = time.time()

    try:
        with imu:
            for s in imu.stream():
                qi, qj, qk, qr = s["quat"]
                ax, ay, az      = s["accel"]
                gx, gy, gz      = s["gyro"]
                roll, pitch, yaw = quat_to_euler(qi, qj, qk, qr)

                print(f"{roll:9.2f}{pitch:9.2f}{yaw:9.2f}  |  "
                      f"{ax:8.3f}{ay:8.3f}{az:8.3f}  |  "
                      f"{gx:9.4f}{gy:9.4f}{gz:9.4f}")
                count += 1

    except KeyboardInterrupt:
        elapsed = time.time() - t_start
        print(f"\nStopped. {count} samples in {elapsed:.1f}s "
              f"({count / elapsed:.1f} Hz)")


if __name__ == "__main__":
    main()
