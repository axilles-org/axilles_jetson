#!/usr/bin/env python3
"""
Dual BNO085 live reader — Bus 7, Addresses 0x4A (IMU-A) and 0x4B (IMU-B).

Both sensors are polled sequentially on the same I2C bus (single-master,
no threads required). Each sensor auto-reconnects independently on errors.

──────────────────────────────────────────────────────────────────────────────
STANDALONE:
    python3 bno_live_dual.py

IMPORT INTO OTHER SCRIPTS:
    from bno_live_dual import DualIMUReader, quat_to_euler

    with DualIMUReader() as dual:
        for sample_a, sample_b in dual.stream():
            # sample_a / sample_b are dicts with keys:
            #   "quat"  → (qi, qj, qk, qr)
            #   "accel" → (ax, ay, az)  m/s²
            #   "gyro"  → (gx, gy, gz)  rad/s
            # Either may be None if that sensor has no new data this cycle.
──────────────────────────────────────────────────────────────────────────────
"""

import time
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning,
                        message="I2C frequency is not settable")

# Re-use IMUReader and quat_to_euler from the single-sensor baseline
from bno085_live import IMUReader, quat_to_euler

# ── Hardware ──────────────────────────────────────────────────────────────────
I2C_BUS   = 7
ADDR_A    = 0x4A   # IMU-A (e.g. left leg / thigh)
ADDR_B    = 0x4B   # IMU-B (e.g. right leg / shank)


# ── DualIMUReader ─────────────────────────────────────────────────────────────
class DualIMUReader:
    """
    Manages two BNO085 sensors on the same I2C bus, different addresses.
    Each reconnects independently — a glitch on one doesn't affect the other.

    Example
    -------
    from bno_live_dual import DualIMUReader, quat_to_euler

    with DualIMUReader() as dual:
        for sa, sb in dual.stream():
            if sa:
                qi, qj, qk, qr = sa["quat"]
            if sb:
                qi, qj, qk, qr = sb["quat"]
    """

    def __init__(self,
                 bus:     int = I2C_BUS,
                 addr_a:  int = ADDR_A,
                 addr_b:  int = ADDR_B):
        print(f"[Dual] Connecting IMU-A (bus {bus}, 0x{addr_a:02X})...")
        self._imu_a = IMUReader(bus=bus, address=addr_a)
        print(f"[Dual] IMU-A connected.")

        print(f"[Dual] Connecting IMU-B (bus {bus}, 0x{addr_b:02X})...")
        self._imu_b = IMUReader(bus=bus, address=addr_b)
        print(f"[Dual] IMU-B connected.")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        for imu in (self._imu_a, self._imu_b):
            try:
                imu._i2c.deinit()
            except Exception:
                pass

    def read(self):
        """
        Non-blocking read from both sensors.
        Returns (sample_a, sample_b) — each is a dict or None if no new data.
        """
        return self._imu_a.read(), self._imu_b.read()

    def stream(self):
        """
        Infinite generator. Yields (sample_a, sample_b) on every loop tick.
        Only yields when at least one sensor has new data.
        Both sensors auto-reconnect independently on errors.
        """
        while True:
            sa, sb = self.read()
            if sa is not None or sb is not None:
                yield sa, sb


# ── Standalone entry point ────────────────────────────────────────────────────
def main():
    try:
        dual = DualIMUReader()
    except KeyboardInterrupt:
        print("\nAborted during init.")
        return

    print("\nStreaming at max rate — Ctrl+C to stop.\n")

    # Header
    hdr = (f"{'':6}"
           f"{'Roll-A':>9}{'Pitch-A':>9}{'Yaw-A':>9}  "
           f"{'Ax-A':>8}{'Ay-A':>8}{'Az-A':>8}  "
           f"{'Gx-A':>9}{'Gy-A':>9}{'Gz-A':>9}"
           f"  ||  "
           f"{'Roll-B':>9}{'Pitch-B':>9}{'Yaw-B':>9}  "
           f"{'Ax-B':>8}{'Ay-B':>8}{'Az-B':>8}  "
           f"{'Gx-B':>9}{'Gy-B':>9}{'Gz-B':>9}")
    print(hdr)
    print("-" * len(hdr))

    count   = 0
    t_start = time.time()

    _nan = float("nan")
    _fmt_imu = (lambda r, p, y, ax, ay, az, gx, gy, gz:
                f"{r:9.2f}{p:9.2f}{y:9.2f}  "
                f"{ax:8.3f}{ay:8.3f}{az:8.3f}  "
                f"{gx:9.4f}{gy:9.4f}{gz:9.4f}")

    def unpack(s):
        if s is None:
            return (_nan,)*9
        qi, qj, qk, qr = s["quat"]
        ax, ay, az      = s["accel"]
        gx, gy, gz      = s["gyro"]
        r, p, y = quat_to_euler(qi, qj, qk, qr)
        return r, p, y, ax, ay, az, gx, gy, gz

    try:
        with dual:
            for sa, sb in dual.stream():
                tag = "AB" if (sa and sb) else ("A-" if sa else "-B")
                line_a = _fmt_imu(*unpack(sa))
                line_b = _fmt_imu(*unpack(sb))
                print(f"[{tag}]  {line_a}  ||  {line_b}")
                count += 1

    except KeyboardInterrupt:
        elapsed = time.time() - t_start
        print(f"\nStopped. {count} samples in {elapsed:.1f}s "
              f"({count / elapsed:.1f} Hz)")


if __name__ == "__main__":
    main()
