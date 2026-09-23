#!/usr/bin/env python3
"""
AS5600 magnetic rotary encoder test — I2C bus 7, address 0x36.

Reads the 12-bit angle, AGC value, and magnet status from the AS5600 and
prints them in a live loop.

AS5600 key registers:
  0x0B  STATUS  — magnet detection flags
  0x0C  RAWANGLE (MSB)
  0x0D  RAWANGLE (LSB)
  0x0E  ANGLE   (MSB)  — filtered / zero-corrected angle
  0x0F  ANGLE   (LSB)
  0x1A  AGC     — automatic gain control (0–255; ideal ~128)
  0x1B  MAGNITUDE (MSB)
  0x1C  MAGNITUDE (LSB)

Press Ctrl+C to stop.
"""

import smbus2
import time
import struct

# ── Hardware ──────────────────────────────────────────────────────────────────
I2C_BUS    = 7
AS5600_ADDR = 0x36

# ── Register addresses ────────────────────────────────────────────────────────
REG_STATUS    = 0x0B
REG_RAW_ANGLE = 0x0C   # 2 bytes, big-endian 12-bit
REG_ANGLE     = 0x0E   # 2 bytes, big-endian 12-bit (zero-corrected)
REG_AGC       = 0x1A   # 1 byte
REG_MAGNITUDE = 0x1B   # 2 bytes, big-endian 12-bit

# ── STATUS bit masks ──────────────────────────────────────────────────────────
STATUS_MH = 0x08   # Magnet too strong
STATUS_ML = 0x10   # Magnet too weak
STATUS_MD = 0x20   # Magnet detected

# ── Scale ─────────────────────────────────────────────────────────────────────
COUNTS_PER_REV  = 4096         # 12-bit encoder
DEGREES_PER_REV = 360.0
COUNTS_TO_DEG   = DEGREES_PER_REV / COUNTS_PER_REV

# ── Sample rate ───────────────────────────────────────────────────────────────
SAMPLE_HZ = 20

# ── Velocity estimation ───────────────────────────────────────────────────────
# Tracks full rotations to compute continuous position (unwrapped angle)
class VelocityEstimator:
    def __init__(self):
        self._prev_angle = None
        self._prev_time  = None
        self._turns      = 0.0

    def update(self, angle_deg: float, now: float):
        """
        Returns (unwrapped_angle_deg, velocity_deg_per_s).
        unwrapped_angle_deg accumulates across full rotations.
        """
        if self._prev_angle is None:
            self._prev_angle = angle_deg
            self._prev_time  = now
            return angle_deg, 0.0

        dt = now - self._prev_time
        delta = angle_deg - self._prev_angle

        # Wrap delta to [-180, +180] to detect direction across 0/360 boundary
        if delta > 180.0:
            delta -= 360.0
        elif delta < -180.0:
            delta += 360.0

        self._turns      += delta
        self._prev_angle  = angle_deg
        self._prev_time   = now

        velocity = delta / dt if dt > 0 else 0.0
        return self._turns, velocity


def read_word(bus: smbus2.SMBus, reg: int) -> int:
    """Read a big-endian 12-bit value from two consecutive registers."""
    data = bus.read_i2c_block_data(AS5600_ADDR, reg, 2)
    return ((data[0] << 8) | data[1]) & 0x0FFF


def decode_status(status_byte: int) -> str:
    if not (status_byte & STATUS_MD):
        return "NO MAGNET"
    if status_byte & STATUS_MH:
        return "TOO STRONG"
    if status_byte & STATUS_ML:
        return "TOO WEAK  "
    return "OK        "


def main():
    interval = 1.0 / SAMPLE_HZ
    vel      = VelocityEstimator()

    print(f"AS5600 encoder test — bus {I2C_BUS}, addr 0x{AS5600_ADDR:02X}")
    print(f"12-bit resolution ({COUNTS_PER_REV} counts/rev) | {SAMPLE_HZ} Hz")
    print()
    print(f"{'Status':<12}  {'Raw':<6}  {'Angle (°)':<11}  {'Unwrapped (°)':<15}  {'Vel (°/s)':<11}  {'AGC':<5}  {'Magnitude'}")
    print("-" * 80)

    with smbus2.SMBus(I2C_BUS) as bus:
        try:
            while True:
                t0  = time.monotonic()
                now = time.time()

                status_byte = bus.read_byte_data(AS5600_ADDR, REG_STATUS)
                raw_counts  = read_word(bus, REG_RAW_ANGLE)
                angle_counts = read_word(bus, REG_ANGLE)
                agc         = bus.read_byte_data(AS5600_ADDR, REG_AGC)
                magnitude   = read_word(bus, REG_MAGNITUDE)

                angle_deg   = angle_counts * COUNTS_TO_DEG
                status_str  = decode_status(status_byte)
                unwrapped, velocity = vel.update(angle_deg, now)

                print(
                    f"{status_str:<12}  "
                    f"{raw_counts:<6d}  "
                    f"{angle_deg:>10.3f}°  "
                    f"{unwrapped:>14.2f}°  "
                    f"{velocity:>10.1f}   "
                    f"{agc:<5d}  "
                    f"{magnitude}"
                )

                elapsed = time.monotonic() - t0
                sleep_time = interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
