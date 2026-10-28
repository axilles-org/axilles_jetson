#!/usr/bin/env python3
"""
AS5600 magnetic rotary encoder test — I2C bus 1, address 0x36.

Reads the 12-bit angle, magnet status, AGC and magnitude from the AS5600 and
prints them in a live loop.  Read-only: nothing is ever written to the chip
(no ZPOS/MPOS writes, no OTP burn), so the zero offset below is software only.

Usage
-----
    python3 test_as5600.py                # bus 1, 0x36, 20 Hz
    python3 test_as5600.py --hz 100       # faster print rate
    python3 test_as5600.py --zero         # current position becomes 0°
    python3 test_as5600.py --bus 7        # old wiring

AS5600 key registers (contiguous blocks are read in one transaction):
  0x0B  STATUS     — magnet detection flags   ┐
  0x0C  RAWANGLE   (MSB)                       │ one 5-byte read
  0x0D  RAWANGLE   (LSB)                       │
  0x0E  ANGLE      (MSB) — zero/range-scaled   │
  0x0F  ANGLE      (LSB)                       ┘
  0x1A  AGC        — automatic gain control    ┐
  0x1B  MAGNITUDE  (MSB)                       │ one 3-byte read
  0x1C  MAGNITUDE  (LSB)                       ┘

Mounting checks:
  • Status should read OK.  TOO WEAK → magnet too far; TOO STRONG → too close
    (datasheet air gap ~0.5–3 mm).
  • AGC should sit mid-range: ~64 when powered at 3.3 V (range 0–128),
    ~128 at 5 V (range 0–255).  Pinned at either end means a bad air gap.
  • Spin the joint a full turn: Angle should sweep 0→360 smoothly with no
    jumps, and Unwrapped should return to its start value.
  • The DIR pin must be tied to GND or VDD — floating, it flips direction.

Press Ctrl+C to stop.
"""

import argparse
import sys
import time

import smbus2

# ── Hardware ──────────────────────────────────────────────────────────────────
I2C_BUS     = 1
AS5600_ADDR = 0x36

# ── Register addresses ────────────────────────────────────────────────────────
REG_STATUS    = 0x0B   # start of STATUS + RAWANGLE + ANGLE block (5 bytes)
REG_AGC       = 0x1A   # start of AGC + MAGNITUDE block (3 bytes)

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
class VelocityEstimator:
    """Tracks full rotations to give a continuous (unwrapped) angle + velocity."""

    def __init__(self):
        self._prev_angle = None
        self._prev_time  = None
        self._unwrapped  = 0.0

    def update(self, angle_deg: float, now: float):
        """
        Returns (unwrapped_angle_deg, velocity_deg_per_s).
        unwrapped_angle_deg accumulates across full rotations.
        """
        if self._prev_angle is None:
            self._prev_angle = angle_deg
            self._prev_time  = now
            self._unwrapped  = angle_deg
            return angle_deg, 0.0

        dt    = now - self._prev_time
        delta = angle_deg - self._prev_angle

        # Wrap delta to [-180, +180] to detect direction across 0/360 boundary
        if delta > 180.0:
            delta -= 360.0
        elif delta < -180.0:
            delta += 360.0

        self._unwrapped  += delta
        self._prev_angle  = angle_deg
        self._prev_time   = now

        velocity = delta / dt if dt > 0 else 0.0
        return self._unwrapped, velocity


# ── AS5600 reads ──────────────────────────────────────────────────────────────
def to_12bit(msb: int, lsb: int) -> int:
    """Combine a big-endian register pair into a 12-bit value."""
    return ((msb & 0x0F) << 8) | lsb


def read_angle_block(bus: smbus2.SMBus, addr: int):
    """One 5-byte read → (status_byte, raw_counts, angle_counts)."""
    d = bus.read_i2c_block_data(addr, REG_STATUS, 5)
    return d[0], to_12bit(d[1], d[2]), to_12bit(d[3], d[4])


def read_magnet_block(bus: smbus2.SMBus, addr: int):
    """One 3-byte read → (agc, magnitude)."""
    d = bus.read_i2c_block_data(addr, REG_AGC, 3)
    return d[0], to_12bit(d[1], d[2])


def decode_status(status_byte: int) -> str:
    if not (status_byte & STATUS_MD):
        return "NO MAGNET"
    if status_byte & STATUS_MH:
        return "TOO STRONG"
    if status_byte & STATUS_ML:
        return "TOO WEAK"
    return "OK"


def wrap_360(deg: float) -> float:
    return deg % DEGREES_PER_REV


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Live AS5600 encoder test.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("-b", "--bus", type=int, default=I2C_BUS,
                        help=f"I2C bus number (default: {I2C_BUS})")
    parser.add_argument("-a", "--addr", type=lambda s: int(s, 0),
                        default=AS5600_ADDR,
                        help=f"I2C address (default: 0x{AS5600_ADDR:02X})")
    parser.add_argument("--hz", type=float, default=SAMPLE_HZ,
                        help=f"Sample/print rate in Hz (default: {SAMPLE_HZ})")
    parser.add_argument("--zero", action="store_true",
                        help="Use the angle at startup as the 0° reference "
                             "(software only, nothing written to the chip)")
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args     = parse_args()
    interval = 1.0 / args.hz
    vel      = VelocityEstimator()

    print(f"AS5600 encoder test — bus {args.bus}, addr 0x{args.addr:02X}")
    print(f"12-bit resolution ({COUNTS_PER_REV} counts/rev) | {args.hz:g} Hz")

    with smbus2.SMBus(args.bus) as bus:
        # Probe once so a wiring problem gives a clear message, not a traceback
        try:
            status_byte, _, angle_counts = read_angle_block(bus, args.addr)
            agc, magnitude = read_magnet_block(bus, args.addr)
        except OSError as exc:
            print(f"\n[ERROR] No response from 0x{args.addr:02X} on bus "
                  f"{args.bus} ({exc}).")
            print(f"        Check wiring/power, then: i2cdetect -y -r {args.bus}")
            sys.exit(1)

        print(f"Found device.  Status: {decode_status(status_byte)} | "
              f"AGC: {agc} | Magnitude: {magnitude}")

        zero_deg = angle_counts * COUNTS_TO_DEG if args.zero else 0.0
        if args.zero:
            print(f"Software zero set at {zero_deg:.3f}° (counts {angle_counts})")

        print()
        print(f"{'Status':<12}  {'Raw':<6}  {'Angle (°)':<11}  {'Unwrapped (°)':<15}  {'Vel (°/s)':<11}  {'AGC':<5}  {'Magnitude'}")
        print("-" * 80)

        count   = 0
        errors  = 0
        t_start = time.monotonic()

        try:
            while True:
                t0 = time.monotonic()

                try:
                    status_byte, raw_counts, angle_counts = read_angle_block(bus, args.addr)
                    agc, magnitude = read_magnet_block(bus, args.addr)
                except OSError as exc:
                    errors += 1
                    print(f"[I2C error #{errors}: {exc}] retrying...", flush=True)
                    time.sleep(interval)
                    continue

                count += 1
                angle_deg  = wrap_360(angle_counts * COUNTS_TO_DEG - zero_deg)
                status_str = decode_status(status_byte)
                unwrapped, velocity = vel.update(angle_deg, t0)

                print(
                    f"{status_str:<12}  "
                    f"{raw_counts:<6d}  "
                    f"{angle_deg:>10.3f}°  "
                    f"{unwrapped:>14.2f}°  "
                    f"{velocity:>10.1f}   "
                    f"{agc:<5d}  "
                    f"{magnitude}"
                )

                elapsed    = time.monotonic() - t0
                sleep_time = interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            elapsed = time.monotonic() - t_start
            rate    = count / elapsed if elapsed > 0 else 0.0
            print(f"\nStopped after {elapsed:.1f}s — {count} samples "
                  f"({rate:.1f} Hz), {errors} I2C errors.")


if __name__ == "__main__":
    main()
