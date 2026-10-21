#!/usr/bin/env python3
"""
Simple BNO085 frequency test.

Pass a target frequency, and it prints the actual fresh-sample frequency
attained while reading a single BNO085.

Usage:
  python3 simple_bno085_frequency.py --freq 100
  python3 simple_bno085_frequency.py --freq 200 --duration 8
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
BNO_PATH = WORKSPACE_ROOT / "Sensor_Tests" / "IMUs" / "BNO085"
if str(BNO_PATH) not in sys.path:
    sys.path.insert(0, str(BNO_PATH))

from bno085_live import IMUReader  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple BNO085 frequency test")
    parser.add_argument("--freq", type=float, required=True, help="Target frequency (Hz)")
    parser.add_argument("--duration", type=float, default=6.0, help="Test duration (seconds)")
    parser.add_argument("--bus", type=int, default=1, help="I2C bus number (default: 1)")
    parser.add_argument(
        "--addr",
        type=lambda value: int(value, 0),
        default=0x4A,
        help="I2C address in hex or decimal (default: 0x4A)",
    )
    return parser.parse_args()


def sample_key(sample: dict) -> tuple:
    qi, qj, qk, qr = sample["quat"]
    ax, ay, az = sample["accel"]
    gx, gy, gz = sample["gyro"]
    return (qi, qj, qk, qr, ax, ay, az, gx, gy, gz)


def main() -> None:
    args = parse_args()

    if args.freq <= 0:
        raise ValueError("--freq must be > 0")
    if args.duration <= 0:
        raise ValueError("--duration must be > 0")

    period = 1.0 / args.freq
    fresh_samples = 0
    repeated_samples = 0
    misses = 0
    total_polls = 0
    last_key = None

    print(
        f"Starting BNO085 frequency test: target={args.freq:.2f} Hz, "
        f"duration={args.duration:.2f}s, bus={args.bus}, addr=0x{args.addr:02X}"
    )

    start_time = time.perf_counter()
    next_tick = start_time

    with IMUReader(bus=args.bus, address=args.addr) as imu:
        while True:
            now = time.perf_counter()
            if now - start_time >= args.duration:
                break

            if now < next_tick:
                time.sleep(next_tick - now)

            total_polls += 1
            sample = imu.read()

            if sample is None:
                misses += 1
            else:
                current_key = sample_key(sample)
                if current_key != last_key:
                    fresh_samples += 1
                    last_key = current_key
                else:
                    repeated_samples += 1

            next_tick += period

    total_elapsed = time.perf_counter() - start_time
    fresh_hz = fresh_samples / total_elapsed if total_elapsed > 0 else 0.0

    print("\n--- Test Summary ---")
    print(f"Target frequency:          {args.freq:.2f} Hz")
    print(f"Fresh samples read:        {fresh_samples}")
    print(f"Repeated/cached samples:   {repeated_samples}")
    print(f"Missed/empty reads:        {misses}")
    print(f"Total poll attempts:       {total_polls}")
    print(f"Elapsed time:              {total_elapsed:.3f}s")
    print(f"Fresh sample frequency:    {fresh_hz:.2f} Hz")


if __name__ == "__main__":
    main()
