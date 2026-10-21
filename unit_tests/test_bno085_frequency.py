#!/usr/bin/env python3
"""
Simple runtime test for a single BNO085 at a requested frequency.

Usage examples
--------------
python3 unit_tests/test_bno085_frequency.py --freq 100
python3 unit_tests/test_bno085_frequency.py --freq 200 --duration 10
python3 unit_tests/test_bno085_frequency.py --freq 100 --bus 7 --addr 0x4A

Notes
-----
- This is a hardware test (not a mocked unit test).
- Reads from one BNO085 using the shared IMUReader implementation.
- Prints summary stats so you can verify target-vs-achieved rate.
"""

import argparse
import sys
import time
from pathlib import Path


# Make the BNO085 folder importable when run from workspace root.
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
BNO_PATH = WORKSPACE_ROOT / "Sensor_Tests" / "IMUs" / "BNO085"
if str(BNO_PATH) not in sys.path:
    sys.path.insert(0, str(BNO_PATH))

from bno085_live import IMUReader  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test one BNO085 at a target frequency.")
    parser.add_argument("--freq", type=float, required=True, help="Target sample frequency in Hz (e.g. 100).")
    parser.add_argument("--duration", type=float, default=5.0, help="Test duration in seconds (default: 5).")
    parser.add_argument("--bus", type=int, default=1, help="I2C bus number (default: 1).")
    parser.add_argument(
        "--addr",
        type=lambda value: int(value, 0),
        default=0x4A,
        help="I2C address in hex or decimal (default: 0x4A).",
    )
    parser.add_argument(
        "--print-samples",
        action="store_true",
        help="Print each valid sample so duplicates can be visually checked.",
    )
    parser.add_argument(
        "--count-unique",
        action="store_true",
        help="Count and print the number of unique valid samples.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.freq <= 0:
        raise SystemExit("[ERROR] --freq must be > 0")
    if args.duration <= 0:
        raise SystemExit("[ERROR] --duration must be > 0")

    period = 1.0 / args.freq
    samples = 0
    misses = 0
    unique_samples = set()

    print(
        f"Starting BNO085 frequency test: target={args.freq:.2f} Hz, "
        f"duration={args.duration:.2f}s, bus={args.bus}, addr=0x{args.addr:02X}"
    )

    start_time = time.perf_counter()
    next_tick = start_time

    with IMUReader(bus=args.bus, address=args.addr) as imu:
        while True:
            now = time.perf_counter()
            elapsed = now - start_time
            if elapsed >= args.duration:
                break

            if now < next_tick:
                time.sleep(next_tick - now)

            sample = imu.read()
            if sample is not None:
                samples += 1
                if args.count_unique:
                    qi, qj, qk, qr = sample["quat"]
                    ax, ay, az = sample["accel"]
                    gx, gy, gz = sample["gyro"]
                    unique_samples.add((qi, qj, qk, qr, ax, ay, az, gx, gy, gz))
                if args.print_samples:
                    qi, qj, qk, qr = sample["quat"]
                    ax, ay, az = sample["accel"]
                    gx, gy, gz = sample["gyro"]
                    t_rel = time.perf_counter() - start_time
                    print(
                        f"[{samples:05d}] t={t_rel:8.4f}s "
                        f"quat=({qi:+.5f}, {qj:+.5f}, {qk:+.5f}, {qr:+.5f}) "
                        f"accel=({ax:+.4f}, {ay:+.4f}, {az:+.4f}) "
                        f"gyro=({gx:+.4f}, {gy:+.4f}, {gz:+.4f})",
                        flush=True,
                    )
            else:
                misses += 1

            next_tick += period

    total_elapsed = time.perf_counter() - start_time
    achieved_hz = samples / total_elapsed if total_elapsed > 0 else 0.0
    expected_samples = int(args.freq * args.duration)

    print("\n--- Test Summary ---")
    print(f"Expected samples (approx): {expected_samples}")
    print(f"Valid samples read:        {samples}")
    print(f"Missed/empty reads:        {misses}")
    print(f"Elapsed time:              {total_elapsed:.3f}s")
    print(f"Achieved frequency:        {achieved_hz:.2f} Hz")
    if args.count_unique:
        unique_count = len(unique_samples)
        duplicate_count = samples - unique_count
        print(f"Unique valid samples:      {unique_count}")
        print(f"Duplicate valid samples:   {duplicate_count}")


if __name__ == "__main__":
    main()
