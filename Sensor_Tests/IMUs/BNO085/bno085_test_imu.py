#!/usr/bin/env python3
"""
BNO085 single-IMU tester — specify address (and optionally bus) at runtime.

Usage
-----
    python3 bno085_test_imu.py --addr 0x4A
    python3 bno085_test_imu.py --addr 0x4B
    python3 bno085_test_imu.py --addr 0x4A --bus 7
    python3 bno085_test_imu.py -a 74        # decimal also accepted

Options
-------
    -a / --addr   I2C address of the IMU  (hex or decimal, required)
    -b / --bus    I2C bus number          (default: 7)
    -q / --quiet  suppress the header / rate summary (useful for piping)

The sensor streams Roll / Pitch / Yaw, raw accelerometer (m/s²), and
gyroscope (rad/s) at the maximum hardware rate.  Auto-reconnects on errors.
Press Ctrl+C to stop.
"""

import argparse
import time

from bno085_live import IMUReader, quat_to_euler

# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Test a single BNO085 IMU at a given I2C address.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "-a", "--addr",
        required=True,
        help="I2C address of the IMU (hex e.g. 0x4A, or decimal e.g. 74)",
    )
    parser.add_argument(
        "-b", "--bus",
        type=int,
        default=7,
        help="I2C bus number (default: 7)",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress header and rate summary",
    )
    return parser.parse_args()


def parse_address(addr_str: str) -> int:
    """Accept '0x4A', '0X4a', or plain decimal '74'."""
    try:
        return int(addr_str, 0)   # int(x, 0) handles 0x prefix automatically
    except ValueError:
        raise SystemExit(f"[ERROR] Invalid address: '{addr_str}'. "
                         "Use hex (e.g. 0x4A) or decimal (e.g. 74).")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args    = parse_args()
    address = parse_address(args.addr)
    bus     = args.bus
    quiet   = args.quiet

    if not quiet:
        print(f"Connecting to BNO085 — bus {bus}, address 0x{address:02X} ...")

    try:
        imu = IMUReader(bus=bus, address=address)
    except KeyboardInterrupt:
        print("\nAborted during init.")
        return

    if not quiet:
        print(f"Connected.  Streaming at max rate — Ctrl+C to stop.\n")

        hdr = (f"{'Roll':>9}{'Pitch':>9}{'Yaw':>9}  |  "
               f"{'Ax':>8}{'Ay':>8}{'Az':>8}  |  "
               f"{'Gx':>9}{'Gy':>9}{'Gz':>9}")
        print(hdr)
        print("-" * len(hdr))

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
                      f"{gx:9.4f}{gy:9.4f}{gz:9.4f}",
                      flush=True)
                count += 1

    except KeyboardInterrupt:
        elapsed = time.time() - t_start
        if not quiet:
            print(f"\nStopped. {count} samples in {elapsed:.1f}s "
                  f"({count / elapsed:.1f} Hz)")


if __name__ == "__main__":
    main()
