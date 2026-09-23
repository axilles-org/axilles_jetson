#!/usr/bin/env python3
"""
imu_rate_test.py

Read both BNO085s (0x4A and 0x4B on I2C bus 1) for 10 seconds and print the
rate in Hz at which data arrived from each one.

    python3 imu_rate_test.py                 # 10 s, 200 Hz report rate
    python3 imu_rate_test.py --seconds 5 --report-hz 100

Two rates are reported per IMU:
    read calls/s   how often the loop polled that sensor
    feature Hz     how often each report's value actually changed (new data)
"""

import argparse
import time
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning,
                        message="I2C frequency is not settable")

from adafruit_extended_bus import ExtendedI2C as I2C
from adafruit_bno08x.i2c import BNO08X_I2C
from adafruit_bno08x import (
    BNO_REPORT_ACCELEROMETER,
    BNO_REPORT_GYROSCOPE,
    BNO_REPORT_GAME_ROTATION_VECTOR,
)

I2C_BUS = 1
ADDRS = {"IMU-4A": 0x4A, "IMU-4B": 0x4B}
BOOT_DELAY = 0.8
FEATURE_RETRIES = 5

FEATURES = {
    "accel": BNO_REPORT_ACCELEROMETER,
    "gyro":  BNO_REPORT_GYROSCOPE,
    "quat":  BNO_REPORT_GAME_ROTATION_VECTOR,
}


def open_imu(i2c, address, report_hz):
    bno = BNO08X_I2C(i2c, address=address)
    time.sleep(BOOT_DELAY)
    interval_us = max(1000, int(1_000_000 / report_hz))
    for feat in FEATURES.values():
        for attempt in range(FEATURE_RETRIES):
            try:
                bno.enable_feature(feat, interval_us)
                break
            except Exception:
                if attempt == FEATURE_RETRIES - 1:
                    raise
                time.sleep(0.2)
    return bno


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--report-hz", type=float, default=200.0,
                    help="report rate requested from each sensor feature")
    args = ap.parse_args()

    i2c = I2C(I2C_BUS)
    imus = {}
    for label, addr in ADDRS.items():
        print(f"Connecting {label} (bus {I2C_BUS}, 0x{addr:02X})...")
        imus[label] = open_imu(i2c, addr, args.report_hz)
    print(f"Both connected. Reading for {args.seconds:.1f} s "
          f"(requested {args.report_hz:.0f} Hz per feature)...\n")

    calls = {k: 0 for k in imus}
    errors = {k: 0 for k in imus}
    changes = {k: {f: 0 for f in FEATURES} for k in imus}
    last = {k: {f: None for f in FEATURES} for k in imus}

    try:
        t0 = time.perf_counter()
        end = t0 + args.seconds
        while time.perf_counter() < end:
            for label, bno in imus.items():
                try:
                    bno._process_available_packets()
                except (OSError, RuntimeError, KeyError) as exc:
                    errors[label] += 1
                    continue
                calls[label] += 1
                for name, feat in FEATURES.items():
                    v = bno._readings.get(feat)
                    if v is not None and v != last[label][name]:
                        changes[label][name] += 1
                        last[label][name] = v
        elapsed = time.perf_counter() - t0
    finally:
        try:
            i2c.deinit()
        except Exception:
            pass

    print(f"Elapsed: {elapsed:.2f} s\n")
    print(f"{'IMU':<8}{'calls/s':>10}{'accel Hz':>10}{'gyro Hz':>10}"
          f"{'quat Hz':>10}{'errors':>8}")
    print("-" * 56)
    for label in imus:
        c = changes[label]
        print(f"{label:<8}{calls[label] / elapsed:>10.1f}"
              f"{c['accel'] / elapsed:>10.1f}"
              f"{c['gyro'] / elapsed:>10.1f}"
              f"{c['quat'] / elapsed:>10.1f}"
              f"{errors[label]:>8}")


if __name__ == "__main__":
    main()
