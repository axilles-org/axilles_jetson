#!/usr/bin/env python3
"""
single_imu_rate_test.py

Read ONE BNO085 for 10 seconds and check whether it delivers 200 Hz on every
report. This answers whether giving each IMU its own I2C bus is enough to hit
200 Hz with the stock adafruit_bno08x driver.

    python3 single_imu_rate_test.py                       # bus 1, 0x4A
    python3 single_imu_rate_test.py --addr 0x4B
    python3 single_imu_rate_test.py --bus 7 --addr 0x4A   # after moving it
    python3 single_imu_rate_test.py --no-quat             # accel + gyro only

For a fair result, only the IMU under test should be generating traffic on its
bus (other IMUs/devices on the same bus can stay connected, just not be read).

Reported:
    report Hz    how many reports of each type the sensor delivered per second
    changed Hz   how often the value differed from the previous one. A still
                 sensor repeats identical gyro/quat values, so this can read
                 low even when every report arrived; judge by report Hz.
    bus busy     fraction of wall time spent inside I2C reads that returned
                 data. Headroom left on the bus = 100% - bus busy.
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

BOOT_DELAY = 0.8
CONNECT_RETRIES = 6
PASS_FRACTION = 0.95        # a feature passes at >= 95% of the requested rate


def connect(bus, address, features, report_hz):
    """Open the IMU and enable features, retrying the whole setup on failure."""
    interval_us = max(1000, int(1_000_000 / report_hz))
    for attempt in range(1, CONNECT_RETRIES + 1):
        i2c = None
        try:
            i2c = I2C(bus)
            bno = BNO08X_I2C(i2c, address=address)
            time.sleep(BOOT_DELAY)
            for feat in features.values():
                bno.enable_feature(feat, interval_us)
            return i2c, bno
        except Exception as exc:
            print(f"  connect attempt {attempt} failed: {exc}")
            if i2c is not None:
                try:
                    i2c.deinit()
                except Exception:
                    pass
            time.sleep(1.0)
    raise RuntimeError(f"Could not connect to 0x{address:02X} on bus {bus}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bus", type=int, default=1)
    ap.add_argument("--addr", type=lambda s: int(s, 0), default=0x4A)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--report-hz", type=float, default=200.0)
    ap.add_argument("--no-quat", action="store_true",
                    help="enable only accel + gyro")
    args = ap.parse_args()

    features = {"accel": BNO_REPORT_ACCELEROMETER,
                "gyro":  BNO_REPORT_GYROSCOPE}
    if not args.no_quat:
        features["quat"] = BNO_REPORT_GAME_ROTATION_VECTOR

    print(f"Connecting IMU 0x{args.addr:02X} on bus {args.bus}...")
    i2c, bno = connect(args.bus, args.addr, features, args.report_hz)

    # Time only the reads that actually carried a packet, so "bus busy"
    # measures the cost of the data itself, not of polling an empty FIFO.
    busy = [0.0]
    packets = [0]
    orig_read = bno._read

    def timed_read(n):
        t = time.perf_counter()
        try:
            return orig_read(n)
        finally:
            busy[0] += time.perf_counter() - t
            packets[0] += 1

    bno._read = timed_read

    # Count every report the driver parses, by report ID.
    counts = {}
    orig_process = bno._process_report

    def counting_process(report_id, report_bytes):
        counts[report_id] = counts.get(report_id, 0) + 1
        return orig_process(report_id, report_bytes)

    bno._process_report = counting_process

    # Warm up so the startup FIFO backlog does not skew the numbers.
    end = time.perf_counter() + 0.5
    while time.perf_counter() < end:
        try:
            bno._process_available_packets()
        except Exception:
            pass
    busy[0] = 0.0
    packets[0] = 0
    counts.clear()

    print(f"Reading for {args.seconds:.1f} s "
          f"({', '.join(features)} at {args.report_hz:.0f} Hz)...\n")

    calls = errors = 0
    changes = {f: 0 for f in features}
    last = {f: None for f in features}
    try:
        t0 = time.perf_counter()
        end = t0 + args.seconds
        while time.perf_counter() < end:
            try:
                bno._process_available_packets()
            except (OSError, RuntimeError, KeyError):
                errors += 1
                continue
            calls += 1
            for name, feat in features.items():
                v = bno._readings.get(feat)
                if v is not None and v != last[name]:
                    changes[name] += 1
                    last[name] = v
        elapsed = time.perf_counter() - t0
    finally:
        try:
            i2c.deinit()
        except Exception:
            pass

    # Each data packet also costs two 4-byte header reads before the payload
    # read timed above; estimate the full per-packet bus cost from that.
    pkt_rate = packets[0] / elapsed
    payload_ms = busy[0] / max(1, packets[0]) * 1e3

    print(f"Elapsed:        {elapsed:.2f} s")
    print(f"Read calls/s:   {calls / elapsed:.1f}")
    print(f"Packets/s:      {pkt_rate:.1f}  "
          f"(expected {len(features) * args.report_hz:.0f})")
    print(f"Payload read:   {payload_ms:.2f} ms/packet")
    print(f"Bus busy:       {busy[0] / elapsed * 100:.0f}% on payload reads "
          f"(headers add roughly another 0.5 ms/packet)")
    print(f"Errors:         {errors}\n")

    target = args.report_hz * PASS_FRACTION
    ok = True
    print(f"  {'':<6}{'report Hz':>10}{'changed Hz':>12}")
    for name, feat in features.items():
        hz = counts.get(feat, 0) / elapsed
        passed = hz >= target
        ok &= passed
        print(f"  {name:<6}{hz:>10.1f}{changes[name] / elapsed:>12.1f}   "
              f"{'PASS' if passed else 'FAIL'}")

    print()
    if ok:
        print(f"FEASIBLE: this IMU alone reaches {args.report_hz:.0f} Hz on "
              f"every report. One IMU per bus should work.")
    else:
        print(f"NOT FEASIBLE as configured: at least one report is below "
              f"{target:.0f} Hz even with the IMU alone.")


if __name__ == "__main__":
    main()
