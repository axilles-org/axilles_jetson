#!/usr/bin/env python3
"""
i2c_speed_probe.py

Measure the actual I2C bus throughput, with no sensor library in the way.

Run on the Jetson:

    python3 i2c_speed_probe.py

Why this exists
---------------
imu_drain_test showed every configuration sitting at ~100% busy time, and showed
that removing the quaternion report - about 14 bytes of payload - saved 5.3 ms per
call. Payload size dominating the cost means the limit is bus bandwidth rather than
per-call overhead or the FIFO depth.

If that is right, the fix is the bus clock, and the clock is the one number we still
do not have: i2c_bench could not find it in the device tree.

So this measures it, by timing raw smbus2 reads of different sizes against the
AS5600. That device is trivial - no packet protocol, no Python parsing - so the
slope of time against bytes is the bus itself:

    time = fixed_overhead + bytes / throughput

A 100 kHz bus delivers roughly 10 KB/s of payload; 400 kHz delivers roughly 40 KB/s.
Knowing which you have settles whether raising the clock is worth doing, and how
much it would buy.

Read-only. Nothing is configured or written.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

AS5600_ADDR = 0x36
AS5600_REG_ANGLE = 0x0E
BNO_FOOT_ADDR = 0x4A


def read_clock_from_system():
    """Every place the bus clock might be recorded, tried in turn."""
    found = []
    roots = [Path("/sys/bus/i2c/devices"), Path("/proc/device-tree")]
    for root in roots:
        if not root.exists():
            continue
        try:
            for p in root.rglob("clock-frequency"):
                try:
                    raw = p.read_bytes()
                    hz = (int.from_bytes(raw, "big") if len(raw) == 4
                          else int(raw.decode().strip()))
                    if 1000 <= hz <= 5_000_000:
                        found.append((str(p), hz))
                except Exception:
                    continue
        except Exception:
            continue
    return found


def time_read(bus, addr, reg, length, trials):
    """Median seconds for one read of `length` bytes."""
    lat = []
    for _ in range(trials):
        t0 = time.perf_counter()
        try:
            bus.read_i2c_block_data(addr, reg, length)
        except OSError:
            pass
        lat.append(time.perf_counter() - t0)
    return statistics.median(lat)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--bus", type=int, default=1)
    args = ap.parse_args()

    print("=" * 74)
    print("  I2C SPEED PROBE")
    print("=" * 74)

    print("\n  clock-frequency entries found on this system:")
    entries = read_clock_from_system()
    if entries:
        for path, hz in entries:
            tag = "  <- likely your bus" if f"i2c-{args.bus}" in path else ""
            print(f"    {hz / 1000:>6.0f} kHz   {path}{tag}")
    else:
        print("    none found. Measuring instead.")

    try:
        import smbus2
    except ImportError as exc:
        print(f"\n  smbus2 unavailable ({exc}). Run this on the Jetson.")
        return 1

    bus = smbus2.SMBus(args.bus)
    try:
        # ---- does the AS5600 answer? ----
        try:
            bus.read_i2c_block_data(AS5600_ADDR, AS5600_REG_ANGLE, 2)
        except OSError as exc:
            print(f"\n  AS5600 at 0x{AS5600_ADDR:02X} did not respond ({exc}).")
            print("  Check the wiring, or pass --bus for a different bus.")
            return 1

        print(f"\n  timing raw reads from the AS5600 at 0x{AS5600_ADDR:02X}"
              f" ({args.trials} trials each)\n")
        print(f"  {'bytes':>7}{'median':>11}{'per byte':>12}")
        print("  " + "-" * 30)

        sizes = [1, 2, 4, 8, 16, 24, 32]
        points = []
        for n in sizes:
            med = time_read(bus, AS5600_ADDR, AS5600_REG_ANGLE, n, args.trials)
            points.append((n, med))
            print(f"  {n:>7}{med * 1e3:>10.3f}m{med / n * 1e6:>11.1f}us")

        # Least-squares slope of time against bytes: the marginal cost per byte.
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        denom = sum((x - mx) ** 2 for x in xs)
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom
        intercept = my - slope * mx

        print(f"\n  fixed overhead per transaction : {intercept * 1e3:.3f} ms")
        print(f"  marginal cost per byte         : {slope * 1e6:.1f} us")
        if slope > 0:
            throughput = 1.0 / slope
            print(f"  implied payload throughput     : {throughput / 1000:.1f} KB/s")
            # 9 clocks per byte on I2C (8 data + 1 ack)
            implied_clock = throughput * 9
            print(f"  implied bus clock              : ~{implied_clock / 1000:.0f} kHz")
        else:
            throughput = float("inf")
            implied_clock = 0
            print("  marginal cost is below the timer noise: overhead dominates,")
            print("  so this bus is not payload-limited at these sizes.")

        # ---- what that means for the IMUs ----
        print("\n" + "=" * 74)
        print("  WHAT THIS MEANS FOR THE IMUs")
        print("=" * 74)
        if slope > 0:
            if implied_clock < 180_000:
                print(f"  ~{implied_clock / 1000:.0f} kHz. This is a 100 kHz bus.")
                print("  Raising it to 400 kHz is a 4x increase in payload capacity and")
                print("  is a configuration change, not a rewiring job. Given that the")
                print("  IMU cost scales with payload, this is the fix to try first.")
                print("\n  On Jetson this is set in the device tree for the I2C node")
                print("  (clock-frequency = <400000>), then reboot. Confirm afterwards")
                print("  by re-running this probe.")
            elif implied_clock < 600_000:
                print(f"  ~{implied_clock / 1000:.0f} kHz. Already at 400 kHz, so the clock")
                print("  is not the remaining problem. That leaves:")
                print("    - moving one IMU to a second I2C bus")
                print("    - switching the BNO085 to SPI, which is far faster")
                print("    - accepting a lower rate and setting IMU_REPORT_HZ to match,")
                print("      so the data is at least fresh rather than a stale backlog")
            else:
                print(f"  ~{implied_clock / 1000:.0f} kHz, which is fast. The bus is not the")
                print("  limit, so the cost is in the driver's Python packet handling.")
        else:
            print("  Per-transaction overhead dominates, not payload. Raising the bus")
            print("  clock would not help much; the cost is in the syscall and driver")
            print("  path per transaction, so the fix is fewer, larger transactions.")

        if intercept > 0:
            max_tx = 1.0 / intercept
            print(f"\n  At {intercept * 1e3:.2f} ms fixed cost, the ceiling is about"
                  f" {max_tx:.0f} transactions/s")
            print("  even for zero-length reads. Two IMUs needing 200 reports/s each")
            print(f"  would need {2 * 200:.0f} transactions/s minimum.")
            if max_tx < 400:
                print("  That alone puts 200 Hz out of reach on this bus, regardless")
                print("  of clock speed. Plan for a second bus or SPI.")
        return 0
    finally:
        bus.close()


if __name__ == "__main__":
    sys.exit(main())
