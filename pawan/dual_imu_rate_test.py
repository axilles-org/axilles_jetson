#!/usr/bin/env python3
"""
dual_imu_rate_test.py

Read BOTH BNO085s (0x4A and 0x4B) on one I2C bus for 10 seconds and report
the rate each report actually arrives at.

    python3 dual_imu_rate_test.py                # lean reader, accel + gyro
    python3 dual_imu_rate_test.py --stock        # stock adafruit reader, accel + gyro
    python3 dual_imu_rate_test.py --quat         # also enable the quaternion
    python3 dual_imu_rate_test.py --bus 7 --seconds 30
    python3 dual_imu_rate_test.py --plot         # also record + plot every value
    python3 dual_imu_rate_test.py --csv          # also save every value to CSV

Lean vs --stock with the same features isolates fix 2 (one read per packet);
adding --quat shows what dropping the quaternion (fix 1) bought.

Reported per IMU and report:
    Hz          reports received per second (counted as they are parsed)
    p99 / max   gap between consecutive reports, ms. At 200 Hz the ideal
                gap is 5 ms; large gaps mean reports were late or dropped.
    lost        packets the sensor sent that were never read (sequence-number
                gaps; lean mode only). Non-zero means the bus fell behind.
    bus busy    time spent in reads that returned data (lean mode only).

--csv saves every report to imu_data_<date>_<time>.csv, one row per report:
    t_s, imu, report, x, y, z, w
t_s is seconds since the script started (same clock for both IMUs, taken when
the report was parsed). accel/gyro fill x, y, z and leave w empty; quat fills
x, y, z, w with i, j, k, real.

--plot records every report value and saves imu_values.png (full run plus a
zoomed 0.2 s window where individual samples are visible), and prints how often
a value was identical to the one before it. Move the IMUs while it records: a
sensor at rest legitimately repeats gyro values, which only change in steps of
about 0.002 rad/s.
"""

import argparse
import csv
import os
import time
from datetime import datetime
from pathlib import Path

T_START = time.perf_counter()      # t_s = 0 in the CSV

from bno085_lean import ACCEL, GYRO, QUAT, LeanBNO085, BOOT_DELAY
from adafruit_bno08x.i2c import BNO08X_I2C
from adafruit_extended_bus import ExtendedI2C as I2C

ADDRS = (0x4A, 0x4B)
NAMES = {ACCEL: "accel", GYRO: "gyro", QUAT: "quat"}
UNITS = {ACCEL: "m/s²", GYRO: "rad/s", QUAT: ""}
AXES = {ACCEL: "xyz", GYRO: "xyz", QUAT: ("i", "j", "k", "real")}
ZOOM_S = 0.2
PASS_FRACTION = 0.95


class StockIMU:
    """Stock adafruit reader with the same counting interface as LeanBNO085."""

    def __init__(self, bus, address, report_hz, features):
        self.label = f"0x{address:02X}"
        self.features = features
        self.counts = {}
        interval_us = max(1000, int(1_000_000 / report_hz))
        for attempt in range(1, 7):
            try:
                self._i2c = I2C(bus)
                self._bno = BNO08X_I2C(self._i2c, address=address)
                time.sleep(BOOT_DELAY)
                for feat in features:
                    self._bno.enable_feature(feat, interval_us)
                break
            except Exception as exc:
                print(f"[{self.label}] connect attempt {attempt} failed: {exc}")
                try:
                    self._i2c.deinit()
                except Exception:
                    pass
                time.sleep(1.0)
        else:
            raise RuntimeError(f"[{self.label}] could not connect")

        orig = self._bno._process_report

        def counting(report_id, report_bytes):
            if report_id in self.features:
                self.counts[report_id] = self.counts.get(report_id, 0) + 1
            return orig(report_id, report_bytes)

        self._bno._process_report = counting

    def poll(self):
        before = sum(self.counts.values())
        self._bno._process_available_packets()
        return sum(self.counts.values()) - before

    def reset_stats(self):
        self.counts.clear()

    def close(self):
        try:
            self._i2c.deinit()
        except Exception:
            pass


def gap_stats(times):
    if len(times) < 2:
        return float("nan"), float("nan")
    gaps = sorted((b - a) * 1e3 for a, b in zip(times, times[1:]))
    return gaps[min(len(gaps) - 1, int(0.99 * len(gaps)))], gaps[-1]


def install_recorder(imu, sink):
    """Record (time, value) for every report as it is parsed, including each
    report inside a batched packet, not just the latest one per poll."""
    bno = imu._bno
    orig = bno._process_report

    def record(report_id, report_bytes):
        orig(report_id, report_bytes)
        if report_id in sink:
            sink[report_id].append((time.perf_counter(), bno._readings[report_id]))

    bno._process_report = record


def write_csv(samples, path):
    rows = []
    for label, per_feature in samples.items():
        for f, data in per_feature.items():
            for t, v in data:
                rows.append((t - T_START, label, NAMES[f], *v))
    rows.sort(key=lambda r: r[0])
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t_s", "imu", "report", "x", "y", "z", "w"])
        for r in rows:
            vals = [f"{x:.6f}" for x in r[3:]]
            w.writerow([f"{r[0]:.6f}", r[1], r[2], *vals, *[""] * (4 - len(vals))])
    print(f"Saved {len(rows)} rows to {path}")


def repeat_stats(samples):
    """(% of samples identical to the previous one, number of distinct values)."""
    vals = [v for _, v in samples]
    if len(vals) < 2:
        return float("nan"), len(set(vals))
    same = sum(a == b for a, b in zip(vals, vals[1:]))
    return 100.0 * same / (len(vals) - 1), len(set(vals))


def plot_values(samples, features, t0, path):
    import matplotlib
    if not os.environ.get("DISPLAY"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(samples)
    fig, axs = plt.subplots(2 * len(features), len(labels), squeeze=False,
                            figsize=(7 * len(labels), 3.2 * 2 * len(features)))
    for col, label in enumerate(labels):
        for i, f in enumerate(features):
            data = samples[label][f]
            if not data:
                continue
            t = [ts - t0 for ts, _ in data]
            mid = t[len(t) // 2]
            for row, zoom in ((2 * i, False), (2 * i + 1, True)):
                ax = axs[row][col]
                for k, axis in enumerate(AXES[f]):
                    y = [v[k] for _, v in data]
                    if zoom:
                        ax.plot(t, y, ".-", ms=4, lw=0.8, label=axis)
                        ax.set_xlim(mid, mid + ZOOM_S)
                    else:
                        ax.plot(t, y, lw=0.8, label=axis)
                title = f"{label} {NAMES[f]}"
                title += f" (zoom {ZOOM_S:.1f} s, dots = samples)" if zoom else ""
                ax.set_title(title)
                ax.set_ylabel(UNITS[f])
                ax.grid(alpha=0.3)
                ax.legend(loc="upper right", fontsize=8)
            if zoom:
                # Autoscale y to the zoomed window only.
                lo = [min(v[k] for ts, v in data if mid <= ts - t0 <= mid + ZOOM_S)
                      for k in range(len(AXES[f]))]
                hi = [max(v[k] for ts, v in data if mid <= ts - t0 <= mid + ZOOM_S)
                      for k in range(len(AXES[f]))]
                pad = max(1e-3, (max(hi) - min(lo)) * 0.1)
                axs[2 * i + 1][col].set_ylim(min(lo) - pad, max(hi) + pad)
    for ax in axs[-1]:
        ax.set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(f"Saved plot to {path}")
    if os.environ.get("DISPLAY"):
        plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bus", type=int, default=1)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--report-hz", type=float, default=200.0)
    ap.add_argument("--quat", action="store_true",
                    help="also enable the game rotation vector")
    ap.add_argument("--stock", action="store_true",
                    help="use the stock adafruit reader instead of the lean one")
    ap.add_argument("--plot", nargs="?", const="imu_values.png", metavar="PNG",
                    help="record every value and save a plot "
                         "(default imu_values.png next to this script)")
    ap.add_argument("--csv", nargs="?", const="", metavar="CSV",
                    help="save every report to CSV "
                         "(default imu_data_<date>_<time>.csv next to this script)")
    args = ap.parse_args()

    features = (ACCEL, GYRO, QUAT) if args.quat else (ACCEL, GYRO)
    mode = "stock" if args.stock else "lean"

    imus = []
    for addr in ADDRS:
        print(f"Connecting 0x{addr:02X} on bus {args.bus} ({mode})...")
        if args.stock:
            imus.append(StockIMU(args.bus, addr, args.report_hz, features))
        else:
            imus.append(LeanBNO085(args.bus, addr, args.report_hz, features))

    errors = {imu.label: 0 for imu in imus}

    def poll(imu):
        try:
            return imu.poll()
        except (OSError, RuntimeError, KeyError, IndexError) as exc:
            errors[imu.label] += 1
            if errors[imu.label] <= 3:
                print(f"[{imu.label}] read error: {exc}")
            return 0

    # Warm up so the FIFO backlog built up during setup is not counted.
    end = time.perf_counter() + 0.5
    while time.perf_counter() < end:
        for imu in imus:
            poll(imu)
    for imu in imus:
        imu.reset_stats()
    for k in errors:
        errors[k] = 0

    names = ", ".join(NAMES[f] for f in features)
    print(f"\nReading both for {args.seconds:.1f} s ({names} at "
          f"{args.report_hz:.0f} Hz, {mode} reader)...\n")

    arrivals = {imu.label: {f: [] for f in features} for imu in imus}
    samples = {imu.label: {f: [] for f in features} for imu in imus}
    if args.plot or args.csv is not None:
        for imu in imus:
            install_recorder(imu, samples[imu.label])
    try:
        t0 = time.perf_counter()
        end = t0 + args.seconds
        while True:
            now = time.perf_counter()
            if now >= end:
                break
            for imu in imus:
                before = dict(imu.counts)
                if poll(imu):
                    t = time.perf_counter()
                    for f in features:
                        if imu.counts.get(f, 0) != before.get(f, 0):
                            arrivals[imu.label][f].append(t)
        elapsed = time.perf_counter() - t0
    finally:
        for imu in imus:
            imu.close()

    # ── report ───────────────────────────────────────────────────────────────
    target = args.report_hz * PASS_FRACTION
    ok = True
    print(f"Elapsed: {elapsed:.2f} s   (ideal gap at {args.report_hz:.0f} Hz: "
          f"{1000 / args.report_hz:.1f} ms)\n")
    print(f"{'IMU':<6}{'report':<7}{'Hz':>8}{'p99 ms':>9}{'max ms':>9}")
    print("-" * 42)
    for imu in imus:
        for f in features:
            hz = imu.counts.get(f, 0) / elapsed
            p99, mx = gap_stats(arrivals[imu.label][f])
            passed = hz >= target
            ok &= passed
            print(f"{imu.label:<6}{NAMES[f]:<7}{hz:>8.1f}{p99:>9.1f}{mx:>9.1f}"
                  f"   {'PASS' if passed else 'FAIL'}")
    print()

    total_busy = 0.0
    for imu in imus:
        line = f"{imu.label}: errors {errors[imu.label]}"
        if not args.stock:
            total_busy += imu.busy_s
            line += (f", packets/s {imu.packets / elapsed:.0f}"
                     f", empty polls/s {imu.empty_polls / elapsed:.0f}"
                     f", continuations {imu.continuations}"
                     f", lost packets {imu.seq_gaps}"
                     f", bus busy {imu.busy_s / elapsed * 100:.0f}%")
            ok &= imu.seq_gaps == 0
        print(line)
    if not args.stock:
        print(f"Bus busy on data, both IMUs: {total_busy / elapsed * 100:.0f}% "
              f"(the rest is empty polls / free for other devices)")

    print()
    print("PASS: both IMUs at target rate." if ok else
          "FAIL: at least one report below target or packets lost.")

    here = Path(__file__).resolve().parent
    if args.csv is not None:
        name = args.csv or f"imu_data_{datetime.now():%Y%m%d_%H%M%S}.csv"
        path = Path(name)
        write_csv(samples, path if path.is_absolute() else here / path)

    if args.plot:
        print(f"\n{'IMU':<6}{'report':<7}{'samples':>9}{'repeated':>10}"
              f"{'distinct':>10}")
        print("-" * 42)
        for label in samples:
            for f in features:
                pct, distinct = repeat_stats(samples[label][f])
                print(f"{label:<6}{NAMES[f]:<7}{len(samples[label][f]):>9}"
                      f"{pct:>9.1f}%{distinct:>10}")
        path = Path(args.plot)
        if not path.is_absolute():
            path = here / path
        plot_values(samples, features, t0, path)


if __name__ == "__main__":
    main()
