#!/usr/bin/env python3
"""
all_sensors_rate_test.py

Read every sensor on the exo's I2C bus together and report the rate each one
actually achieves, plus the effective rate of a complete sample row.

    python3 all_sensors_rate_test.py                   # everything at 200 Hz
    python3 all_sensors_rate_test.py --enc-hz 0 --fsr-hz 0   # encoder/ADC flat out
    python3 all_sensors_rate_test.py --csv             # also save every sample
    python3 all_sensors_rate_test.py --plot            # also plot every sample
    python3 all_sensors_rate_test.py --quat            # IMUs also send quaternion

Sensors (bus 1 by default, addresses as in data_collection.py):
    0x4A  BNO085 IMU-foot    accel + gyro (lean reader, bno085_lean.py)
    0x4B  BNO085 IMU-shank   accel + gyro
    0x36  AS5600 encoder     ankle angle
    0x48  ADS1115 ADC        AIN0 = toe FSR, AIN1 = heel FSR

How the loop shares the bus
---------------------------
One thread, one loop. Each pass: read the encoder if it is due, advance the ADC
state machine, then poll each IMU once, advancing the ADC again after each IMU.
The IMUs fill all remaining bus time.

The ADS1115 runs single-shot at 860 SPS, alternating channels: trigger AIN0,
wait out the conversion, read it, trigger AIN1, wait, read it. The conversion
is timed (CONV_S) rather than polled via the ready bit, which saves one I2C
transaction per sample. One FSR sample = both channels read.

Reported
--------
    Hz          samples per second for each stream
    p99 / max   gap between consecutive samples, ms (5 ms is ideal at 200 Hz)
    bus ms/s    time spent in I2C transactions per second, per device
    effective   slowest stream's rate, and the share of 200 Hz ticks at which
                every stream had delivered a new sample since the last tick
                (what a 200 Hz logger would see as a fully fresh row)

--plot saves all_sensors.png: every stream over the whole run (left) and a
0.2 s window with a dot per sample (right), and prints how often a sample was
identical to the one before it. Move the rig while recording; a still sensor
legitimately repeats values.
"""

import argparse
import csv
import os
import time
from datetime import datetime
from pathlib import Path

T_START = time.perf_counter()      # t_s = 0 in the CSV

import smbus2

from bno085_lean import ACCEL, GYRO, QUAT, LeanBNO085

IMU_ADDRS = {"foot": 0x4A, "shank": 0x4B}
ENC_ADDR = 0x36
ADS_ADDR = 0x48

_AS5600_REG_ANGLE = 0x0E
_ADS_REG_CONV = 0x00
_ADS_REG_CFG = 0x01
# Single-shot, ±4.096 V, 860 SPS, comparator off (same as data_collection.py).
_ADS_CFG = {0: [0xC3, 0xE3], 1: [0xD3, 0xE3]}
CONV_S = 0.0013     # 1/860 s = 1.16 ms, +10% for the ADS1115's clock tolerance

NAMES = {ACCEL: "accel", GYRO: "gyro", QUAT: "quat"}
PASS_FRACTION = 0.95
ZOOM_S = 0.2

# Plot rows: (title, streams drawn together, legend per value, y label)
AXES_XYZ = ("x", "y", "z")
PLOT_KINDS = {
    "accel": (AXES_XYZ, "m/s²"),
    "gyro": (AXES_XYZ, "rad/s"),
    "quat": (("i", "j", "k", "real"), ""),
}


class Stream:
    """Arrival times (and optionally values) of one data stream."""

    def __init__(self, name, target_hz, keep_values):
        self.name = name
        self.target_hz = target_hz
        self.times = []
        self.values = [] if keep_values else None

    def add(self, t, value):
        self.times.append(t)
        if self.values is not None:
            self.values.append(value)

    def clear(self):
        self.times.clear()
        if self.values is not None:
            self.values.clear()


class Encoder:
    def __init__(self, bus, stream):
        self.bus = bus
        self.stream = stream
        self.busy_s = 0.0
        self.errors = 0

    def read(self):
        t0 = time.perf_counter()
        try:
            d = self.bus.read_i2c_block_data(ENC_ADDR, _AS5600_REG_ANGLE, 2)
        except OSError:
            self.errors += 1
            return
        finally:
            self.busy_s += time.perf_counter() - t0
        raw = ((d[0] & 0x0F) << 8) | d[1]
        self.stream.add(time.perf_counter(), (raw * 360.0 / 4096.0,))


class ADS1115:
    """Two channels, single-shot, timed conversions, one call = one step."""

    def __init__(self, bus, toe, heel):
        self.bus = bus
        self.streams = (toe, heel)
        self.ch = None              # channel converting, None when idle
        self.t_trigger = 0.0
        self.busy_s = 0.0
        self.errors = 0

    @property
    def idle(self):
        return self.ch is None

    def _xfer(self, fn, *a):
        t0 = time.perf_counter()
        try:
            return fn(*a)
        finally:
            self.busy_s += time.perf_counter() - t0

    def start(self):
        self._trigger(0)

    def _trigger(self, ch):
        try:
            self._xfer(self.bus.write_i2c_block_data, ADS_ADDR, _ADS_REG_CFG,
                       _ADS_CFG[ch])
        except OSError:
            self.errors += 1
            self.ch = None
            return
        self.ch = ch
        self.t_trigger = time.perf_counter()

    def step(self, now):
        """Read the finished conversion, then start the next channel."""
        if self.ch is None or now - self.t_trigger < CONV_S:
            return
        ch = self.ch
        try:
            d = self._xfer(self.bus.read_i2c_block_data, ADS_ADDR, _ADS_REG_CONV, 2)
        except OSError:
            self.errors += 1
            self.ch = None
            return
        raw = (d[0] << 8) | d[1]
        raw = raw if raw < 0x8000 else raw - 0x10000
        self.streams[ch].add(time.perf_counter(), (raw,))
        if ch == 0:
            self._trigger(1)
        else:
            self.ch = None


def attach_imu_streams(imu, streams):
    """Timestamp every report as it is parsed, including batched ones."""
    bno = imu._bno
    orig = bno._process_report

    def record(report_id, report_bytes):
        orig(report_id, report_bytes)
        s = streams.get(report_id)
        if s is not None:
            s.add(time.perf_counter(), bno._readings[report_id])

    bno._process_report = record


def gap_stats(times):
    if len(times) < 2:
        return float("nan"), float("nan")
    gaps = sorted((b - a) * 1e3 for a, b in zip(times, times[1:]))
    return gaps[min(len(gaps) - 1, int(0.99 * len(gaps)))], gaps[-1]


def fresh_row_fraction(streams, t0, t1, hz):
    """Share of ticks at `hz` where every stream delivered since the last tick."""
    period = 1.0 / hz
    n_ticks = int((t1 - t0) / period)
    if n_ticks < 1:
        return float("nan")
    fresh = [True] * n_ticks
    for s in streams:
        hit = [False] * n_ticks
        for t in s.times:
            k = int((t - t0) / period)
            if 0 <= k < n_ticks:
                hit[k] = True
        fresh = [a and b for a, b in zip(fresh, hit)]
    return 100.0 * sum(fresh) / n_ticks


def write_csv(streams, path):
    rows = []
    for s in streams:
        for t, v in zip(s.times, s.values):
            rows.append((t - T_START, s.name, *v))
    rows.sort(key=lambda r: r[0])
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t_s", "stream", "v1", "v2", "v3", "v4"])
        for r in rows:
            vals = [f"{x:.6f}" if isinstance(x, float) else str(x) for x in r[2:]]
            w.writerow([f"{r[0]:.6f}", r[1], *vals, *[""] * (4 - len(vals))])
    print(f"Saved {len(rows)} rows to {path}")


def repeat_stats(values):
    """(% of samples identical to the previous one, number of distinct values)."""
    if len(values) < 2:
        return float("nan"), len(set(values))
    same = sum(a == b for a, b in zip(values, values[1:]))
    return 100.0 * same / (len(values) - 1), len(set(values))


def plot_streams(streams, path):
    import matplotlib
    if not os.environ.get("DISPLAY"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_name = {s.name: s for s in streams}
    # One row per IMU stream, then the encoder, then both FSRs together.
    rows = []
    for s in streams:
        kind = s.name.split("_", 1)[-1]
        if kind in PLOT_KINDS:
            legend, unit = PLOT_KINDS[kind]
            rows.append((s.name, [(s, legend)], unit))
    rows.append(("encoder", [(by_name["encoder"], ("angle",))], "deg"))
    rows.append(("FSRs", [(by_name["toe_fsr"], ("toe",)),
                          (by_name["heel_fsr"], ("heel",))], "raw ADC counts"))

    all_t = [t for s in streams for t in s.times]
    if not all_t:
        print("Nothing recorded, no plot.")
        return
    mid = (min(all_t) + max(all_t)) / 2 - T_START
    fig, axs = plt.subplots(len(rows), 2, squeeze=False,
                            figsize=(15, 2.6 * len(rows)))
    for r, (title, members, unit) in enumerate(rows):
        for c, zoom in ((0, False), (1, True)):
            ax = axs[r][c]
            lo, hi = [], []
            for s, legend in members:
                t = [x - T_START for x in s.times]
                for k, lab in enumerate(legend):
                    y = [v[k] for v in s.values]
                    if zoom:
                        ax.plot(t, y, ".-", ms=4, lw=0.8, label=lab)
                        win = [yy for tt, yy in zip(t, y) if mid <= tt <= mid + ZOOM_S]
                        lo += win[:1] and [min(win)]
                        hi += win[:1] and [max(win)]
                    else:
                        ax.plot(t, y, lw=0.8, label=lab)
            if zoom:
                ax.set_xlim(mid, mid + ZOOM_S)
                if lo:
                    pad = max(1e-3, (max(hi) - min(lo)) * 0.1)
                    ax.set_ylim(min(lo) - pad, max(hi) + pad)
            ax.set_title(title + (f" (zoom {ZOOM_S:.1f} s, dots = samples)"
                                  if zoom else ""))
            ax.set_ylabel(unit)
            ax.grid(alpha=0.3)
            ax.legend(loc="upper right", fontsize=8)
    for ax in axs[-1]:
        ax.set_xlabel("time since script start (s)")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    print(f"Saved plot to {path}")
    if os.environ.get("DISPLAY"):
        plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bus", type=int, default=1)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--hz", type=float, default=200.0,
                    help="IMU report rate, and the target for every stream")
    ap.add_argument("--enc-hz", type=float, default=None,
                    help="encoder read rate (default --hz, 0 = every loop)")
    ap.add_argument("--fsr-hz", type=float, default=None,
                    help="FSR sample rate, both channels (default --hz, 0 = flat out)")
    ap.add_argument("--quat", action="store_true",
                    help="IMUs also send the game rotation vector")
    ap.add_argument("--csv", nargs="?", const="", metavar="CSV",
                    help="save every sample (default all_sensors_<date>_<time>.csv)")
    ap.add_argument("--plot", nargs="?", const="all_sensors.png", metavar="PNG",
                    help="plot every sample (default all_sensors.png)")
    args = ap.parse_args()

    enc_hz = args.hz if args.enc_hz is None else args.enc_hz
    fsr_hz = args.hz if args.fsr_hz is None else args.fsr_hz
    features = (ACCEL, GYRO, QUAT) if args.quat else (ACCEL, GYRO)
    keep = args.csv is not None or args.plot is not None

    # ── connect ──────────────────────────────────────────────────────────────
    imus = {}
    imu_streams = {}
    for name, addr in IMU_ADDRS.items():
        print(f"Connecting IMU-{name} (0x{addr:02X}) on bus {args.bus}...")
        imus[name] = LeanBNO085(args.bus, addr, args.hz, features,
                                label=f"IMU-{name}")
        imu_streams[name] = {f: Stream(f"{name}_{NAMES[f]}", args.hz, keep)
                             for f in features}
        attach_imu_streams(imus[name], imu_streams[name])

    bus = smbus2.SMBus(args.bus)
    enc = Encoder(bus, Stream("encoder", enc_hz or None, keep))
    ads = ADS1115(bus, Stream("toe_fsr", fsr_hz or None, keep),
                  Stream("heel_fsr", fsr_hz or None, keep))

    streams = [s for per in imu_streams.values() for s in per.values()]
    streams += [enc.stream, *ads.streams]

    imu_errors = {name: 0 for name in imus}

    def poll_imus():
        for name, imu in imus.items():
            try:
                imu.poll()
            except (OSError, RuntimeError, KeyError, IndexError):
                imu_errors[name] += 1
            # Check the ADC between IMU polls too, so a finished conversion
            # waits at most one IMU read instead of a whole loop pass.
            ads.step(time.perf_counter())

    enc_period = 1.0 / enc_hz if enc_hz > 0 else 0.0
    fsr_period = 1.0 / fsr_hz if fsr_hz > 0 else 0.0

    def run(duration):
        now = time.perf_counter()
        end = now + duration
        next_enc = next_fsr = now
        while now < end:
            if now >= next_enc:
                enc.read()
                next_enc = max(next_enc + enc_period, now) if enc_period else now
            ads.step(now)
            if ads.idle and now >= next_fsr:
                ads.start()
                next_fsr = max(next_fsr + fsr_period, now) if fsr_period else now
            poll_imus()
            now = time.perf_counter()

    # Warm up: drain the IMU backlog from setup, then reset every counter.
    run(0.5)
    for s in streams:
        s.clear()
    for imu in imus.values():
        imu.reset_stats()
    enc.busy_s = ads.busy_s = 0.0
    enc.errors = ads.errors = 0
    for k in imu_errors:
        imu_errors[k] = 0

    print(f"\nReading all sensors for {args.seconds:.1f} s "
          f"(IMUs {args.hz:.0f} Hz, encoder "
          f"{f'{enc_hz:.0f} Hz' if enc_hz else 'every loop'}, FSR "
          f"{f'{fsr_hz:.0f} Hz' if fsr_hz else 'flat out'})...\n")
    try:
        t0 = time.perf_counter()
        run(args.seconds)
        t1 = time.perf_counter()
    finally:
        for imu in imus.values():
            imu.close()
        bus.close()
    elapsed = t1 - t0

    # ── report ───────────────────────────────────────────────────────────────
    print(f"Elapsed: {elapsed:.2f} s\n")
    print(f"{'stream':<14}{'Hz':>8}{'target':>8}{'p99 ms':>9}{'max ms':>9}")
    print("-" * 52)
    ok = True
    rates = []
    for s in streams:
        hz = len(s.times) / elapsed
        rates.append(hz)
        p99, mx = gap_stats(s.times)
        tgt = f"{s.target_hz:.0f}" if s.target_hz else "max"
        verdict = ""
        if s.target_hz:
            passed = hz >= s.target_hz * PASS_FRACTION
            ok &= passed
            verdict = "PASS" if passed else "FAIL"
        print(f"{s.name:<14}{hz:>8.1f}{tgt:>8}{p99:>9.1f}{mx:>9.1f}   {verdict}")

    print(f"\n{'device':<14}{'bus ms/s':>9}{'errors':>8}   notes")
    print("-" * 52)
    total_busy = 0.0
    for name, imu in imus.items():
        total_busy += imu.busy_s
        ok &= imu.seq_gaps == 0
        print(f"{'IMU-' + name:<14}{imu.busy_s / elapsed * 1e3:>9.0f}"
              f"{imu_errors[name]:>8}   lost packets {imu.seq_gaps}, "
              f"empty polls/s {imu.empty_polls / elapsed:.0f}")
    for label, dev in (("encoder", enc), ("ADS1115", ads)):
        total_busy += dev.busy_s
        print(f"{label:<14}{dev.busy_s / elapsed * 1e3:>9.0f}{dev.errors:>8}")
    print(f"{'total':<14}{total_busy / elapsed * 1e3:>9.0f}"
          f"{'':>8}   ({total_busy / elapsed * 100:.0f}% of the bus on data; "
          f"IMU empty polls use the rest)")

    fresh = fresh_row_fraction(streams, t0, t1, args.hz)
    print(f"\nEffective rate (slowest stream): {min(rates):.1f} Hz")
    print(f"Fully fresh rows at {args.hz:.0f} Hz: {fresh:.1f}% of ticks "
          f"(= {fresh / 100 * args.hz:.0f} Hz of rows where every sensor is new)")
    print()
    print("PASS: every stream at target." if ok else
          "FAIL: a stream is below target or IMU packets were lost.")

    here = Path(__file__).resolve().parent
    if args.csv is not None:
        name = args.csv or f"all_sensors_{datetime.now():%Y%m%d_%H%M%S}.csv"
        path = Path(name)
        write_csv(streams, path if path.is_absolute() else here / path)

    if args.plot is not None:
        print(f"\n{'stream':<14}{'samples':>9}{'repeated':>10}{'distinct':>10}")
        print("-" * 43)
        for s in streams:
            pct, distinct = repeat_stats(s.values)
            print(f"{s.name:<14}{len(s.values):>9}{pct:>9.1f}%{distinct:>10}")
        path = Path(args.plot)
        plot_streams(streams, path if path.is_absolute() else here / path)


if __name__ == "__main__":
    main()
