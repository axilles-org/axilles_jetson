#!/usr/bin/env python3
"""
i2c_bench.py

Find out WHY every sensor is stuck at 20-30 Hz, before changing anything.

Run on the Jetson, exo connected (it does not need to be worn):

    python3 i2c_bench.py

walk_check showed foot 19.4, shank 19.4, encoder 31.7, FSRs ~17 Hz - about 104
sensor updates per second in total. Everything degrading together points at a
throughput ceiling rather than at thread scheduling, but "throughput" has several
possible causes and they need different fixes:

  bus clock        100 kHz vs 400 kHz is a 4x difference in payload capacity.
  payload size     Each IMU is configured for accel + gyro + GAME_ROTATION_VECTOR.
                   The quaternion is never used - Georgia Tech has none, so no
                   model feature can consume it - yet it costs bus time every read.
  driver overhead  adafruit_bno08x parses packets in Python. If a single isolated
                   read is already slow, the bus is not the limit; the CPU is.
  contention       Python threads serialise on the GIL, so three threads doing
                   Python-heavy I2C work may not overlap at all.

Each is measured separately below, so the fix follows from a number rather than a
guess. Nothing here writes to the sensors beyond normal reads.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import threading
import time
from pathlib import Path


def bus_clock_hz():
    """Best-effort read of the I2C bus speed from sysfs or the device tree."""
    candidates = [
        "/sys/bus/i2c/devices/i2c-1/of_node/clock-frequency",
        "/proc/device-tree/i2c@c240000/clock-frequency",
        "/proc/device-tree/bpmp/i2c/clock-frequency",
    ]
    for path in candidates:
        p = Path(path)
        if p.exists():
            try:
                raw = p.read_bytes()
                if len(raw) == 4:                      # big-endian device-tree cell
                    return int.from_bytes(raw, "big"), path
                return int(raw.decode().strip()), path
            except Exception:
                continue
    return None, None


def timeit(fn, seconds=3.0):
    """Call fn as fast as possible for `seconds`; return (calls/s, median ms, p95 ms)."""
    lat = []
    n = 0
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        t0 = time.perf_counter()
        try:
            fn()
        except Exception:
            pass
        lat.append(time.perf_counter() - t0)
        n += 1
    if not lat:
        return 0.0, 0.0, 0.0
    lat.sort()
    return (n / seconds, statistics.median(lat) * 1e3,
            lat[min(len(lat) - 1, int(0.95 * len(lat)))] * 1e3)


def run_parallel(tasks, seconds=3.0):
    """Run each task in its own thread for `seconds`; return {name: calls/s}."""
    stop = threading.Event()
    counts = {name: 0 for name in tasks}
    lock = threading.Lock()

    def runner(name, fn):
        local = 0
        while not stop.is_set():
            try:
                fn()
            except Exception:
                pass
            local += 1
        with lock:
            counts[name] = local

    threads = [threading.Thread(target=runner, args=(n, f), daemon=True)
               for n, f in tasks.items()]
    for t in threads:
        t.start()
    time.sleep(seconds)
    stop.set()
    for t in threads:
        t.join(timeout=2.0)
    return {n: c / seconds for n, c in counts.items()}


def header(title):
    print("\n" + "=" * 74)
    print(f"  {title}")
    print("=" * 74)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=3.0,
                    help="duration of each individual measurement")
    args = ap.parse_args()

    print("=" * 74)
    print("  I2C THROUGHPUT BENCHMARK")
    print("=" * 74)

    clock, src = bus_clock_hz()
    if clock:
        print(f"  bus clock: {clock / 1000:.0f} kHz   (from {src})")
        if clock <= 100_000:
            print("  ^ 100 kHz. Raising this to 400 kHz is the single biggest lever")
            print("    available, and costs nothing but a device-tree setting.")
    else:
        print("  bus clock: could not read it from sysfs or the device tree.")
        print("  Check manually:  sudo i2cdetect -F 1")

    # Import the hub the same way the calibration does.
    import importlib.util
    path = Path(__file__).resolve().parent.parent / "Data collection" / "data_collection.py"
    if not path.exists():
        print(f"\nCannot find {path}.")
        return 1
    spec = importlib.util.spec_from_file_location("_dc", path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except ImportError as exc:
        print(f"\nSensor drivers unavailable ({exc}). Run this on the Jetson.")
        return 1

    print(f"\n  data_collection.py IMU_REPORT_HZ = {mod.IMU_REPORT_HZ}")
    mod.IMU_REPORT_HZ = 200.0
    print("  overridden to 200.0 for this benchmark")

    hub = mod.SensorHub()

    try:
        # ---------------- 1. each sensor alone ----------------
        header("1. EACH SENSOR ALONE  (no competition - this is the ceiling)")
        print(f"  {'sensor':<16}{'rate':>10}{'median':>10}{'p95':>10}")
        alone = {}
        for name, fn in (("foot IMU", hub.read_imu_foot),
                         ("shank IMU", hub.read_imu_shank),
                         ("encoder", hub.read_encoder),
                         ("FSR pair", hub.read_fsr)):
            r, med, p95 = timeit(fn, args.seconds)
            alone[name] = r
            print(f"  {name:<16}{r:>9.1f}H{med:>9.2f}m{p95:>9.2f}m")

        slowest = min(alone, key=alone.get)
        print(f"\n  slowest alone: {slowest} at {alone[slowest]:.1f} Hz")
        if alone["foot IMU"] < 100:
            print("  The foot IMU cannot reach 100 Hz even with the bus to itself.")
            print("  That is not contention - it is the driver or the bus clock.")
        else:
            print("  Each sensor can individually exceed what it achieves together,")
            print("  so the loss is contention between them.")

        # ---------------- 2. both IMUs together ----------------
        header("2. BOTH IMUs TOGETHER  (what the rotation fit depends on)")
        both = run_parallel({"foot IMU": hub.read_imu_foot,
                             "shank IMU": hub.read_imu_shank}, args.seconds)
        for n, r in both.items():
            drop = 100 * (1 - r / alone[n]) if alone[n] else 0
            print(f"  {n:<16}{r:>9.1f}H   ({drop:+.0f}% vs alone)")

        # ---------------- 3. everything at once ----------------
        header("3. EVERYTHING AT ONCE  (what walk_check actually ran)")
        allr = run_parallel({"foot IMU": hub.read_imu_foot,
                             "shank IMU": hub.read_imu_shank,
                             "encoder": hub.read_encoder,
                             "FSR pair": hub.read_fsr}, args.seconds)
        total = sum(allr.values())
        for n, r in allr.items():
            print(f"  {n:<16}{r:>9.1f}H")
        print(f"  {'TOTAL':<16}{total:>9.1f} reads/s aggregate")

        # ---------------- 4. is the quaternion costing us? ----------------
        header("4. DOES DROPPING THE QUATERNION HELP?")
        print("  Georgia Tech has no quaternions, so no model feature can use one.")
        print("  Re-enabling the foot IMU with accel + gyro only, and re-measuring:\n")
        try:
            from adafruit_bno08x import (BNO_REPORT_ACCELEROMETER,
                                         BNO_REPORT_GYROSCOPE)

            # Taken off the loaded module, not re-imported: the package exports
            # ExtendedI2C and data_collection.py aliases it to ExtI2C, so importing
            # the alias name here fails.
            i2c = mod.ExtI2C(mod.I2C_BUS)
            bno = mod.BNO08X_I2C(i2c, address=mod.IMU_FOOT_ADDR)
            time.sleep(mod.BOOT_DELAY)
            interval = max(1000, int(1_000_000 / 200.0))
            for feat in (BNO_REPORT_ACCELEROMETER, BNO_REPORT_GYROSCOPE):
                bno.enable_feature(feat, interval)

            def read_two():
                bno._process_available_packets(max_packets=1)
                return (bno._readings.get(BNO_REPORT_ACCELEROMETER),
                        bno._readings.get(BNO_REPORT_GYROSCOPE))

            r2, med2, _ = timeit(read_two, args.seconds)
            gain = 100 * (r2 / alone["foot IMU"] - 1) if alone["foot IMU"] else 0
            print(f"  accel + gyro only : {r2:>9.1f}H  ({med2:.2f} ms median)")
            print(f"  with quaternion   : {alone['foot IMU']:>9.1f}H")
            print(f"  change            : {gain:+.0f}%")
            if gain > 20:
                print("\n  Worth doing: drop GAME_ROTATION_VECTOR from _FastIMU._connect")
                print("  and stop requiring quat in _FastIMU.read.")
            else:
                print("\n  Not the bottleneck. The cost is elsewhere.")
            try:
                i2c.deinit()
            except Exception:
                pass
        except Exception as exc:
            print(f"  could not test: {exc}")

        # ---------------- verdict ----------------
        header("WHAT THIS MEANS")
        need = 100.0
        if alone["foot IMU"] < need:
            print(f"  A single IMU alone reaches only {alone['foot IMU']:.0f} Hz, below the")
            print(f"  {need:.0f} Hz the gait cycle needs. Contention is not the main problem.")
            print("  In order of expected payoff:")
            print("    1. Raise the I2C bus clock to 400 kHz if it is at 100 kHz.")
            print("    2. Drop the unused quaternion report (see section 4).")
            print("    3. Move one IMU to a second I2C bus so they stop sharing.")
        elif allr["foot IMU"] < need <= both["foot IMU"]:
            print("  Both IMUs together are fine; adding the encoder and FSRs is what")
            print("  breaks it. Give the encoder and FSRs their own slower threads")
            print(f"  rather than polling them at 200 Hz - the ADS1115 in particular")
            print("  cannot benefit from being polled faster than it converts.")
        elif allr["foot IMU"] < need:
            print("  Each sensor is fast alone but slow together, so this is pure")
            print("  contention - bus arbitration, or the GIL serialising the Python")
            print("  parsing in adafruit_bno08x. Try the bus clock first, then")
            print("  splitting the sensors across two buses.")
        else:
            print("  Everything reaches target here, which contradicts walk_check.")
            print("  The difference is the sampling loop, not the sensors - re-run")
            print("  walk_check and compare.")
        return 0
    finally:
        hub.close()


if __name__ == "__main__":
    sys.exit(main())
