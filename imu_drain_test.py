#!/usr/bin/env python3
"""
imu_drain_test.py

Test the actual hypothesis behind the 19.4 Hz IMUs, and measure each candidate fix.

Run on the Jetson, exo connected (need not be worn):

    python3 imu_drain_test.py

The hypothesis
--------------
data_collection.py enables THREE reports per IMU (quaternion, accel, gyro) and its
read() drains exactly ONE packet per call:

    self._bno._process_available_packets(max_packets=1)

At IMU_REPORT_HZ = 200 the sensor emits 3 x 200 = 600 packets/s. i2c_bench measured
65 read() calls/s, so 65 packets/s are consumed out of 600 produced. Since packets
rotate between the three report types, each individual feature lands at about
65 / 3 = 21.7 Hz - which is what walk_check saw (19.4 Hz).

If that is right, the bus is not the main limit. Two changes should each help, and
they compound:

    drain more per call   amortises the fixed per-call cost over several packets
    drop the quaternion   Georgia Tech has none, so no model feature can use it;
                          it is a third of the packets for nothing

This script measures four configurations and reports the PER-FEATURE update rate,
which is the number that actually matters - a high packet rate split across three
reports still leaves each one slow.

It only reads. Your data_collection.py is not modified.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

TARGET_HZ = 100.0        # what the gait cycle needs per feature


def load_dc():
    path = Path("Data collection") / "data_collection.py"
    if not path.exists():
        raise FileNotFoundError(f"Cannot find {path}. Run from the repository root.")
    spec = importlib.util.spec_from_file_location("_dc", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def measure(mod, address, features, max_packets, report_hz, seconds):
    """
    Open one BNO085 with the given features, then drain it as fast as possible.

    Returns (calls_per_s, per_feature_hz, median_ms). per_feature_hz counts how often
    each report's VALUE actually changes, which is the rate the gait cycle sees.
    """
    # Reuse the classes data_collection.py already imported rather than importing
    # them again here. The package exports ExtendedI2C, which that module aliases to
    # ExtI2C; importing the alias name directly fails, and taking them off the loaded
    # module means this can never drift from whatever actually works on the robot.
    BNO08X_I2C = mod.BNO08X_I2C
    i2c = mod.ExtI2C(mod.I2C_BUS)
    try:
        bno = BNO08X_I2C(i2c, address=address)
        time.sleep(mod.BOOT_DELAY)
        interval = max(1000, int(1_000_000 / report_hz))
        for feat in features:
            for attempt in range(5):
                try:
                    bno.enable_feature(feat, interval)
                    break
                except Exception:
                    if attempt == 4:
                        raise
                    time.sleep(0.2)

        # Warm up so FIFO startup transients do not skew the measurement.
        end = time.perf_counter() + 0.5
        while time.perf_counter() < end:
            try:
                bno._process_available_packets(max_packets=max_packets)
            except Exception:
                pass

        calls = 0
        changes = {f: 0 for f in features}
        last = {f: None for f in features}
        lat = []
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            t0 = time.perf_counter()
            try:
                bno._process_available_packets(max_packets=max_packets)
            except Exception:
                pass
            lat.append(time.perf_counter() - t0)
            calls += 1
            for f in features:
                v = bno._readings.get(f)
                if v is not None and v != last[f]:
                    changes[f] += 1
                    last[f] = v

        lat.sort()
        med = lat[len(lat) // 2] * 1e3 if lat else 0.0
        per_feature = min(changes.values()) / seconds if changes else 0.0
        return calls / seconds, per_feature, med, {f: c / seconds
                                                   for f, c in changes.items()}
    finally:
        try:
            i2c.deinit()
        except Exception:
            pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--report-hz", type=float, default=200.0)
    ap.add_argument("--which", choices=("foot", "shank"), default="foot")
    args = ap.parse_args()

    print("=" * 76)
    print("  IMU DRAIN TEST")
    print("=" * 76)

    try:
        mod = load_dc()
    except (FileNotFoundError, ImportError) as exc:
        print(f"{exc}\nRun this on the Jetson with the sensors attached.")
        return 1

    try:
        from adafruit_bno08x import (BNO_REPORT_ACCELEROMETER,
                                     BNO_REPORT_GAME_ROTATION_VECTOR,
                                     BNO_REPORT_GYROSCOPE)
    except ImportError as exc:
        print(f"Sensor drivers unavailable ({exc}). Run this on the Jetson.")
        return 1

    addr = mod.IMU_FOOT_ADDR if args.which == "foot" else mod.IMU_SHANK_ADDR
    quat, acc, gyr = (BNO_REPORT_GAME_ROTATION_VECTOR, BNO_REPORT_ACCELEROMETER,
                      BNO_REPORT_GYROSCOPE)

    configs = [
        ("current: 3 reports, 1 packet/call", [quat, acc, gyr], 1),
        ("drain 10 packets/call", [quat, acc, gyr], 10),
        ("drop quaternion, 1 packet/call", [acc, gyr], 1),
        ("drop quaternion + drain 10", [acc, gyr], 10),
        ("drop quaternion + drain 30", [acc, gyr], 30),
    ]

    print(f"  IMU: {args.which} at 0x{addr:02X}, report rate {args.report_hz:.0f} Hz")
    print(f"  each configuration measured for {args.seconds:.0f} s\n")
    print(f"  {'configuration':<36}{'calls/s':>9}{'per-feature':>13}{'median':>9}")
    print("  " + "-" * 67)

    results = {}
    for label, feats, mp in configs:
        try:
            calls, per_f, med, detail = measure(mod, addr, feats, mp,
                                                args.report_hz, args.seconds)
        except (ImportError, AttributeError, NameError) as exc:
            # A setup problem will fail identically for every configuration, so stop
            # rather than printing the same error five times.
            print(f"\n  Cannot open the IMU: {exc}")
            print("  This is a setup problem, not a measurement result, so the")
            print("  remaining configurations are skipped.")
            return 1
        except Exception as exc:
            print(f"  {label:<36}   failed: {exc}")
            continue
        results[label] = per_f
        flag = "  <- enough" if per_f >= TARGET_HZ else ""
        print(f"  {label:<36}{calls:>8.0f}H{per_f:>12.1f}H{med:>8.2f}m{flag}")
        time.sleep(0.3)

    print("\n" + "=" * 76)
    print("  VERDICT")
    print("=" * 76)
    if not results:
        print("  Nothing measured.")
        return 1

    base = results.get("current: 3 reports, 1 packet/call")
    best_label = max(results, key=results.get)
    best = results[best_label]

    if base:
        print(f"  current configuration gives {base:.1f} Hz per feature")
        print(f"  best measured here         : {best:.1f} Hz  ({best_label})")
        if base > 0:
            print(f"  improvement                : {best / base:.1f}x")
    print()
    if best >= TARGET_HZ:
        print(f"  That clears the {TARGET_HZ:.0f} Hz the gait cycle needs. The fix is a")
        print("  software change only - no rewiring, no bus-clock change.")
        print("\n  I will wire this into the calibration path so it is used")
        print("  automatically, leaving data_collection.py untouched.")
    else:
        print(f"  Still short of {TARGET_HZ:.0f} Hz per feature. Software alone is not")
        print("  enough, so the remaining options are hardware:")
        print("    - raise the I2C bus clock to 400 kHz")
        print("    - move one IMU onto a second I2C bus")
        print(f"  Alternatively accept {best:.0f} Hz and lower IMU_REPORT_HZ to match,")
        print("  which at least makes the data FRESH rather than a stale backlog.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
