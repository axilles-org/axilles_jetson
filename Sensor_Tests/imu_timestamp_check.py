#!/usr/bin/env python3
"""
imu_timestamp_check.py — decode the BNO085's own timestamps and sequence numbers.

Every input packet the BNO085 sends looks like
    SHTP header (4 B: length, channel, channel sequence)
    0xFB base timestamp   (int32, 100 us ticks: how long before the host-interrupt
                           reference point the batch's base time lies)
    report(s):  [id, report sequence, status, delay, data...]
                id 0x01 accelerometer, 0x02 gyroscope (10 bytes each)
                sequence  8-bit, counted SEPARATELY for each report id
                delay     14 bits (status bits 7..2 = high, byte 3 = low), 100 us ticks,
                          added to the base time (CEVA sh2.c: t = t_int - base + delay)
    0xFA timestamp rebase (int32, 100 us) may appear when the base wraps

This script reads the IMUs exactly like the monitors (LeanBNO085, same rate),
captures every packet before it is parsed, and reports per IMU and per report:

    LOST      reports missing by that report's OWN sequence number (true loss)
    CH LOST   packets missing by the SHTP channel sequence (any report)
    sensor-time interval   spacing of samples on the IMU's clock
                           (host receive time - base + delay; error < 1 poll,
                           as the INT pin is not wired and receive time stands in for it)
    host interval          spacing of when the Jetson received them
    accel/gyro phase       how far each gyro sample is from the nearest accel sample

Usage:  python3 imu_timestamp_check.py --duration 30 [--sensors foot,shank] [--hz 100]
Writes logs/ts_<sensors>_<time>.csv (one row per report). Bench only needs the IMUs.
"""

import argparse
import csv
import struct
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import sensor_dropout_monitor as sdm                      # noqa: E402
from bno085_lean import ACCEL, GYRO, LeanBNO085            # noqa: E402

sdm.BUS_USERS = sdm.BUS_USERS + ("imu_timestamp_check.py", "true_dropout_monitor.py",
                                 "imu_only_dropout_monitor.py", "imu_diag.py")
IMU_ADDRS = {"foot": sdm.FOOT_ADDR, "shank": sdm.SHANK_ADDR}
NAMES = {0x01: "acc", 0x02: "gyro"}
TICK = 100e-6


def parse_cargo(cargo):
    """-> (base_s or None, [(report_id, seq, delay_s)], leftover_bytes)."""
    i, base, rebase, out = 0, None, 0.0, []
    while i < len(cargo):
        rid = cargo[i]
        if rid == 0xFB and i + 5 <= len(cargo):
            base = struct.unpack_from("<i", cargo, i + 1)[0] * TICK
            i += 5
        elif rid == 0xFA and i + 5 <= len(cargo):
            rebase = struct.unpack_from("<i", cargo, i + 1)[0] * TICK
            i += 5
        elif rid in NAMES and i + 10 <= len(cargo):
            seq, status, dlo = cargo[i + 1], cargo[i + 2], cargo[i + 3]
            delay = (((status >> 2) << 8) | dlo) * TICK
            out.append((rid, seq, delay))
            i += 10
        else:
            return base, out, len(cargo) - i
    return (None if base is None else base + rebase), out, 0


def stats(x):
    if not x:
        return "n/a"
    x = sorted(x)
    pick = lambda q: x[min(len(x) - 1, int(q * (len(x) - 1)))]
    return (f"median {pick(.5) * 1e3:6.2f}  p1 {pick(.01) * 1e3:6.2f}  p99 {pick(.99) * 1e3:6.2f}  "
            f"min {x[0] * 1e3:6.2f}  max {x[-1] * 1e3:6.2f} ms")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sensors", default="foot,shank", help="foot and/or shank")
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--hz", type=float, default=sdm.IMU_HZ, help="report rate requested")
    ap.add_argument("--out-dir", default=str(HERE / "logs"))
    args = ap.parse_args()
    imus = [p for p in args.sensors.split(",") if p in IMU_ADDRS]
    if not imus:
        ap.error("--sensors takes foot and/or shank")
    busy = sdm.other_bus_user()
    if busy:
        sys.exit(f"Another process is using I2C bus {sdm.I2C_BUS}; stop it first:\n  {busy}")

    rec = []                                   # (imu, t_host, ch_seq, base, rid, seq, delay)
    recording = [False]                        # off while draining start-up backlog
    leftovers = defaultdict(int)
    dev = {}
    for p in imus:
        imu = LeanBNO085(sdm.I2C_BUS, IMU_ADDRS[p], args.hz, (ACCEL, GYRO), label=f"IMU-{p}")
        orig = imu._handle

        def hooked(channel, seq, cargo, _p=p, _orig=orig):
            if channel == 3 and recording[0]:
                t = time.perf_counter()
                base, reps, left = parse_cargo(bytes(cargo))
                if left:
                    leftovers[_p] += 1
                for rid, rseq, delay in reps:
                    rec.append((_p, t, seq, base, rid, rseq, delay))
            return _orig(channel, seq, cargo)
        imu._handle = hooked
        dev[p] = imu
    # An IMU initialised first is not read while the next one initialises (~2 s), so its
    # FIFO fills and the BNO085 overwrites the oldest reports. Drain that backlog first,
    # otherwise it shows up as hundreds of "lost" reports at t = 0.
    t_drain = time.perf_counter() + 0.5
    while time.perf_counter() < t_drain:
        for imu in dev.values():
            try:
                imu.poll()
            except (OSError, KeyError, RuntimeError, IndexError, struct.error):
                pass
    recording[0] = True
    print(f"[ts] start-up backlog drained; reading {', '.join(imus)} at {args.hz:g} Hz "
          f"for {args.duration:g} s ...", flush=True)

    errors = defaultdict(int)
    t_end = time.perf_counter() + args.duration
    while time.perf_counter() < t_end:
        for p, imu in dev.items():
            for _ in range(sdm.MAX_POLLS):
                try:
                    if not imu.poll():
                        break
                except (OSError, KeyError, RuntimeError, IndexError, struct.error) as e:
                    errors[(p, type(e).__name__)] += 1
                    break
    for imu in dev.values():
        imu.close()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"ts_{'-'.join(imus)}_{datetime.now():%Y%m%d_%H%M%S}.csv"
    t0 = rec[0][1] if rec else 0.0
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["imu", "report", "t_host_s", "shtp_seq", "report_seq", "base_ms", "delay_ms",
                    "t_sensor_s"])
        for p, t, cs, base, rid, rs, d in rec:
            ts = t - (base or 0.0) + d
            w.writerow([p, NAMES[rid], f"{t - t0:.6f}", cs, rs, f"{(base or 0) * 1e3:.1f}",
                        f"{d * 1e3:.1f}", f"{ts - t0:.6f}"])

    print(f"\nreports captured: {len(rec)}   ({path})")
    for p in imus:
        print(f"\n=== {p} IMU ===")
        r = [x for x in rec if x[0] == p]
        # SHTP channel sequence, one per packet
        pk, lost_ch, last = [], 0, None
        for x in r:
            if pk and pk[-1] == (x[1], x[2]):
                continue
            pk.append((x[1], x[2]))
            if last is not None:
                lost_ch += (x[2] - last - 1) & 0xFF
            last = x[2]
        print(f"  packets {len(pk)},  CH LOST (SHTP channel sequence) {lost_ch},  "
              f"unparsed packet tails {leftovers[p]},  read errors "
              f"{dict((k[1], v) for k, v in errors.items() if k[0] == p) or 'none'}")
        per = {}
        for rid, name in NAMES.items():
            rr = [x for x in r if x[4] == rid]
            per[name] = rr
            if len(rr) < 2:
                print(f"  {name}: {len(rr)} reports")
                continue
            lost, gaps = 0, []
            for a, b in zip(rr, rr[1:]):
                d = (b[5] - a[5] - 1) & 0xFF
                if d:
                    lost += d
                    gaps.append(f"{b[1] - t0:.2f}s:{d}")
            ts = [x[1] - (x[3] or 0.0) + x[6] for x in rr]
            dur = rr[-1][1] - rr[0][1]
            print(f"  {name:4s}: {len(rr)} reports, {len(rr) / dur:6.1f} Hz,  "
                  f"LOST by its own sequence number: {lost}" + (f"  at {', '.join(gaps[:8])}" if gaps else ""))
            print(f"        sensor-time interval  {stats([b - a for a, b in zip(ts, ts[1:])])}")
            print(f"        host  receive interval {stats([b[1] - a[1] for a, b in zip(rr, rr[1:])])}")
            print(f"        age at receipt        {stats([(x[3] or 0.0) - x[6] for x in rr])}")
        if per.get("acc") and per.get("gyro"):
            ta = sorted(x[1] - (x[3] or 0.0) + x[6] for x in per["acc"])
            import bisect
            off = []
            for x in per["gyro"]:
                tg = x[1] - (x[3] or 0.0) + x[6]
                j = bisect.bisect_left(ta, tg)
                cand = [abs(tg - ta[k]) for k in (j - 1, j) if 0 <= k < len(ta)]
                off.append(min(cand))
            print(f"  accel/gyro phase: each gyro sample is {stats(off)} from the nearest accel sample")
            print(f"        accel and gyro arrive in separate packets: "
                  f"{len({(x[1], x[2]) for x in per['acc']} & {(x[1], x[2]) for x in per['gyro']})} "
                  f"packets carried both")


if __name__ == "__main__":
    main()
