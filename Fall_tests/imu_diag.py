#!/usr/bin/env python3
"""
imu_diag.py -- catch BNO085 dropouts and find out WHY they happen.

Streams both IMUs exactly like fall_logger.py (LeanBNO085, 200 Hz accel+gyro)
and reads the AS5600 encoder every pass as a bus-health reference. For each
IMU it logs:

    QUIET     no report for > --quiet-ms while the IMU still ACKs (it stopped streaming)
    NAK       first failed read (Errno 121 etc.): the IMU stopped answering
    BACK      IMU answers again; how long it was gone
    CAUSE     reset cause read BEFORE re-initialising (SH-2 product-ID byte 1):
                1 power-on / brown-out   -> lost power or VCC/GND contact
                2 internal system reset  3 watchdog   4 external (RST pin)   5 other
              plus whether the boot signature (ch0 advertisement, ch1 reset-complete)
              was waiting in the FIFO, i.e. the chip really rebooted
    PARSE     KeyError / RuntimeError while the IMU still ACKs (garbled or
              unexpected packet), with the raw bytes of the last read
    ENC_ERR   encoder read failed (would point at the shared bus, not one IMU)

Recovery never blocks the other sensors while an IMU is down: only the final
re-initialisation (about 1 s) pauses the loop, and that is logged.

usage (system python3, no motor needed):
    python3 imu_diag.py --seconds 120 --label bench_tap_foot
    python3 imu_diag.py --seconds 300 --label worn_walk_1.0
Events go to Fall_tests/diag_runs/<timestamp>_<label>.csv
"""
import argparse
import csv
import os
import signal
import struct
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pawan"))
sys.path.insert(0, str(HERE))

import smbus2  # noqa: E402

import bno085_lean  # noqa: E402
from bno085_lean import LeanBNO085  # noqa: E402

IMUS = {"foot": 0x4A, "shank": 0x4B}
ENC_ADDR, ENC_REG = 0x36, 0x0E
CAUSES = {0: "n/a", 1: "power-on/brown-out", 2: "internal", 3: "watchdog",
          4: "external RST pin", 5: "other"}
_stop = False


def now():
    return time.monotonic_ns()


def read_packet(fd, timeout_s):
    t_end = time.monotonic() + timeout_s
    while time.monotonic() < t_end:
        hdr = os.read(fd, 4)
        ln = struct.unpack_from("<H", hdr)[0] & 0x7FFF
        if ln in (0, 0x7FFF):
            time.sleep(0.002)
            continue
        data = os.read(fd, ln)
        return data[2], bytes(data[4:ln])
    return None


def reset_cause(fd):
    """(cause, packets seen before the reply). Leaves sensor config untouched."""
    seen = []
    t_end = time.monotonic() + 0.4
    while time.monotonic() < t_end:
        p = read_packet(fd, 0.03)
        if p is None:
            break
        seen.append(f"ch{p[0]}:{len(p[1])}B:{p[1][:1].hex()}")
    os.write(fd, bytes([6, 0, 2, 0, 0xF9, 0x00]))
    t_end = time.monotonic() + 1.0
    while time.monotonic() < t_end:
        p = read_packet(fd, 0.2)
        if p and p[0] == 2 and p[1][:1] == b"\xf8":
            return p[1][1], seen
        if p:
            seen.append(f"ch{p[0]}:{len(p[1])}B:{p[1][:1].hex()}")
    return None, seen


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--label", default="diag")
    ap.add_argument("--bus", type=int, default=1)
    ap.add_argument("--imu-hz", type=float, default=200.0)
    ap.add_argument("--quiet-ms", type=float, default=30.0)
    ap.add_argument("--imus", default="foot,shank")
    args = ap.parse_args()
    signal.signal(signal.SIGINT, lambda *_: globals().__setitem__("_stop", True))
    signal.signal(signal.SIGTERM, lambda *_: globals().__setitem__("_stop", True))

    out_dir = HERE / "diag_runs"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"{datetime.now():%Y%m%d_%H%M%S}_{args.label}.csv"
    f = open(path, "w", newline="")
    w = csv.writer(f)
    w.writerow(["t_s", "imu", "event", "detail"])
    t0 = now()

    def ev(imu, event, detail=""):
        t = (now() - t0) / 1e9
        w.writerow([f"{t:.4f}", imu, event, detail])
        f.flush()
        print(f"{t:9.3f} s  {imu:6s} {event:8s} {detail}", flush=True)

    # capture the bytes of the most recent IMU read for PARSE events
    last_read = {}
    _orig_read = os.read

    def tracking_read(fd, n):
        data = _orig_read(fd, n)
        last_read[fd] = data
        return data
    bno085_lean.os.read = tracking_read

    names = [n for n in args.imus.split(",") if n]
    imus, st = {}, {}
    for n in names:
        imus[n] = LeanBNO085(bus=args.bus, address=IMUS[n], report_hz=args.imu_hz, label=n)
        cause, _ = reset_cause(imus[n]._fd)
        imus[n].reconnect()           # the cause query consumed reports; start clean
        ev(n, "START", f"cause after driver init = {cause} ({CAUSES.get(cause, '?')})")
        st[n] = dict(up=True, last_rep=now(), quiet=False, down_t=0, reports=0, naks=0,
                     parse=0, reboots=0, quiets=0)
    bus = smbus2.SMBus(args.bus)
    enc_ok = enc_err = 0
    print(f"monitoring {names} for {args.seconds:.0f} s -> {path}  (Ctrl+C stops)")

    t_end = now() + args.seconds * 1e9
    while not _stop and now() < t_end:
        for n, imu in imus.items():
            s = st[n]
            if s["up"]:
                try:
                    k = imu.poll()
                    if k:
                        if s["quiet"]:
                            ev(n, "STREAM", f"reports resumed after "
                               f"{(now() - s['last_rep']) / 1e6:.1f} ms quiet")
                        s["quiet"] = False
                        s["last_rep"] = now()
                        s["reports"] += k
                    elif not s["quiet"] and (now() - s["last_rep"]) / 1e6 > args.quiet_ms:
                        s["quiet"] = True
                        s["quiets"] += 1
                        ev(n, "QUIET", f"no report for {args.quiet_ms:.0f} ms, still ACKing")
                except OSError as e:
                    s["up"], s["down_t"] = False, now()
                    s["naks"] += 1
                    ev(n, "NAK", f"{e}; last report {(now() - s['last_rep']) / 1e6:.1f} ms before")
                except (KeyError, RuntimeError, IndexError, struct.error) as e:
                    s["parse"] += 1
                    raw = last_read.get(imu._fd, b"")[:24].hex()
                    ev(n, "PARSE", f"{type(e).__name__}: {e} raw={raw}")
            else:
                # down: probe with a single header read, never block the loop
                try:
                    os.read(imu._fd, 4)
                except OSError:
                    continue
                gone = (now() - s["down_t"]) / 1e6
                cause, seen = reset_cause(imu._fd)
                booted = any(x.startswith("ch0:") for x in seen) or any(x.startswith("ch1:") for x in seen)
                s["reboots"] += bool(booted)
                ev(n, "BACK", f"answered again after {gone:.1f} ms")
                ev(n, "CAUSE", f"{cause} ({CAUSES.get(cause, '?')}); rebooted={booted}; "
                   f"fifo={' '.join(seen) or 'empty'}")
                t_r = now()
                try:
                    imu.reconnect()
                    s["up"], s["last_rep"], s["quiet"] = True, now(), False
                    ev(n, "REINIT", f"loop paused {(now() - t_r) / 1e6:.0f} ms")
                except RuntimeError as e:
                    ev(n, "REINIT_FAIL", str(e))
                    s["down_t"] = now()
        try:
            bus.read_i2c_block_data(ENC_ADDR, ENC_REG, 2)
            enc_ok += 1
        except OSError as e:
            enc_err += 1
            ev("enc", "ENC_ERR", str(e))
        time.sleep(0.0005)

    dur = (now() - t0) / 1e9
    print(f"\n=== {dur:.1f} s  ({path.name})")
    for n, s in st.items():
        print(f"{n:6s} reports {s['reports']} ({s['reports'] / dur:.0f}/s)  quiet {s['quiets']}  "
              f"NAK {s['naks']}  reboots {s['reboots']}  parse errors {s['parse']}")
    print(f"encoder reads {enc_ok}, errors {enc_err}")
    w.writerow([f"{dur:.4f}", "all", "END",
                "; ".join(f"{n}: naks={s['naks']} reboots={s['reboots']} parse={s['parse']} "
                          f"quiet={s['quiets']}" for n, s in st.items()) + f"; enc_err={enc_err}"])
    f.close()
    for imu in imus.values():
        imu.close()


if __name__ == "__main__":
    main()
