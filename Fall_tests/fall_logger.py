#!/usr/bin/env python3
"""
fall_logger.py -- Jetson logger for the Fall Test Plan (Section 3).

Every run gets one folder, Fall_tests/runs/<run_id>/, containing:
    run.json      test number, run ID, date, condition, versions, notes, config
    samples.csv   one row per sensor read (time.monotonic_ns read-complete time,
                  raw value, converted value, read status, control-loop ID)
    control.csv   one row per control-loop pass (start/end time and the sample
                  ID of every channel the loop used)
    can.csv       CAN send/receive rows (only with --actuator)
    events.csv    driver restarts and other run events

Sensor reading reuses the single-threaded lean loop from
pawan/all_sensors_rate_test.py. All rows are kept in memory and written after
the run, so disk writes never delay the loop.

Examples
    # Test 1 step 1: one sensor alone, 5 min
    python3 fall_logger.py --test 1 --condition bench_encoder_alone --sensors encoder --seconds 300
    # Test 1 step 2 / Test 2 step 1: full set, still
    python3 fall_logger.py --test 1,2 --condition bench_all_still --seconds 300
    # Test 2 step 4: positive control, shank gyro frozen in software
    python3 fall_logger.py --test 2 --condition bench_freeze --freeze imu_shank_gyro --seconds 30
    # Test 1 step 4: walking with actuator enabled (needs can0 up)
    python3 fall_logger.py --test 1,2,3 --condition walk_1.0_act_on --speed 1.0 --actuator --seconds 300
"""

import argparse
import csv
import gc
import json
import os
import signal
import struct
import subprocess
import sys
import time
from array import array
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pawan"))

import smbus2  # noqa: E402

from bno085_lean import ACCEL, GYRO, LeanBNO085  # noqa: E402

IMU_ADDRS = {"foot": 0x4A, "shank": 0x4B}
ENC_ADDR = 0x36
ADS_ADDR = 0x48
_AS5600_REG_ANGLE = 0x0E
_ADS_REG_CONV = 0x00
_ADS_REG_CFG = 0x01
_ADS_CFG = {0: [0xC3, 0xE3], 1: [0xD3, 0xE3]}   # single-shot, +-4.096 V, 860 SPS
ADS_V_PER_COUNT = 4.096 / 32767.0
CONV_S = 0.0013

IMU_RECONNECT_AFTER = 10       # consecutive failed polls before a driver restart

# CAN (copied from Cubemars_AK80-9/ak_80_torque_input.py)
CAN_INTERFACE = "can0"
MOTOR_ID = 0x68
MODE_MIT = 8
MIT_P = (-12.56, 12.56)
MIT_V = (-65.0, 65.0)
MIT_T = (-22.0, 22.0)
MIT_KP = (0.0, 500.0)
MIT_KD = (0.0, 5.0)
TORQUE_CAP_NM = 2.0            # this logger never commands more than this

ALL_SENSORS = ("imu_foot", "imu_shank", "encoder", "fsr")
SAMPLE_HEADER = ["run_id", "sample_id", "channel", "t_read_ns", "raw",
                 "converted", "status", "loop_id"]


def now_ns():
    return time.monotonic_ns()


class Recorder:
    """In-memory sample table. loop_id is filled in when a loop uses a sample."""

    def __init__(self, channels, freeze=None):
        self.channels = list(channels)
        self.rows = []                 # (channel, t_ns, raw, converted, status)
        self.loop_of = array("q")      # first loop that used each sample, -1 = none
        self.latest = {ch: -1 for ch in self.channels}   # newest OK sample id
        self.freeze = freeze
        self._frozen = None

    def ok(self, channel, raw, converted):
        if channel == self.freeze:
            # Positive control: the driver keeps handing back its first value.
            if self._frozen is None:
                self._frozen = (raw, converted)
            raw, converted = self._frozen
        sid = len(self.rows)
        self.rows.append((channel, now_ns(), raw, converted, "ok"))
        self.loop_of.append(-1)
        self.latest[channel] = sid

    def fail(self, channel, err):
        self.rows.append((channel, now_ns(), "", "", f"fail:{type(err).__name__}"))
        self.loop_of.append(-1)


class Encoder:
    def __init__(self, bus, rec):
        self.bus, self.rec = bus, rec

    def read(self):
        try:
            d = self.bus.read_i2c_block_data(ENC_ADDR, _AS5600_REG_ANGLE, 2)
        except OSError as e:
            self.rec.fail("encoder", e)
            return
        raw = ((d[0] & 0x0F) << 8) | d[1]
        self.rec.ok("encoder", raw, f"{raw * 360.0 / 4096.0:.4f}")


class ADS1115:
    """Two channels, single-shot, timed conversions (as in Pawan's rate test)."""
    NAMES = ("fsr_toe", "fsr_heel")

    def __init__(self, bus, rec):
        self.bus, self.rec = bus, rec
        self.ch = None
        self.t_trigger = 0.0

    @property
    def idle(self):
        return self.ch is None

    def start(self):
        self._trigger(0)

    def _trigger(self, ch):
        try:
            self.bus.write_i2c_block_data(ADS_ADDR, _ADS_REG_CFG, _ADS_CFG[ch])
        except OSError as e:
            self.rec.fail(self.NAMES[ch], e)
            self.ch = None
            return
        self.ch = ch
        self.t_trigger = time.perf_counter()

    def step(self, now):
        if self.ch is None or now - self.t_trigger < CONV_S:
            return
        ch = self.ch
        try:
            d = self.bus.read_i2c_block_data(ADS_ADDR, _ADS_REG_CONV, 2)
        except OSError as e:
            self.rec.fail(self.NAMES[ch], e)
            self.ch = None
            return
        raw = (d[0] << 8) | d[1]
        raw = raw if raw < 0x8000 else raw - 0x10000
        self.rec.ok(self.NAMES[ch], raw, f"{max(0, raw) * ADS_V_PER_COUNT:.5f}")
        if ch == 0:
            self._trigger(1)
        else:
            self.ch = None


def attach_imu(imu, name, rec):
    """Record every accel/gyro report as it is parsed (batched ones included)."""
    bno = imu._bno
    orig = bno._process_report
    chans = {ACCEL: f"imu_{name}_accel", GYRO: f"imu_{name}_gyro"}

    def record(report_id, report_bytes):
        orig(report_id, report_bytes)
        ch = chans.get(report_id)
        if ch is not None:
            v = bno._readings[report_id]
            s = ";".join(f"{x:.6f}" for x in v[:3])
            rec.ok(ch, s, s)

    bno._process_report = record


# ── CAN ─────────────────────────────────────────────────────────────────────
def _f2u(x, lo_hi, bits):
    lo, hi = lo_hi
    x = max(lo, min(hi, x))
    return int((x - lo) / (hi - lo) * ((1 << bits) - 1))


def mit_frame(torque_nm):
    kp, kd = _f2u(0.0, MIT_KP, 12), _f2u(0.0, MIT_KD, 12)
    p, v = _f2u(0.0, MIT_P, 16), _f2u(0.0, MIT_V, 12)
    t = _f2u(torque_nm, MIT_T, 12)
    return [kp >> 4, ((kp & 0xF) << 4) | (kd >> 8), kd & 0xFF, p >> 8, p & 0xFF,
            v >> 4, ((v & 0xF) << 4) | (t >> 8), t & 0xFF]


class Actuator:
    HEADER = ["run_id", "loop_id", "dir", "t_ns", "torque_req_nm", "position_deg",
              "speed_erpm", "current_a", "temp_c", "error", "status"]

    def __init__(self, torque_nm):
        import can
        self.can = can
        self.torque = max(-TORQUE_CAP_NM, min(TORQUE_CAP_NM, torque_nm))
        self.bus = can.interface.Bus(channel=CAN_INTERFACE, interface="socketcan")
        self.rows = []

    def step(self, loop_id, torque=None):
        tq = self.torque if torque is None else torque
        msg = self.can.Message(arbitration_id=(MODE_MIT << 8) | MOTOR_ID,
                               data=mit_frame(tq), is_extended_id=True)
        try:
            self.bus.send(msg)
            status = "ok"
        except (self.can.CanError, OSError) as e:
            status = f"fail:{type(e).__name__}"
        self.rows.append((loop_id, "tx", now_ns(), f"{tq:.3f}", "", "", "", "", "", status))
        while True:                      # drain replies without blocking
            try:
                r = self.bus.recv(timeout=0)
            except (self.can.CanError, OSError) as e:
                self.rows.append((loop_id, "rx", now_ns(), "", "", "", "", "", "",
                                  f"fail:{type(e).__name__}"))
                break
            if r is None:
                break
            t = now_ns()
            if len(r.data) < 8:
                self.rows.append((loop_id, "rx", t, "", "", "", "", "", "", "short"))
                continue
            d = bytes(r.data)
            self.rows.append((loop_id, "rx", t, "",
                              f"{struct.unpack('>h', d[0:2])[0] * 0.1:.1f}",
                              f"{struct.unpack('>h', d[2:4])[0] * 10.0:.0f}",
                              f"{struct.unpack('>h', d[4:6])[0] * 0.01:.2f}",
                              d[6], d[7], "ok"))

    def close(self):
        for _ in range(3):
            self.step(-1, 0.0)
            time.sleep(0.005)
        self.bus.shutdown()


# ── main ────────────────────────────────────────────────────────────────────
def git_commit():
    try:
        return subprocess.check_output(
            ["git", "-C", str(HERE.parent), "rev-parse", "--short", "HEAD"],
            text=True).strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", required=True, help="test number(s), e.g. 1 or 1,2,3")
    ap.add_argument("--condition", required=True, help="short tag, e.g. walk_1.0_act_off")
    ap.add_argument("--seconds", type=float, default=300.0)
    ap.add_argument("--sensors", default=",".join(ALL_SENSORS),
                    help=f"subset of {','.join(ALL_SENSORS)}")
    ap.add_argument("--freeze", default=None,
                    help="positive control: hold this channel's value constant")
    ap.add_argument("--actuator", action="store_true",
                    help="enable the AK80-9 over CAN and log can.csv")
    ap.add_argument("--torque", type=float, default=0.0,
                    help=f"constant MIT torque command with --actuator (|t| <= {TORQUE_CAP_NM})")
    ap.add_argument("--speed", type=float, default=None, help="treadmill speed, m/s")
    ap.add_argument("--subject", default="", help="subject code (never a name)")
    ap.add_argument("--notes", default="")
    ap.add_argument("--hw", default="", help="hardware version")
    ap.add_argument("--cable", default="", help="cable type")
    ap.add_argument("--bus", type=int, default=1)
    ap.add_argument("--imu-hz", type=float, default=200.0)
    ap.add_argument("--enc-hz", type=float, default=400.0)
    ap.add_argument("--fsr-hz", type=float, default=200.0)
    ap.add_argument("--loop-hz", type=float, default=100.0, help="control-loop rate")
    ap.add_argument("--cpu", type=int, default=5,
                    help="pin the logger to this CPU core (-1 = no pinning)")
    ap.add_argument("--keep-gc", action="store_true",
                    help="leave Python garbage collection on while recording")
    args = ap.parse_args()

    sensors = [s.strip() for s in args.sensors.split(",") if s.strip()]
    bad = set(sensors) - set(ALL_SENSORS)
    if bad:
        ap.error(f"unknown sensors {bad}")

    channels = []
    for name in ("foot", "shank"):
        if f"imu_{name}" in sensors:
            channels += [f"imu_{name}_accel", f"imu_{name}_gyro"]
    if "encoder" in sensors:
        channels.append("encoder")
    if "fsr" in sensors:
        channels += list(ADS1115.NAMES)
    if args.freeze and args.freeze not in channels:
        ap.error(f"--freeze must be one of {channels}")

    run_id = f"T{args.test.replace(',', '-')}_{datetime.now():%Y%m%d_%H%M%S}_{args.condition}"
    out = HERE / "runs" / run_id
    out.mkdir(parents=True)
    rec = Recorder(channels, args.freeze)
    events = []

    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("flag", True))
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("flag", True))

    # connect
    imus = {}
    for name, addr in IMU_ADDRS.items():
        if f"imu_{name}" in sensors:
            print(f"Connecting IMU-{name} (0x{addr:02X})...")
            imus[name] = LeanBNO085(args.bus, addr, args.imu_hz, (ACCEL, GYRO),
                                    label=f"IMU-{name}")
            attach_imu(imus[name], name, rec)
    bus = smbus2.SMBus(args.bus)
    enc = Encoder(bus, rec) if "encoder" in sensors else None
    ads = ADS1115(bus, rec) if "fsr" in sensors else None
    act = Actuator(args.torque) if args.actuator else None
    imu_fail_streak = {n: 0 for n in imus}
    imu_restarts = {n: 0 for n in imus}

    def poll_imus(now):
        for name, imu in imus.items():
            try:
                imu.poll()
                imu_fail_streak[name] = 0
            except (OSError, RuntimeError, KeyError, IndexError) as e:
                rec.fail(f"imu_{name}_accel", e)
                rec.fail(f"imu_{name}_gyro", e)
                imu_fail_streak[name] += 1
                if imu_fail_streak[name] >= IMU_RECONNECT_AFTER:
                    events.append((now_ns(), "restart", f"imu_{name}", str(e)))
                    imu_restarts[name] += 1
                    try:
                        imu.reconnect()
                        attach_imu(imu, name, rec)
                    except RuntimeError as e2:
                        events.append((now_ns(), "restart_failed", f"imu_{name}", str(e2)))
                    imu_fail_streak[name] = 0
            if ads:
                ads.step(time.perf_counter())

    enc_p = 1.0 / args.enc_hz if args.enc_hz > 0 else 0.0
    fsr_p = 1.0 / args.fsr_hz if args.fsr_hz > 0 else 0.0
    loop_p = 1.0 / args.loop_hz
    control = []          # (loop_id, t_start, t_end, *sample ids)

    def run(duration, record_loops):
        now = time.perf_counter()
        end = now + duration
        next_enc = next_fsr = next_loop = now
        loop_id = 0
        while now < end and not stop["flag"]:
            if enc and now >= next_enc:
                enc.read()
                next_enc = max(next_enc + enc_p, now) if enc_p else now
            if ads:
                ads.step(now)
                if ads.idle and now >= next_fsr:
                    ads.start()
                    next_fsr = max(next_fsr + fsr_p, now) if fsr_p else now
            poll_imus(now)
            now = time.perf_counter()
            # Like the controller, start looping once every input has a first value.
            if (record_loops and now >= next_loop
                    and (loop_id or min(rec.latest.values()) >= 0)):
                t0 = now_ns()
                used = [rec.latest[ch] for ch in channels]
                for sid in used:
                    if sid >= 0 and rec.loop_of[sid] < 0:
                        rec.loop_of[sid] = loop_id
                if act:
                    act.step(loop_id)
                control.append((loop_id, t0, now_ns(), *used))
                loop_id += 1
                next_loop += loop_p
                if next_loop <= now:
                    next_loop = now + loop_p
                now = time.perf_counter()

    print("Warm-up 0.5 s...")
    run(0.5, record_loops=False)
    rec.rows.clear()
    rec.loop_of = array("q")
    rec.latest = {ch: -1 for ch in channels}
    for imu in imus.values():
        imu.reset_stats()

    # Stall reduction: a GC pass over millions of buffered rows, or the
    # scheduler moving the process between cores, can pause the loop for tens
    # of ms. Pin to one core and keep GC off until the rows are written.
    pinned = None
    if args.cpu >= 0:
        try:
            os.sched_setaffinity(0, {args.cpu})
            pinned = args.cpu
        except OSError as e:
            print(f"CPU pinning failed: {e}")
    if not args.keep_gc:
        gc.collect()
        gc.freeze()
        gc.disable()

    print(f"Recording {run_id} for {args.seconds:.0f} s  (Ctrl+C stops early)")
    t_start = now_ns()
    try:
        run(args.seconds, record_loops=True)
    finally:
        t_end = now_ns()
        gc.enable()
        if act:
            act.close()
        for imu in imus.values():
            imu.close()
        bus.close()

    # write files
    with open(out / "samples.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(SAMPLE_HEADER)
        for sid, (ch, t, raw, conv, st) in enumerate(rec.rows):
            w.writerow((run_id, sid, ch, t, raw, conv, st, rec.loop_of[sid]))
    with open(out / "control.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "loop_id", "t_start_ns", "t_end_ns",
                    *[f"sid_{ch}" for ch in channels]])
        for r in control:
            w.writerow((run_id, *r))
    if act:
        with open(out / "can.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(Actuator.HEADER)
            for r in act.rows:
                w.writerow((run_id, *r))
    with open(out / "events.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_ns", "event", "channel", "detail"])
        w.writerows(events)

    meta = {
        "test": args.test, "run_id": run_id,
        "date": datetime.now().isoformat(timespec="seconds"),
        "condition": args.condition, "treadmill_speed_mps": args.speed,
        "hardware_version": args.hw, "cable_type": args.cable,
        "software_commit": git_commit(), "model_hash": None,
        "feature_list_hash": None, "subject": args.subject, "notes": args.notes,
        "t_start_ns": t_start, "t_end_ns": t_end,
        "duration_s": (t_end - t_start) / 1e9,
        "stopped_early": stop["flag"],
        "channels": channels, "freeze": args.freeze,
        "actuator": args.actuator, "torque_cmd_nm": act.torque if act else None,
        "rates_hz": {"imu_report": args.imu_hz, "encoder": args.enc_hz,
                     "fsr": args.fsr_hz, "loop": args.loop_hz},
        "imu_restarts": imu_restarts,
        "imu_lost_packets": {n: imu.seq_gaps for n, imu in imus.items()},
        "n_samples": len(rec.rows), "n_loops": len(control),
        "gc_disabled": not args.keep_gc, "cpu_pinned": pinned,
    }
    (out / "run.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved {len(rec.rows)} samples, {len(control)} loops "
          f"in {meta['duration_s']:.1f} s -> {out}")


if __name__ == "__main__":
    main()
