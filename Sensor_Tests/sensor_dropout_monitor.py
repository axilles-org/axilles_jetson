#!/usr/bin/env python3
"""
sensor_dropout_monitor.py — shows which sensors actually delivered data,
in 10 ms (100 Hz) frames, and flags every I2C fault on bus 1.

WHY
───
The ML deploy keeps the last value of a sensor until a new one arrives, so a
dropout looks like a flat line, or is not visible at all.  Two powered walking
runs (2026-11-02) stopped on bus 1 faults at heel strike: a corrupt BNO085
packet (KeyError: 255) and an I2C transfer that hung until the kernel timed it
out (all four sensors froze together).  This script puts the same traffic on
the bus as the deploy, without the motor, and makes every missed reading and
every fault visible while someone walks or taps the leg.

WHAT IT READS  (bus 1, same order and rates as ML_model/src/exo/deploy/jetson_io.py)
    foot IMU   BNO085  0x4A  accel + gyro at 100 Hz, via pawan/bno085_lean.py
    shank IMU  BNO085  0x4B  accel + gyro at 100 Hz
    encoder    AS5600  0x36  angle register, once per pass
    FSRs       ADS1115 0x48  toe (AIN0) then heel (AIN1), 200 Hz single-shot

It never opens CAN and never talks to the motor.  It writes to I2C only to
configure the IMUs and start ADC conversions, as the deploy does.

Reading happens in a background thread.  The main thread prints one row per
10 ms frame, so a hung I2C call shows up as NaN rows plus a HUNG line instead
of freezing the output.

WHAT IT PRINTS
    rows     one per 10 ms frame.  A value is the newest reading that arrived
             in that frame.  NaN = no new reading in that frame; the last value
             is NOT repeated.
    GAP      a stream had no new reading for at least --gap-ms.
    HUNG     an I2C call is blocked right now (printed while it is stuck).
    STALL    an I2C call returned after blocking at least --stall-ms.
    I2CERR   an I2C call raised OSError (errno 110 = timed out, 121 = no ACK).
    BADPKT   a BNO085 packet that could not be parsed (e.g. report id 0xFF).
    BADENC   AS5600 set bits that are always 0, so the read was corrupt.
    SEQ      BNO085 packets missing, by sequence number.
    RECON    an IMU was silent for 1 s and was re-initialised.

A single NaN next to a frame that got two readings is the IMU clock drifting
against the Jetson clock, not a fault.  Look for GAP, HUNG/STALL, I2CERR and
BADPKT.

Do not run this alongside the deploy or another sensor script, and close the
dashboard's Health Monitor page first: a second reader takes bytes out of the
BNO085 stream and causes BADPKT on its own.

USAGE
    python3 sensor_dropout_monitor.py                  # every frame
    python3 sensor_dropout_monitor.py --only-misses    # only frames with a NaN
    python3 sensor_dropout_monitor.py --duration 60

Every frame (NaN included) is saved to logs/dropout_<time>.csv and every event
to logs/dropout_<time>_events.log, next to this script.
"""

import argparse
import csv
import math
import os
import struct
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import smbus2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pawan"))
from bno085_lean import ACCEL, GYRO, LeanBNO085          # noqa: E402

# ── Hardware constants ─────────────────────────────────────────────────────────
I2C_BUS          = 1
FOOT_ADDR        = 0x4A
SHANK_ADDR       = 0x4B
AS5600_ADDR      = 0x36
AS5600_REG_ANGLE = 0x0E

# Same rates as the deploy (jetson_io.JetsonSensors defaults).
IMU_HZ           = 100.0
FSR_HZ           = 200.0
MAX_POLLS        = 4         # packets drained from each IMU per pass

FRAME_S          = 0.010     # 100 Hz frames
FRAME_GRACE_S    = 0.002     # wait this long after a frame ends before judging it
SILENT_S         = 1.0       # an IMU this quiet is re-initialised
STARTUP_S        = 3.0       # wait for every stream's first reading

# Processes that also read bus 1 (dashboard/backend/health.py list, plus the
# sensor scripts in this folder).
BUS_USERS = ("jetson_deploy.py", "jetson_mock_deploy.py", "TBE_controller/main.py",
             "test4_run.py", "fall_logger.py", "sensor_parse.py",
             "live_all_sensor_plot.py", "sensor_dropout_monitor.py")

# ── Streams ────────────────────────────────────────────────────────────────────
STREAMS = ("foot_acc", "foot_gyro", "shank_acc", "shank_gyro", "encoder", "fsr")
CSV_COLS = {
    "foot_acc":   ("foot_ax", "foot_ay", "foot_az"),
    "foot_gyro":  ("foot_gx", "foot_gy", "foot_gz"),
    "shank_acc":  ("shank_ax", "shank_ay", "shank_az"),
    "shank_gyro": ("shank_gx", "shank_gy", "shank_gz"),
    "encoder":    ("encoder_deg",),
    "fsr":        ("toe_fsr_raw", "heel_fsr_raw"),
}
LABELS = {
    "foot_acc":   "foot acc m/s2",
    "foot_gyro":  "foot gyro rad/s",
    "shank_acc":  "shank acc m/s2",
    "shank_gyro": "shank gyro rad/s",
    "encoder":    "enc deg",
    "fsr":        "toe / heel",
}
CELL = 7

# ── Terminal colour ────────────────────────────────────────────────────────────
COLOR = sys.stdout.isatty()
RED, YELLOW, BOLD = "31", "33", "1"
EVENT_COLOR = {"HUNG": RED, "STALL": RED, "I2CERR": RED, "BADPKT": RED,
               "BADENC": RED, "CRASH": RED, "GAP": YELLOW, "SEQ": YELLOW,
               "RECON": YELLOW}


def paint(text, code):
    return f"\033[{code}m{text}\033[0m" if COLOR else text


# ── ADS1115 ────────────────────────────────────────────────────────────────────
class TimedADS1115:
    """Toe and heel FSRs, read exactly as jetson_io.TimedADS1115 does:
    single-shot at 860 SPS, AIN0 then AIN1, conversion timed instead of polled.
    Copied rather than imported so this script does not load the exo package."""

    ADDR = 0x48
    REG_CONV = 0x00
    REG_CFG = 0x01
    CFG = ([0xC3, 0xE3], [0xD3, 0xE3])     # ±4.096 V, 860 SPS, comparator off
    CONV_S = 0.0013                         # 1/860 s = 1.16 ms, +10% clock tolerance

    def __init__(self, bus, sample_hz):
        self._bus = bus
        self._period = 1.0 / sample_hz
        self._next = 0.0
        self._ch = None                     # channel converting, None when idle
        self._t_trigger = 0.0
        self._raw = [0, 0]

    def step(self, now):
        """Returns (toe, heel) raw counts when both channels are new."""
        if self._ch is None:
            if now >= self._next:
                self._next = max(self._next + self._period, now)
                self._trigger(0)
            return None
        if now - self._t_trigger < self.CONV_S:
            return None
        d = self._bus.read_i2c_block_data(self.ADDR, self.REG_CONV, 2)
        raw = (d[0] << 8) | d[1]
        self._raw[self._ch] = raw - 0x10000 if raw >= 0x8000 else raw
        if self._ch == 0:
            self._trigger(1)
            return None
        self._ch = None
        return self._raw[0], self._raw[1]

    def _trigger(self, ch):
        self._bus.write_i2c_block_data(self.ADDR, self.REG_CFG, self.CFG[ch])
        self._ch = ch
        self._t_trigger = time.perf_counter()


# ── Reader thread ──────────────────────────────────────────────────────────────
class Reader(threading.Thread):
    """Polls every sensor in the deploy's order and timestamps each reading.

    samples[stream]  deque of (t, values), consumed by the frame loop
    events           deque of (t, kind, text)
    busy             (device, t_start) while an I2C call is in progress
    """

    def __init__(self, stall_s, reconnect):
        super().__init__(name="dropout-reader", daemon=True)
        self.samples = {s: deque() for s in STREAMS}
        self.events = deque()
        self.busy = None
        self._stall_s = stall_s
        self._reconnect = reconnect
        self._stop = threading.Event()

        self.imus = []
        for prefix, addr in (("foot", FOOT_ADDR), ("shank", SHANK_ADDR)):
            print(f"[monitor] connecting {prefix} IMU (bus {I2C_BUS}, 0x{addr:02X}) "
                  f"at {IMU_HZ:g} Hz...", flush=True)
            self.imus.append((prefix, LeanBNO085(I2C_BUS, addr, IMU_HZ, (ACCEL, GYRO),
                                                 label=f"IMU-{prefix}")))
        print(f"[monitor] opening bus {I2C_BUS} for the encoder and FSR ADC...", flush=True)
        self.bus = smbus2.SMBus(I2C_BUS)
        self.fsr = TimedADS1115(self.bus, FSR_HZ)
        self._last_report = {}

    def stop(self):
        self._stop.set()

    def _event(self, kind, text):
        self.events.append((time.perf_counter(), kind, text))

    def _call(self, device, fn, *args):
        """Run one I2C call, marking it busy so a hang is visible from outside."""
        t0 = time.perf_counter()
        self.busy = (device, t0)
        try:
            return fn(*args)
        finally:
            self.busy = None
            blocked = time.perf_counter() - t0
            if blocked >= self._stall_s:
                self._event("STALL", f"{device} I2C call blocked {blocked * 1000:.0f} ms")

    def _poll_imu(self, prefix, imu):
        n_acc, n_gyro = imu.counts.get(ACCEL, 0), imu.counts.get(GYRO, 0)
        seq_gaps = imu.seq_gaps
        try:
            n = self._call(f"{prefix} IMU", imu.poll)
        except OSError as exc:
            self._event("I2CERR", f"{prefix} IMU: {exc}")
            return 0
        except (KeyError, RuntimeError, IndexError, struct.error) as exc:
            self._event("BADPKT", f"{prefix} IMU: {type(exc).__name__}({exc})")
            return 0
        now = time.perf_counter()
        if imu.seq_gaps != seq_gaps:
            self._event("SEQ", f"{prefix} IMU: {imu.seq_gaps - seq_gaps} packet(s) missing")
        if imu.counts.get(ACCEL, 0) != n_acc:
            self.samples[f"{prefix}_acc"].append((now, imu.accel))
        if imu.counts.get(GYRO, 0) != n_gyro:
            self.samples[f"{prefix}_gyro"].append((now, imu.gyro))
        if n:
            self._last_report[prefix] = now
        return n

    def _reconnect_if_silent(self, prefix, imu):
        now = time.perf_counter()
        silent = now - self._last_report.setdefault(prefix, now)
        if not self._reconnect or silent < SILENT_S:
            return
        self._event("RECON", f"{prefix} IMU silent for {silent:.1f} s; re-initialising "
                             f"(all sensors pause while it reconnects)")
        try:
            imu.reconnect()
            self._event("RECON", f"{prefix} IMU reconnected")
        except RuntimeError as exc:
            self._event("RECON", f"{prefix} IMU: {exc}")
        self._last_report[prefix] = time.perf_counter()

    def _step_fsr(self):
        try:
            pair = self._call("FSR ADC", self.fsr.step, time.perf_counter())
        except OSError as exc:
            self._event("I2CERR", f"FSR ADC: {exc}")
            return
        if pair is not None:
            self.samples["fsr"].append((time.perf_counter(), pair))

    def _read_encoder(self):
        try:
            d = self._call("encoder", self.bus.read_i2c_block_data,
                           AS5600_ADDR, AS5600_REG_ANGLE, 2)
        except OSError as exc:
            self._event("I2CERR", f"encoder: {exc}")
            return
        if d[0] & 0xF0:
            self._event("BADENC", f"encoder returned 0x{d[0]:02X}{d[1]:02X}; "
                                  f"the top 4 bits are always 0")
            return
        raw = ((d[0] & 0x0F) << 8) | d[1]
        self.samples["encoder"].append((time.perf_counter(), (raw * 360.0 / 4096.0,)))

    def run(self):
        try:
            while not self._stop.is_set():
                self._step_fsr()
                for prefix, imu in self.imus:
                    for _ in range(MAX_POLLS):
                        n = self._poll_imu(prefix, imu)
                        self._step_fsr()
                        if not n:
                            break
                    self._reconnect_if_silent(prefix, imu)
                self._read_encoder()
        except Exception as exc:                            # noqa: BLE001
            self._event("CRASH", f"reader thread stopped: {exc!r}")

    def close(self):
        for _, imu in self.imus:
            try:
                imu.close()
            except Exception:
                pass
        try:
            self.bus.close()
        except Exception:
            pass


# ── Helpers ────────────────────────────────────────────────────────────────────
def other_bus_user():
    """Command line of another process that reads bus 1, or None."""
    me = os.getpid()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == me:
            continue
        try:
            cmd = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="ignore")
        except OSError:
            continue                    # the process ended while being read
        if "python" in cmd and any(name in cmd for name in BUS_USERS):
            return cmd.strip()
    return None


def header_line():
    parts = []
    for s in STREAMS:
        width = CELL * len(CSV_COLS[s])
        if s == "fsr":
            parts.append(f"{'toe':>{CELL}}{'heel':>{CELL}}")
        else:
            parts.append(f"{LABELS[s]:>{width}}")
    return f"{'t [s]':>8} |" + " |".join(parts) + " | missing"


def cells(stream, values):
    if values is None:
        return "".join(paint(f"{'NaN':>{CELL}}", RED) for _ in CSV_COLS[stream])
    if stream == "fsr":
        return "".join(f"{v:>{CELL}d}" for v in values)
    return "".join(f"{v:>{CELL}.2f}" for v in values)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only-misses", action="store_true",
                    help="print only frames where a sensor gave nothing new")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="seconds to run (default: until Ctrl+C)")
    ap.add_argument("--gap-ms", type=float, default=30.0,
                    help="report a GAP when a stream is silent this long")
    ap.add_argument("--stall-ms", type=float, default=20.0,
                    help="report an I2C call that blocks this long")
    ap.add_argument("--no-reconnect", action="store_true",
                    help="do not re-initialise an IMU that goes silent")
    ap.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "logs"))
    args = ap.parse_args()

    busy_cmd = other_bus_user()
    if busy_cmd:
        print(f"Another process is using I2C bus {I2C_BUS}; stop it first:\n  {busy_cmd}")
        sys.exit(1)

    gap_frames = max(1, math.ceil(args.gap_ms / 1000.0 / FRAME_S))
    stall_s = args.stall_ms / 1000.0

    try:
        reader = Reader(stall_s, reconnect=not args.no_reconnect)
    except KeyboardInterrupt:
        print("\nAborted during init.")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"dropout_{stamp}.csv"
    log_path = out_dir / f"dropout_{stamp}_events.log"
    csv_file = open(csv_path, "w", newline="")
    log_file = open(log_path, "w")
    writer = csv.writer(csv_file)
    writer.writerow(["t_s"] + [c for s in STREAMS for c in CSV_COLS[s]]
                    + [f"n_{s}" for s in STREAMS])

    reader.start()

    # Wait for every stream's first reading so start-up is not counted as a gap.
    t_wait = time.perf_counter()
    while (not all(reader.samples[s] for s in STREAMS)
           and time.perf_counter() - t_wait < STARTUP_S):
        time.sleep(0.01)
    for s in STREAMS:
        if not reader.samples[s]:
            print(paint(f"[monitor] {s}: no reading after {STARTUP_S:.0f} s", RED))
        reader.samples[s].clear()

    t0 = time.perf_counter()
    counts = {}
    readings = {s: 0 for s in STREAMS}
    missed = {s: 0 for s in STREAMS}
    sec_missed = {s: 0 for s in STREAMS}
    miss_run = {s: 0 for s in STREAMS}
    longest = {s: 0 for s in STREAMS}
    announced_hang = None

    def log_event(t, kind, text):
        counts[kind] = counts.get(kind, 0) + 1
        line = f"[{t - t0:8.3f}s] {kind:<6} {text}"
        log_file.write(line + "\n")
        print(paint(line, EVENT_COLOR.get(kind, BOLD)), flush=True)

    print(f"\n[monitor] live. {FRAME_S * 1000:.0f} ms frames, GAP >= {gap_frames * FRAME_S * 1000:.0f} ms, "
          f"STALL >= {args.stall_ms:.0f} ms. Ctrl+C to stop.\n")
    print(header_line())

    frame = 0
    try:
        while True:
            frame_end = t0 + (frame + 1) * FRAME_S
            if args.duration and frame_end - t0 > args.duration:
                break
            wait = frame_end + FRAME_GRACE_S - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
            now = time.perf_counter()

            while reader.events:
                log_event(*reader.events.popleft())
            busy = reader.busy
            if busy and busy[1] != announced_hang and now - busy[1] >= stall_s:
                announced_hang = busy[1]
                log_event(now, "HUNG", f"{busy[0]} I2C call still blocked after "
                                       f"{(now - busy[1]) * 1000:.0f} ms")
            if not reader.is_alive():
                break

            row, n_row, missing = {}, {}, []
            for s in STREAMS:
                q, values, n = reader.samples[s], None, 0
                while q and q[0][0] <= frame_end:
                    values = q.popleft()[1]
                    n += 1
                row[s], n_row[s] = values, n
                readings[s] += n
                if values is None:
                    missing.append(s)
                    missed[s] += 1
                    sec_missed[s] += 1
                    miss_run[s] += 1
                    continue
                if miss_run[s] >= gap_frames:
                    start = frame_end - FRAME_S * (miss_run[s] + 1) - t0
                    log_event(now, "GAP", f"{s}: no new reading for {miss_run[s] * FRAME_S * 1000:.0f} ms "
                                          f"({start:.2f}-{start + miss_run[s] * FRAME_S:.2f} s)")
                longest[s] = max(longest[s], miss_run[s])
                miss_run[s] = 0

            t_frame = frame_end - t0
            writer.writerow([f"{t_frame:.3f}"]
                            + [f"{v:.5f}" if isinstance(v, float) else v
                               for s in STREAMS
                               for v in (row[s] if row[s] is not None
                                         else [math.nan] * len(CSV_COLS[s]))]
                            + [n_row[s] for s in STREAMS])
            if missing or not args.only_misses:
                print(f"{t_frame:8.2f} |" + " |".join(cells(s, row[s]) for s in STREAMS)
                      + " | " + paint(" ".join(missing), RED))

            frame += 1
            if frame % round(1.0 / FRAME_S) == 0:
                print(paint(f"── {t_frame:.0f} s ── frames missed in the last 1 s: "
                            + "  ".join(f"{s} {sec_missed[s]}" for s in STREAMS)
                            + " | events so far: "
                            + ("  ".join(f"{k} {v}" for k, v in sorted(counts.items())) or "none"),
                            BOLD))
                sec_missed = {s: 0 for s in STREAMS}
            if not args.only_misses and frame % round(5.0 / FRAME_S) == 0:
                print(header_line())
    except KeyboardInterrupt:
        pass
    finally:
        reader.stop()
        reader.join(timeout=1.0)
        while reader.events:
            log_event(*reader.events.popleft())
        for s in STREAMS:
            longest[s] = max(longest[s], miss_run[s])
        if reader.is_alive():
            print(paint("[monitor] reader thread still blocked in an I2C call at exit", RED))
        else:
            reader.close()
        csv_file.close()

        elapsed = frame * FRAME_S
        lines = [f"\nSummary over {elapsed:.1f} s ({frame} frames of {FRAME_S * 1000:.0f} ms)",
                 f"  {'stream':<11}{'readings':>9}{'frames missed':>17}{'longest gap':>13}"]
        for s in STREAMS:
            pct = 100.0 * missed[s] / frame if frame else 0.0
            lines.append(f"  {s:<11}{readings[s]:>9}{missed[s]:>9} ({pct:4.1f}%)"
                         f"{longest[s] * FRAME_S * 1000:>10.0f} ms")
        lines.append("  events: " + ("  ".join(f"{k} {v}" for k, v in sorted(counts.items()))
                                     or "none"))
        for prefix, imu in reader.imus:
            lines.append(f"  {prefix} IMU: {imu.packets} packets, {imu.seq_gaps} missing by "
                         f"sequence number, {imu.continuations} continuation reads")
        lines += [f"  frames: {csv_path}", f"  events: {log_path}"]
        summary = "\n".join(lines)
        print(summary)
        log_file.write(summary + "\n")
        log_file.close()


if __name__ == "__main__":
    main()
