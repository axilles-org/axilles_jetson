#!/usr/bin/env python3
"""
true_dropout_monitor.py — report only TRUE dropouts, not frame-alignment artifacts.

sensor_dropout_monitor.py bins readings into 10 ms frames. A 100 Hz stream then
shows a NaN frame whenever one reading slips into the next frame (that frame
gets two) — the IMU clock drifting against the Jetson clock. No data is lost.

This script never bins. It timestamps every reading and measures the real
interval between consecutive readings of each stream, and it watches every
I2C call. A TRUE dropout is any of:

    GAP       interval between two readings > --gap-factor x expected spacing
              (default 3x, as in the test plan: 30 ms at 100 Hz, 15 ms at 200 Hz)
    SILENT    a stream is silent right now for longer than its GAP threshold
    LOSS      BNO085 packets missing by sequence number (sent but never read)
    I2CERR    an I2C call raised OSError  (errno 121 = no ACK, 110 = timed out)
    BADPKT    a BNO085 packet that could not be parsed
    REBOOT    a BNO085 sent its boot messages (it reset by itself)
    RESET     reset cause read when a failed IMU answered again, BEFORE re-init:
                1 power-on / brown-out   2 internal   3 watchdog   4 RST pin   5 other
    STALL     an I2C call blocked >= --stall-ms
    BADENC    AS5600 returned bits that are always 0 (corrupt read)
    PAUSE     the loop was paused re-initialising an IMU (gaps in OTHER
              streams during a pause are labelled as caused by it)

Each GAP is labelled with what happened inside it (I2CERR, RESET, STALL,
PAUSE ...) or "no error seen" if the sensor simply went quiet.

Reads the same sensors, in the same order and at the same rates as
sensor_dropout_monitor.py (imported, unchanged). A failed IMU is probed without
blocking the other sensors; it is re-initialised only once it answers again
(use --no-reconnect to leave it down).

USAGE
    python3 true_dropout_monitor.py --duration 120                       # all sensors
    python3 true_dropout_monitor.py --duration 120 --sensors foot        # one IMU
    python3 true_dropout_monitor.py --duration 120 --sensors foot,shank --no-reconnect

Writes logs/true_<sensors>_<time>_events.csv and a summary in ..._summary.txt.
Never opens CAN, never talks to the motor.
"""

import argparse
import csv
import errno as errno_mod
import os
import struct
import sys
import threading
import time
from collections import deque, defaultdict
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import sensor_dropout_monitor as sdm                      # noqa: E402
import bno085_lean                                         # noqa: E402  (path set by sdm)
from bno085_lean import ACCEL, GYRO, LeanBNO085            # noqa: E402

sdm.BUS_USERS = sdm.BUS_USERS + ("true_dropout_monitor.py", "imu_only_dropout_monitor.py",
                                 "imu_diag.py")

SENSORS = ("foot", "shank", "encoder", "fsr")
IMU_ADDRS = {"foot": sdm.FOOT_ADDR, "shank": sdm.SHANK_ADDR}
STREAMS_OF = {"foot": ("foot_acc", "foot_gyro"), "shank": ("shank_acc", "shank_gyro"),
              "encoder": ("encoder",), "fsr": ("fsr",)}
ENCODER_REQ_HZ = 200.0            # plan requirement, used as the encoder's expected spacing
CAUSES = {0: "n/a", 1: "power-on/brown-out", 2: "internal", 3: "watchdog",
          4: "external RST pin", 5: "other"}
PROBE_S = 0.005                   # how often a failed IMU is probed

COLOR = sys.stdout.isatty()
RED, YELLOW, GREEN, BOLD = "31", "33", "32", "1"
EVENT_COLOR = {"GAP": RED, "SILENT": RED, "LOSS": RED, "I2CERR": RED, "BADPKT": RED,
               "REBOOT": RED, "RESET": RED, "STALL": RED, "BADENC": RED, "CRASH": RED,
               "PAUSE": YELLOW, "BACK": YELLOW, "REINIT": YELLOW}


def paint(text, code):
    return f"\033[{code}m{text}\033[0m" if COLOR else text


# ── BNO085 reset cause (SH-2 product-ID response, byte 1) ──────────────────────
def _read_packet(fd, timeout_s):
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
    """(cause, packets waiting before the reply). Does not reconfigure the IMU."""
    seen = []
    t_end = time.monotonic() + 0.4
    while time.monotonic() < t_end:
        p = _read_packet(fd, 0.03)
        if p is None:
            break
        seen.append(f"ch{p[0]}:{len(p[1])}B")
    os.write(fd, bytes([6, 0, 2, 0, 0xF9, 0x00]))
    t_end = time.monotonic() + 1.0
    while time.monotonic() < t_end:
        p = _read_packet(fd, 0.2)
        if p and p[0] == 2 and p[1][:1] == b"\xf8":
            return p[1][1], seen
        if p:
            seen.append(f"ch{p[0]}:{len(p[1])}B")
    return None, seen


# ── reader ─────────────────────────────────────────────────────────────────────
class Reader(threading.Thread):
    def __init__(self, sensors, stall_s, reconnect):
        super().__init__(name="true-dropout-reader", daemon=True)
        self.sensors = sensors
        self.stall_s = stall_s
        self.reconnect = reconnect
        self.stop_evt = threading.Event()      # not _stop: shadows Thread._stop()
        self.events = deque()                  # (t, kind, stream_or_device, text)
        self.last_t = {}                       # stream -> time of newest reading
        self.readings = defaultdict(int)
        self.intervals = defaultdict(list)     # stream -> [(t_prev, t_now)]
        self.busy = None
        self.paused = []                       # [(t0, t1)] loop paused for re-init
        self.read_fsr = "fsr" in sensors
        self.read_enc = "encoder" in sensors

        self._last_raw = {}
        orig_read = os.read

        def tracking_read(fd, n):              # keep the last bytes per IMU fd
            data = orig_read(fd, n)
            self._last_raw[fd] = data
            return data
        bno085_lean.os.read = tracking_read

        self.imus = {}
        for p in ("foot", "shank"):
            if p in sensors:
                print(f"[true] connecting {p} IMU (bus {sdm.I2C_BUS}, 0x{IMU_ADDRS[p]:02X}) "
                      f"at {sdm.IMU_HZ:g} Hz...", flush=True)
                self.imus[p] = dict(imu=LeanBNO085(sdm.I2C_BUS, IMU_ADDRS[p], sdm.IMU_HZ,
                                                   (ACCEL, GYRO), label=f"IMU-{p}"),
                                    up=True, down_t=0.0, next_probe=0.0, seq=0)
        self.bus = None
        if self.read_fsr or self.read_enc:
            import smbus2
            self.bus = smbus2.SMBus(sdm.I2C_BUS)
            self.fsr = sdm.TimedADS1115(self.bus, sdm.FSR_HZ)

    def event(self, kind, who, text, t=None):
        self.events.append((time.perf_counter() if t is None else t, kind, who, text))

    def _call(self, who, fn, *args):
        t0 = time.perf_counter()
        self.busy = (who, t0)
        try:
            return fn(*args)
        finally:
            self.busy = None
            dt = time.perf_counter() - t0
            if dt >= self.stall_s:
                self.event("STALL", who, f"I2C call blocked {dt * 1e3:.0f} ms", t0)

    def _reading(self, stream, t):
        prev = self.last_t.get(stream)
        if prev is not None:
            self.intervals[stream].append((prev, t))
        self.last_t[stream] = t
        self.readings[stream] += 1

    def _oserror(self, who, exc):
        code = getattr(exc, "errno", None)
        name = errno_mod.errorcode.get(code, "?") if code else "?"
        hint = {121: "no ACK: device not answering", 110: "timed out: bus stuck"}.get(code, "")
        self.event("I2CERR", who, f"errno {code} {name} {hint} ({exc})".strip())

    # IMU ---------------------------------------------------------------------
    def _poll_imu(self, p):
        s = self.imus[p]
        imu = s["imu"]
        if not s["up"]:
            return self._probe_imu(p)
        n_acc, n_gyro, seq = imu.counts.get(ACCEL, 0), imu.counts.get(GYRO, 0), imu.seq_gaps
        try:
            n = self._call(f"{p} IMU", imu.poll)
        except OSError as exc:
            self._oserror(f"{p} IMU", exc)
            s["up"], s["down_t"], s["next_probe"] = False, time.perf_counter(), 0.0
            return 0
        except (KeyError, RuntimeError, IndexError, struct.error) as exc:
            raw = self._last_raw.get(imu._fd, b"")
            ch = raw[2] if len(raw) > 2 else None
            if ch in (0, 1):
                self.event("REBOOT", f"{p} IMU", f"boot message on SHTP channel {ch} "
                           f"({len(raw)} B): the IMU reset by itself")
            else:
                self.event("BADPKT", f"{p} IMU", f"{type(exc).__name__}({exc}) raw={raw[:20].hex()}")
            return 0
        t = time.perf_counter()
        if imu.seq_gaps != seq:
            self.event("LOSS", f"{p} IMU", f"{imu.seq_gaps - seq} packet(s) missing by sequence number")
        if imu.counts.get(ACCEL, 0) != n_acc:
            self._reading(f"{p}_acc", t)
        if imu.counts.get(GYRO, 0) != n_gyro:
            self._reading(f"{p}_gyro", t)
        return n

    def _probe_imu(self, p):
        s = self.imus[p]
        now = time.perf_counter()
        if now < s["next_probe"]:
            return 0
        s["next_probe"] = now + PROBE_S
        try:
            os.read(s["imu"]._fd, 4)
        except OSError:
            return 0
        gone = (now - s["down_t"]) * 1e3
        try:
            cause, seen = reset_cause(s["imu"]._fd)
        except OSError as exc:
            # answered once, then stopped again (still booting or hit again):
            # stay down and keep probing instead of killing the reader thread
            self.event("BACK", f"{p} IMU", f"answered after {gone:.0f} ms, then failed again "
                       f"while reading the reset cause ({exc}); still probing")
            s["next_probe"] = time.perf_counter() + PROBE_S
            return 0
        booted = any(x.startswith(("ch0:", "ch1:")) for x in seen)
        self.event("BACK", f"{p} IMU", f"answering again after {gone:.0f} ms")
        self.event("RESET", f"{p} IMU", f"reset cause {cause} ({CAUSES.get(cause, '?')}); "
                   f"rebooted={'yes' if booted else 'no'}; waiting packets: {' '.join(seen) or 'none'}")
        if not self.reconnect:
            s["next_probe"] = float("inf")         # leave it down
            self.event("REINIT", f"{p} IMU", "--no-reconnect: left down")
            return 0
        t0 = time.perf_counter()
        try:
            s["imu"].reconnect()
            s["up"] = True
            t1 = time.perf_counter()
            self.paused.append((t0, t1))
            self.event("PAUSE", "loop", f"re-initialising {p} IMU paused all reads {(t1 - t0) * 1e3:.0f} ms", t0)
        except RuntimeError as exc:
            self.event("REINIT", f"{p} IMU", f"failed: {exc}")
            s["down_t"] = time.perf_counter()
        return 0

    # encoder / FSR ------------------------------------------------------------
    def _step_fsr(self):
        try:
            pair = self._call("FSR ADC", self.fsr.step, time.perf_counter())
        except OSError as exc:
            self._oserror("FSR ADC", exc)
            return
        if pair is not None:
            self._reading("fsr", time.perf_counter())

    def _read_encoder(self):
        try:
            d = self._call("encoder", self.bus.read_i2c_block_data,
                           sdm.AS5600_ADDR, sdm.AS5600_REG_ANGLE, 2)
        except OSError as exc:
            self._oserror("encoder", exc)
            return
        if d[0] & 0xF0:
            self.event("BADENC", "encoder", f"returned 0x{d[0]:02X}{d[1]:02X}")
            return
        self._reading("encoder", time.perf_counter())

    def run(self):
        try:
            while not self.stop_evt.is_set():
                if self.read_fsr:
                    self._step_fsr()
                for p in self.imus:
                    for _ in range(sdm.MAX_POLLS):
                        n = self._poll_imu(p)
                        if self.read_fsr:
                            self._step_fsr()
                        if not n:
                            break
                if self.read_enc:
                    self._read_encoder()
        except Exception as exc:                            # noqa: BLE001
            self.event("CRASH", "reader", f"reader thread stopped: {exc!r}")

    def close(self):
        for s in self.imus.values():
            try:
                s["imu"].close()
            except Exception:
                pass
        if self.bus is not None:
            try:
                self.bus.close()
            except Exception:
                pass


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sensors", default="foot,shank,encoder,fsr",
                    help="comma list of foot, shank, encoder, fsr (default: all)")
    ap.add_argument("--duration", type=float, default=0.0, help="seconds (default: until Ctrl+C)")
    ap.add_argument("--gap-factor", type=float, default=3.0,
                    help="GAP = interval > this x expected spacing (default 3, as in the plan)")
    ap.add_argument("--stall-ms", type=float, default=20.0)
    ap.add_argument("--no-reconnect", action="store_true",
                    help="leave a failed IMU down instead of re-initialising it")
    ap.add_argument("--status-s", type=float, default=5.0, help="status line period")
    ap.add_argument("--out-dir", default=str(HERE / "logs"))
    args = ap.parse_args()

    sensors = [p.strip() for p in args.sensors.split(",") if p.strip()]
    if not sensors or any(p not in SENSORS for p in sensors):
        ap.error(f"--sensors takes any of {', '.join(SENSORS)}; got {args.sensors!r}")
    sensors = [p for p in SENSORS if p in sensors]
    streams = [s for p in sensors for s in STREAMS_OF[p]]
    spacing = {s: (1.0 / ENCODER_REQ_HZ if s == "encoder" else
                   1.0 / sdm.FSR_HZ if s == "fsr" else 1.0 / sdm.IMU_HZ) for s in streams}
    thresh = {s: args.gap_factor * spacing[s] for s in streams}

    busy_cmd = sdm.other_bus_user()
    if busy_cmd:
        print(f"Another process is using I2C bus {sdm.I2C_BUS}; stop it first:\n  {busy_cmd}")
        sys.exit(1)

    reader = Reader(sensors, args.stall_ms / 1000.0, reconnect=not args.no_reconnect)
    tag = "-".join(sensors)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ev_path = out / f"true_{tag}_{stamp}_events.csv"
    sum_path = out / f"true_{tag}_{stamp}_summary.txt"
    ev_file = open(ev_path, "w", newline="")
    ev_csv = csv.writer(ev_file)
    ev_csv.writerow(["t_s", "kind", "who", "detail"])

    reader.start()
    t_wait = time.perf_counter()               # first reading of every stream = start
    while not all(s in reader.last_t for s in streams) and time.perf_counter() - t_wait < 3.0:
        time.sleep(0.01)
    for s in streams:
        if s not in reader.last_t:
            print(paint(f"[true] {s}: no reading after 3 s", RED))
    t0 = time.perf_counter()
    for s in streams:                          # count only from t0
        reader.intervals[s].clear()
        reader.readings[s] = 0
        reader.last_t.setdefault(s, t0)

    counts = defaultdict(int)
    gap_log = defaultdict(list)                # stream -> [(start, end, label)]
    all_events = []
    silent_now = set()
    done = {s: 0 for s in streams}             # intervals already checked

    def emit(t, kind, who, text):
        counts[kind] += 1
        all_events.append((t, kind, who, text))
        ev_csv.writerow([f"{t - t0:.4f}", kind, who, text])
        ev_file.flush()
        print(paint(f"[{t - t0:8.3f}s] {kind:<6} {who:<10} {text}", EVENT_COLOR.get(kind, BOLD)),
              flush=True)

    def label_gap(a, b, stream):
        dev = stream.split("_")[0]
        why = sorted({k for (t, k, who, _) in all_events
                      if a - 0.05 <= t <= b + 0.01 and k not in ("GAP", "SILENT")
                      and (k == "PAUSE" or who.startswith(dev) or k == "STALL")})
        if any(p0 <= b and p1 >= a for p0, p1 in reader.paused) and "PAUSE" not in why:
            why.append("PAUSE")
        return ", ".join(why) or "no error seen (sensor went quiet)"

    print(f"\n[true] live: {', '.join(streams)}.  GAP thresholds: "
          + ", ".join(f"{s} {thresh[s] * 1e3:.0f} ms" for s in streams)
          + ".  Only real problems are printed. Ctrl+C to stop.\n")
    next_status = t0 + args.status_s
    try:
        while True:
            now = time.perf_counter()
            if args.duration and now - t0 >= args.duration:
                break
            while reader.events:
                emit(*reader.events.popleft())
            # completed gaps
            for s in streams:
                iv = reader.intervals[s]
                while done[s] < len(iv):
                    a, b = iv[done[s]]
                    done[s] += 1
                    if b - a > thresh[s]:
                        lab = label_gap(a, b, s)
                        gap_log[s].append((a, b, lab))
                        emit(b, "GAP", s, f"no reading for {(b - a) * 1e3:.0f} ms "
                             f"({a - t0:.3f}-{b - t0:.3f} s)  cause: {lab}")
                        silent_now.discard(s)
            # ongoing silence
            for s in streams:
                quiet = now - reader.last_t[s]
                if quiet > max(thresh[s], 0.1) and s not in silent_now:
                    silent_now.add(s)
                    emit(now, "SILENT", s, f"no reading for {quiet * 1e3:.0f} ms so far")
            busy = reader.busy
            if busy and now - busy[1] >= 0.2 and counts.get(("hung", busy[1])) is None:
                counts[("hung", busy[1])] = 1
                emit(now, "STALL", busy[0], f"I2C call still blocked after {(now - busy[1]) * 1e3:.0f} ms")
            if now >= next_status:
                el = now - t0
                print(paint(f"── {el:5.0f} s ── " + "  ".join(
                    f"{s} {reader.readings[s] / el:.0f}Hz" for s in streams)
                    + " │ true dropouts: " + (", ".join(
                        f"{k} {v}" for k, v in sorted(counts.items()) if isinstance(k, str)
                        and k not in ("BACK", "REINIT")) or "none"), BOLD), flush=True)
                next_status += args.status_s
            if not reader.is_alive():
                break
            time.sleep(0.02)
    except KeyboardInterrupt:
        pass
    finally:
        reader.stop_evt.set()
        reader.join(timeout=2.0)
        while reader.events:
            emit(*reader.events.popleft())
        elapsed = time.perf_counter() - t0
        for s in streams:                      # gaps completed during shutdown
            iv = reader.intervals[s]
            for a, b in iv[done[s]:]:
                if b - a > thresh[s]:
                    gap_log[s].append((a, b, label_gap(a, b, s)))
        if not reader.is_alive():
            reader.close()
        ev_file.close()

        L = [f"true_dropout_monitor  {datetime.now():%Y-%m-%d %H:%M:%S}  sensors: {', '.join(sensors)}  "
             f"(not addressed: {', '.join(p for p in SENSORS if p not in sensors) or 'none'})",
             f"duration {elapsed:.1f} s, reconnect {'off' if args.no_reconnect else 'on'}, "
             f"GAP = interval > {args.gap_factor:g} x expected spacing", "",
             f"  {'stream':<11}{'readings':>9}{'rate Hz':>9}{'expected':>10}{'max gap ms':>12}"
             f"{'GAPs':>6}{'time lost s':>13}"]
        for s in streams:
            iv = reader.intervals[s]
            mx = max((b - a for a, b in iv), default=0.0) * 1e3
            lost = sum(b - a for a, b, _ in gap_log[s])
            L.append(f"  {s:<11}{reader.readings[s]:>9}{reader.readings[s] / max(elapsed, 1e-9):>9.1f}"
                     f"{1 / spacing[s]:>10.0f}{mx:>12.1f}{len(gap_log[s]):>6}{lost:>13.3f}")
        for p, st in reader.imus.items():
            L.append(f"  {p} IMU: {st['imu'].packets} packets, {st['imu'].seq_gaps} lost by sequence "
                     f"number, {st['imu'].continuations} continuation reads")
        kinds = {k: v for k, v in counts.items() if isinstance(k, str)}
        L.append("  events: " + ("  ".join(f"{k} {v}" for k, v in sorted(kinds.items())) or "none"))
        resets = [e for e in all_events if e[1] == "RESET"]
        for t, _, who, text in resets:
            L.append(f"  {who} at {t - t0:.2f} s: {text}")
        true_kinds = {"GAP", "SILENT", "LOSS", "I2CERR", "BADPKT", "REBOOT", "RESET", "BADENC", "CRASH"}
        n_true = sum(v for k, v in kinds.items() if k in true_kinds)
        if n_true == 0:
            L.append(paint("  VERDICT: no true dropouts. Every stream delivered within "
                           f"{args.gap_factor:g}x its spacing, no I2C errors, no lost packets.", GREEN))
        else:
            L.append(paint(f"  VERDICT: TRUE DROPOUTS FOUND ({n_true} events). See GAP causes above "
                           f"and {ev_path.name}.", RED))
        L += [f"  events: {ev_path}", f"  summary: {sum_path}"]
        text = "\n".join(L)
        print("\n" + text)
        import re
        sum_path.write_text(re.sub(r"\033\[[0-9;]*m", "", text) + "\n")


if __name__ == "__main__":
    main()
