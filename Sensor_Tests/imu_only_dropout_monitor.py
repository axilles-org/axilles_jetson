#!/usr/bin/env python3
"""
imu_only_dropout_monitor.py — dropout isolation: read only the sensors you choose.

Same as sensor_dropout_monitor.py (same drivers, rates, read order, 10 ms frames,
events and summary — the reader and helpers are imported from it, unchanged),
except that sensors not listed in --sensors are NEVER addressed. They stay
wired to the bus but put no traffic on it. Default: both IMUs only.

    dropouts stop      -> traffic / timing from the unselected sensors is involved
    dropouts continue  -> not traffic; next step is unplugging those sensors
                          (electrical loading / power), then the IMU wiring

Run each case with the same motion (same taps or same walk):
    python3 sensor_dropout_monitor.py   --duration 120 --only-misses                    # baseline, all
    python3 imu_only_dropout_monitor.py --duration 120 --only-misses                    # both IMUs
    python3 imu_only_dropout_monitor.py --duration 120 --only-misses --sensors foot     # foot IMU alone
    python3 imu_only_dropout_monitor.py --duration 120 --only-misses --sensors shank    # shank IMU alone
    python3 imu_only_dropout_monitor.py --duration 120 --only-misses --sensors foot,encoder

Options
    --sensors LIST      comma list of foot, shank, encoder, fsr (default foot,shank)
                        --imus is an alias kept for the earlier commands
    --no-reconnect      never re-initialise a silent IMU. A re-init pauses the
                        whole loop ~1-3 s, which can make the OTHER sensors show
                        errors, so this keeps each sensor's events its own.
    --idle-us N         sleep N us per pass to stand in for the time the skipped
                        reads used to take, without touching the bus
                        (0 = loop as fast as possible, polling the rest more often)

Writes logs/sel_<sensors>_<time>.csv and logs/sel_<sensors>_<time>_events.log.
Never opens CAN, never talks to the motor.
"""

import argparse
import csv
import math
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import sensor_dropout_monitor as sdm                      # noqa: E402
from bno085_lean import ACCEL, GYRO, LeanBNO085            # noqa: E402  (path set by sdm)

# also refuse to run next to another copy of this script (affects this process only)
sdm.BUS_USERS = sdm.BUS_USERS + ("imu_only_dropout_monitor.py",)

IMU_ADDRS = {"foot": sdm.FOOT_ADDR, "shank": sdm.SHANK_ADDR}
SENSORS = ("foot", "shank", "encoder", "fsr")     # sensor -> streams: foot_acc/foot_gyro, ...


class SelectedReader(sdm.Reader):
    """sdm.Reader that only addresses the selected sensors, in the same order
    as the original (FSR step, then each IMU with an FSR step after every poll,
    then the encoder)."""

    def __init__(self, stall_s, reconnect, sensors, idle_s):
        threading.Thread.__init__(self, name="selected-reader", daemon=True)
        self.samples = {s: deque() for s in sdm.STREAMS}
        self.events = deque()
        self.busy = None
        self._stall_s = stall_s
        self._reconnect = reconnect
        # not self._stop: that name shadows threading.Thread._stop(), which
        # join() calls once the thread has finished (TypeError at exit)
        self._stop_evt = threading.Event()
        self._idle_s = idle_s
        self._last_report = {}
        self.read_fsr = "fsr" in sensors
        self.read_enc = "encoder" in sensors
        self.imus = []
        for prefix in (p for p in ("foot", "shank") if p in sensors):
            addr = IMU_ADDRS[prefix]
            print(f"[select] connecting {prefix} IMU (bus {sdm.I2C_BUS}, 0x{addr:02X}) "
                  f"at {sdm.IMU_HZ:g} Hz...", flush=True)
            self.imus.append((prefix, LeanBNO085(sdm.I2C_BUS, addr, sdm.IMU_HZ, (ACCEL, GYRO),
                                                 label=f"IMU-{prefix}")))
        self.bus = None
        if self.read_fsr or self.read_enc:
            print(f"[select] opening bus {sdm.I2C_BUS} for the "
                  + " and ".join(n for n, on in (("encoder", self.read_enc),
                                                 ("FSR ADC", self.read_fsr)) if on) + "...",
                  flush=True)
            import smbus2
            self.bus = smbus2.SMBus(sdm.I2C_BUS)
            self.fsr = sdm.TimedADS1115(self.bus, sdm.FSR_HZ)
        skipped = [f"{n} 0x{a:02X}" for n, a, on in (
            ("foot IMU", sdm.FOOT_ADDR, "foot" in sensors),
            ("shank IMU", sdm.SHANK_ADDR, "shank" in sensors),
            ("encoder", sdm.AS5600_ADDR, self.read_enc),
            ("FSR ADC", sdm.TimedADS1115.ADDR, self.read_fsr)) if not on]
        print(f"[select] not addressed: {', '.join(skipped) or 'nothing'}", flush=True)

    def run(self):
        try:
            while not self._stop_evt.is_set():
                if self.read_fsr:
                    self._step_fsr()
                for prefix, imu in self.imus:
                    for _ in range(sdm.MAX_POLLS):
                        n = self._poll_imu(prefix, imu)
                        if self.read_fsr:
                            self._step_fsr()
                        if not n:
                            break
                    self._reconnect_if_silent(prefix, imu)
                if self.read_enc:
                    self._read_encoder()
                if self._idle_s:
                    time.sleep(self._idle_s)
        except Exception as exc:                            # noqa: BLE001
            self._event("CRASH", f"reader thread stopped: {exc!r}")

    def stop(self):
        self._stop_evt.set()

    def close(self):
        for _, imu in self.imus:
            try:
                imu.close()
            except Exception:
                pass
        if self.bus is not None:
            try:
                self.bus.close()
            except Exception:
                pass


def header_line(streams):
    parts = [f"{sdm.LABELS[s]:>{sdm.CELL * len(sdm.CSV_COLS[s])}}" for s in streams]
    return f"{'t [s]':>8} |" + " |".join(parts) + " | missing"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sensors", "--imus", dest="sensors", default="foot,shank",
                    help="comma list of foot, shank, encoder, fsr (default foot,shank)")
    ap.add_argument("--only-misses", action="store_true",
                    help="print only frames where a sensor gave nothing new")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="seconds to run (default: until Ctrl+C)")
    ap.add_argument("--gap-ms", type=float, default=30.0)
    ap.add_argument("--stall-ms", type=float, default=20.0)
    ap.add_argument("--no-reconnect", action="store_true")
    ap.add_argument("--idle-us", type=float, default=0.0)
    ap.add_argument("--out-dir", default=str(HERE / "logs"))
    args = ap.parse_args()

    sensors = [p.strip() for p in args.sensors.split(",") if p.strip()]
    bad = [p for p in sensors if p not in SENSORS]
    if bad or not sensors:
        ap.error(f"--sensors takes any of {', '.join(SENSORS)}; got {args.sensors!r}")
    sensors = [p for p in SENSORS if p in sensors]           # fixed order, no duplicates
    streams = tuple(s for s in sdm.STREAMS if s.split("_")[0] in sensors)
    tag = "-".join(sensors)

    busy_cmd = sdm.other_bus_user()
    if busy_cmd:
        print(f"Another process is using I2C bus {sdm.I2C_BUS}; stop it first:\n  {busy_cmd}")
        sys.exit(1)

    FRAME_S = sdm.FRAME_S
    gap_frames = max(1, math.ceil(args.gap_ms / 1000.0 / FRAME_S))
    stall_s = args.stall_ms / 1000.0
    paint = sdm.paint

    try:
        reader = SelectedReader(stall_s, not args.no_reconnect, sensors, args.idle_us / 1e6)
    except KeyboardInterrupt:
        print("\nAborted during init.")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"sel_{tag}_{stamp}.csv"
    log_path = out_dir / f"sel_{tag}_{stamp}_events.log"
    csv_file = open(csv_path, "w", newline="")
    log_file = open(log_path, "w")
    log_file.write(f"# imu_only_dropout_monitor: sensors={sensors} reconnect={not args.no_reconnect} "
                   f"idle_us={args.idle_us:g}; sensors not listed are not addressed\n")
    writer = csv.writer(csv_file)
    writer.writerow(["t_s"] + [c for s in streams for c in sdm.CSV_COLS[s]]
                    + [f"n_{s}" for s in streams])

    reader.start()
    t_wait = time.perf_counter()
    while (not all(reader.samples[s] for s in streams)
           and time.perf_counter() - t_wait < sdm.STARTUP_S):
        time.sleep(0.01)
    for s in streams:
        if not reader.samples[s]:
            print(paint(f"[select] {s}: no reading after {sdm.STARTUP_S:.0f} s", sdm.RED))
        reader.samples[s].clear()

    t0 = time.perf_counter()
    counts = {}
    readings = {s: 0 for s in streams}
    missed = {s: 0 for s in streams}
    sec_missed = {s: 0 for s in streams}
    miss_run = {s: 0 for s in streams}
    longest = {s: 0 for s in streams}
    announced_hang = None

    def log_event(t, kind, text):
        counts[kind] = counts.get(kind, 0) + 1
        line = f"[{t - t0:8.3f}s] {kind:<6} {text}"
        log_file.write(line + "\n")
        print(paint(line, sdm.EVENT_COLOR.get(kind, sdm.BOLD)), flush=True)

    print(f"\n[select] live. {FRAME_S * 1000:.0f} ms frames, GAP >= "
          f"{gap_frames * FRAME_S * 1000:.0f} ms, STALL >= {args.stall_ms:.0f} ms. Ctrl+C to stop.\n")
    print(header_line(streams))

    frame = 0
    try:
        while True:
            frame_end = t0 + (frame + 1) * FRAME_S
            if args.duration and frame_end - t0 > args.duration:
                break
            wait = frame_end + sdm.FRAME_GRACE_S - time.perf_counter()
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
            for s in streams:
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
                               for s in streams
                               for v in (row[s] if row[s] is not None
                                         else [math.nan] * len(sdm.CSV_COLS[s]))]
                            + [n_row[s] for s in streams])
            if missing or not args.only_misses:
                print(f"{t_frame:8.2f} |" + " |".join(sdm.cells(s, row[s]) for s in streams)
                      + " | " + paint(" ".join(missing), sdm.RED))

            frame += 1
            if frame % round(1.0 / FRAME_S) == 0:
                print(paint(f"── {t_frame:.0f} s ── frames missed in the last 1 s: "
                            + "  ".join(f"{s} {sec_missed[s]}" for s in streams)
                            + " | events so far: "
                            + ("  ".join(f"{k} {v}" for k, v in sorted(counts.items())) or "none"),
                            sdm.BOLD))
                sec_missed = {s: 0 for s in streams}
            if not args.only_misses and frame % round(5.0 / FRAME_S) == 0:
                print(header_line(streams))
    except KeyboardInterrupt:
        pass
    finally:
        reader.stop()
        reader.join(timeout=1.0)
        while reader.events:
            log_event(*reader.events.popleft())
        for s in streams:
            longest[s] = max(longest[s], miss_run[s])
        if reader.is_alive():
            print(paint("[select] reader thread still blocked in an I2C call at exit", sdm.RED))
        else:
            reader.close()
        csv_file.close()

        elapsed = frame * FRAME_S
        lines = [f"\nSummary over {elapsed:.1f} s ({frame} frames of {FRAME_S * 1000:.0f} ms) "
                 f"- sensors read: {', '.join(sensors)}; "
                 f"not addressed: {', '.join(p for p in SENSORS if p not in sensors) or 'none'}",
                 f"  {'stream':<11}{'readings':>9}{'frames missed':>17}{'longest gap':>13}"]
        for s in streams:
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
