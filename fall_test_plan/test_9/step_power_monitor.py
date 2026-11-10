#!/usr/bin/env python3
"""
step_power_monitor.py: live steps, strides and motor power, saved on Ctrl+C.

    python3 fall_test_plan/test_9/step_power_monitor.py              # alongside the ML controller
    python3 fall_test_plan/test_9/step_power_monitor.py --sensors    # no controller running: read FSRs itself
    python3 fall_test_plan/test_9/step_power_monitor.py --phase-resistance 0.1   # add copper loss to power

Updates one terminal line like Sensor_Tests/sensor_parse.py does:
strides and steps so far, cadence, motor current, speed, torque, temperature,
power now and the average power since the monitor started.

THIS SCRIPT NEVER TRANSMITS ON can0.
    It opens a raw CAN socket with a receive filter for the AK80-9 status frames
    (0x2968) and the MIT command frames other programs send (0x868), and only
    ever calls recv() on it. There is no send call anywhere in this file, and it
    imports no motor code. It cannot command the motor. Motor data only appears
    if something else (the controller) is driving the motor and the motor is
    sending status frames.

Where the steps come from
    default    The running controller's live stream: the RunLogger's UDP relay
               (127.0.0.1:47270). If the dashboard backend already owns that port,
               the monitor connects to the dashboard's /ws/live WebSocket instead.
               ML controller samples carry `stance`; TBE samples carry
               heel_fsr/toe_fsr, which are thresholded here. Nothing touches I2C.
               If the monitor takes the UDP port first, start the dashboard
               before the monitor, or the dashboard cannot bind it.
    --sensors  Reads the toe (AIN0) and heel (AIN1) FSRs on the ADS1115 (bus 1,
               0x48) the same way the ML controller does. Use only when no
               controller is running: two processes on the same ADC corrupt
               each other's conversions.

Steps and strides
    A heel strike is the stance flag going 0 -> 1, ignored if it comes less than
    --min-stride-s after the previous one. Only the exo leg is instrumented, so
    strides = heel strikes of that leg, and steps = 2 x strides (an estimate
    that assumes the other leg stepped in between).

Motor power (from CAN status frames, no voltage)
    Iq, speed and temperature come from the motor's status frames. Torque is
    Kt x |Iq|, with the sign of the last command frame seen on the bus (the
    reported Iq sign is unreliable). Output speed = ERPM / pole pairs / gear.
    Mechanical power P = torque x output speed. "P+ avg" is the positive
    (motoring) mechanical energy divided by elapsed time; "|P| avg" counts
    braking too. With --phase-resistance, copper loss 1.5 x R x Iq^2 is added
    to both. Bus voltage is not in the status frames, so supply power is not
    shown.

On Ctrl+C, saves into fall_test_plan/test_9/runs/<YYYYmmdd_HHMMSS>_<steps>steps/:
    gait.csv          every stance sample (time, stance, heel/toe FSR, source)
    heel_strikes.csv  time of every counted heel strike
    can.csv           every decoded status and command frame, with torque and power
    summary.json      totals, averages and settings
"""

import argparse
import csv
import json
import math
import socket
import struct
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent
RUNS_DIR = TEST_DIR / "runs"

# ── Motor / CAN (same values as ML_model/src/exo/deploy/motor.py) ────────────
CAN_INTERFACE = "can0"
MOTOR_ID = 0x68
STATUS_ID = (0x29 << 8) | MOTOR_ID      # servo-mode status, ~50 Hz, extended id
CMD_ID = (8 << 8) | MOTOR_ID            # MIT torque command, extended id
MIT_T_MIN, MIT_T_MAX = -22.0, 22.0      # torque encoding range the controller uses
KT_NM_PER_A = 0.7044                    # AK80-9 torque constant, V3 manual p.31
POLE_PAIRS = 21                         # AK80-9: ERPM / 21 = rotor rpm
GEAR_RATIO = 9.0                        # AK80-9 internal planetary
CMD_FRESH_S = 0.25                      # command frame older than this: torque sign unknown
MAX_DT_S = 0.1                          # longer gaps between status frames are not integrated

CAN_EFF_FLAG = 0x80000000
CAN_RTR_FLAG = 0x40000000
CAN_EFF_MASK = 0x1FFFFFFF
CAN_FRAME = struct.Struct("=IB3x8s")    # struct can_frame

# ── Live stream (dashboard/backend/run_logger.py, server.py) ─────────────────
RELAY_HOST, RELAY_PORT = "127.0.0.1", 47270
WS_URL = "ws://127.0.0.1:8000/ws/live"

# ── FSR stance (ML_model/configs/default.yaml deploy.sensor_adapter) ─────────
HEEL_FSR_THRESHOLD = 22446.0
TOE_FSR_THRESHOLD = 8737.2
FSR_DEBOUNCE_S = 0.10
FSR_FILTER_HZ = 20.0
I2C_BUS = 1
FSR_HZ = 200.0

DISP_INTERVAL = 0.25
STALE_S = 1.0                           # source silent this long: shown as stale


# ─────────────────────────────────────────────────────────────────────────────
# Steps
# ─────────────────────────────────────────────────────────────────────────────

class StanceFromFSR:
    """Low-pass, threshold and debounce, as ML_model's SensorAdapter._stance_flag."""

    def __init__(self):
        self.heel = self.toe = None
        self.stance = 0
        self.age = 0.0
        self.t_prev = None

    def update(self, t, heel, toe):
        dt = 0.0 if self.t_prev is None else max(0.0, t - self.t_prev)
        self.t_prev = t
        if self.heel is None:
            self.heel, self.toe = heel, toe
        else:
            k = 2 * math.pi * FSR_FILTER_HZ * dt
            a = k / (k + 1.0)
            self.heel += a * (heel - self.heel)
            self.toe += a * (toe - self.toe)
        raw = 1 if (self.heel > HEEL_FSR_THRESHOLD or self.toe >= TOE_FSR_THRESHOLD) else 0
        if raw != self.stance:
            self.age += dt
            if self.age >= FSR_DEBOUNCE_S:
                self.stance, self.age = raw, 0.0
        else:
            self.age = 0.0
        return self.stance


class StepCounter:
    def __init__(self, min_stride_s, max_stride_s):
        self.min_stride_s = min_stride_s
        self.max_stride_s = max_stride_s
        self.prev_stance = None
        self.strikes = []                 # times (s since start) of counted heel strikes
        self.rejected = 0
        self.periods = deque(maxlen=10)
        self.stance = None
        self.lock = threading.Lock()

    def update(self, t, stance):
        with self.lock:
            stance = int(stance)
            if self.prev_stance == 0 and stance == 1:
                if not self.strikes or t - self.strikes[-1] >= self.min_stride_s:
                    if self.strikes:
                        p = t - self.strikes[-1]
                        if p <= self.max_stride_s:
                            self.periods.append(p)
                    self.strikes.append(t)
                else:
                    self.rejected += 1
            self.prev_stance = stance
            self.stance = stance

    @property
    def strides(self):
        return len(self.strikes)

    @property
    def steps(self):
        return 2 * len(self.strikes)

    def cadence(self):
        """Steps per minute from the last few strides, or None."""
        with self.lock:
            if not self.periods:
                return None
            p = sorted(self.periods)[len(self.periods) // 2]
        return 120.0 / p


# ─────────────────────────────────────────────────────────────────────────────
# Gait sources
# ─────────────────────────────────────────────────────────────────────────────

class GaitSource(threading.Thread):
    """Base: feeds (t, stance, heel, toe) into the step counter and the log."""

    def __init__(self, t0, counter):
        super().__init__(daemon=True)
        self.t0 = t0
        self.counter = counter
        self.stop_event = threading.Event()
        self.rows = []                    # (t_s, wall, stance, heel, toe, source)
        self.last_t = None
        self.label = ""
        self.error = ""
        self.run_id = None
        self.fsr = StanceFromFSR()

    def push(self, stance=None, heel=None, toe=None):
        t = time.monotonic() - self.t0
        if stance is None:
            if heel is None or toe is None:
                return
            stance = self.fsr.update(t, float(heel), float(toe))
        self.counter.update(t, stance)
        self.rows.append((round(t, 4), round(time.time(), 4), int(stance),
                          heel, toe, self.label))
        self.last_t = t

    def push_sample(self, msg):
        """One RunLogger sample (dict) from the UDP relay or WebSocket."""
        if not isinstance(msg, dict):
            return
        self.run_id = msg.get("run_id", self.run_id)
        heel = msg.get("heel_fsr_raw", msg.get("heel_fsr"))
        toe = msg.get("toe_fsr_raw", msg.get("toe_fsr"))
        stance = msg.get("stance")
        try:
            stance = None if stance is None else int(float(stance) >= 0.5)
        except (TypeError, ValueError):
            stance = None
        self.push(stance, heel, toe)

    def stop(self):
        self.stop_event.set()


class StreamSource(GaitSource):
    """RunLogger UDP relay, or the dashboard WebSocket if the port is taken."""

    def __init__(self, t0, counter, ws_url):
        super().__init__(t0, counter)
        self.ws_url = ws_url
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            self.sock.bind((RELAY_HOST, RELAY_PORT))
            self.sock.settimeout(0.2)
            self.label = "udp"
        except OSError:
            self.sock.close()
            self.sock = None
            self.label = "ws"

    def run(self):
        if self.sock is not None:
            while not self.stop_event.is_set():
                try:
                    data = self.sock.recv(65536)
                except socket.timeout:
                    continue
                except OSError as exc:
                    self.error = f"udp: {exc}"
                    break
                try:
                    self.push_sample(json.loads(data))
                except ValueError:
                    pass
            self.sock.close()
            return

        try:
            from websockets.sync.client import connect
        except ImportError:
            self.error = "websockets package missing"
            return
        while not self.stop_event.is_set():
            try:
                with connect(self.ws_url, open_timeout=2.0) as ws:
                    self.error = ""
                    while not self.stop_event.is_set():
                        try:
                            data = ws.recv(timeout=0.2)
                        except TimeoutError:
                            continue
                        try:
                            self.push_sample(json.loads(data))
                        except ValueError:
                            pass
            except Exception as exc:           # dashboard down or restarted: retry
                self.error = f"ws: {type(exc).__name__}"
                self.stop_event.wait(1.0)


class FSRSource(GaitSource):
    """Toe (AIN0) and heel (AIN1) FSRs on the ADS1115, raw counts. Same timing
    as TimedADS1115 in ML_model/src/exo/deploy/jetson_io.py (copied, so this
    file imports no deploy or motor code)."""

    ADDR = 0x48
    REG_CONV, REG_CFG = 0x00, 0x01
    CFG = ([0xC3, 0xE3], [0xD3, 0xE3])     # ±4.096 V, 860 SPS, single shot
    CONV_S = 0.0013

    def __init__(self, t0, counter, bus):
        super().__init__(t0, counter)
        import smbus2
        self.bus = smbus2.SMBus(bus)
        self.label = "fsr"

    def _convert(self, ch):
        self.bus.write_i2c_block_data(self.ADDR, self.REG_CFG, self.CFG[ch])
        time.sleep(self.CONV_S)
        d = self.bus.read_i2c_block_data(self.ADDR, self.REG_CONV, 2)
        raw = (d[0] << 8) | d[1]
        return raw - 0x10000 if raw >= 0x8000 else raw

    def run(self):
        period = 1.0 / FSR_HZ
        nxt = time.monotonic()
        while not self.stop_event.is_set():
            try:
                toe = self._convert(0)
                heel = self._convert(1)
                self.error = ""
                self.push(None, heel, toe)
            except OSError as exc:
                self.error = f"i2c: {exc.strerror or exc}"
                time.sleep(0.05)
            nxt += period
            delay = nxt - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                nxt = time.monotonic()
        try:
            self.bus.close()
        except OSError:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# CAN listener (receive only)
# ─────────────────────────────────────────────────────────────────────────────

class CanListener(threading.Thread):
    def __init__(self, t0, interface, phase_r):
        super().__init__(daemon=True)
        self.t0 = t0
        self.phase_r = phase_r
        self.stop_event = threading.Event()
        self.rows = []
        self.error = ""
        self.status = None                # latest decoded status
        self.cmd_nm = None
        self.cmd_t = None
        self.n_status = self.n_cmd = 0
        self.t_first = self.t_last = None
        self.e_pos = self.e_abs = self.e_net = self.e_cu = 0.0
        self.iq_abs_sum = self.iq_sq_sum = 0.0
        self.iq_peak = 0.0
        self.dt_sum = 0.0
        self.p_now = None
        self.lock = threading.Lock()

        self.sock = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        mask = CAN_EFF_MASK | CAN_EFF_FLAG | CAN_RTR_FLAG
        filters = struct.pack("=IIII", STATUS_ID | CAN_EFF_FLAG, mask, CMD_ID | CAN_EFF_FLAG, mask)
        self.sock.setsockopt(socket.SOL_CAN_RAW, socket.CAN_RAW_FILTER, filters)
        self.sock.bind((interface,))
        self.sock.settimeout(0.2)

    def run(self):
        while not self.stop_event.is_set():
            try:
                frame = self.sock.recv(CAN_FRAME.size)
            except socket.timeout:
                continue
            except OSError as exc:            # e.g. can0 went down
                self.error = f"can: {exc.strerror or exc}"
                self.stop_event.wait(0.5)
                continue
            if len(frame) < CAN_FRAME.size:
                continue
            can_id, dlc, data = CAN_FRAME.unpack(frame)
            if not can_id & CAN_EFF_FLAG:
                continue
            arb = can_id & CAN_EFF_MASK
            t = time.monotonic() - self.t0
            if arb == CMD_ID and dlc >= 8:
                self._on_command(t, data)
            elif arb == STATUS_ID and dlc >= 6:
                self._on_status(t, data[:dlc])
        self.sock.close()

    def _on_command(self, t, d):
        u = ((d[6] & 0x0F) << 8) | d[7]
        nm = u * (MIT_T_MAX - MIT_T_MIN) / 4095.0 + MIT_T_MIN
        with self.lock:
            self.cmd_nm, self.cmd_t = nm, t
            self.n_cmd += 1
        self.rows.append((round(t, 4), round(time.time(), 4), "cmd", round(nm, 4),
                          "", "", "", "", "", "", "", ""))

    def _on_status(self, t, d):
        pos = struct.unpack(">h", d[0:2])[0] * 0.1
        erpm = struct.unpack(">h", d[2:4])[0] * 10.0
        iq = struct.unpack(">h", d[4:6])[0] * 0.01
        temp = d[6] if len(d) > 6 else None
        err = d[7] if len(d) > 7 else None
        omega = erpm / POLE_PAIRS / GEAR_RATIO * 2 * math.pi / 60.0     # output rad/s
        with self.lock:
            fresh = self.cmd_t is not None and t - self.cmd_t <= CMD_FRESH_S
            sign = (1.0 if self.cmd_nm >= 0 else -1.0) if fresh else None
            tq_abs = KT_NM_PER_A * abs(iq)
            p_abs = tq_abs * abs(omega)
            p_net = sign * tq_abs * omega if sign is not None else None
            p_cu = 1.5 * self.phase_r * iq * iq if self.phase_r is not None else 0.0
            if self.t_last is not None:
                dt = min(t - self.t_last, MAX_DT_S)
                self.dt_sum += dt
                self.e_abs += (p_abs + p_cu) * dt
                self.e_cu += p_cu * dt
                if p_net is not None:
                    self.e_net += p_net * dt
                    self.e_pos += (max(p_net, 0.0) + p_cu) * dt
                else:                      # no command on the bus: direction unknown
                    self.e_pos += p_cu * dt
                self.iq_abs_sum += abs(iq) * dt
                self.iq_sq_sum += iq * iq * dt
            else:
                self.t_first = t
            self.iq_peak = max(self.iq_peak, abs(iq))
            self.t_last = t
            self.n_status += 1
            self.p_now = (max(p_net, 0.0) if p_net is not None else 0.0) + p_cu
            tq = sign * tq_abs if sign is not None else None
            self.status = {"pos_deg": pos, "erpm": erpm, "iq_a": iq, "temp_c": temp,
                           "error": err, "omega_deg_s": math.degrees(omega), "torque_nm": tq}
        self.rows.append((round(t, 4), round(time.time(), 4), "status", "",
                          round(pos, 1), erpm, round(iq, 2), temp, err,
                          round(math.degrees(omega), 2),
                          "" if tq is None else round(tq, 4),
                          "" if p_net is None else round(p_net, 4)))

    def averages(self, elapsed):
        with self.lock:
            dur = self.dt_sum
            return {
                "p_pos_avg_w": self.e_pos / elapsed if elapsed > 0 else None,
                "p_abs_avg_w": self.e_abs / elapsed if elapsed > 0 else None,
                "p_net_avg_w": self.e_net / elapsed if elapsed > 0 else None,
                "copper_loss_avg_w": (self.e_cu / elapsed if elapsed > 0 else None)
                                     if self.phase_r is not None else None,
                "energy_pos_j": self.e_pos, "energy_abs_j": self.e_abs, "energy_net_j": self.e_net,
                "mean_abs_iq_a": self.iq_abs_sum / dur if dur > 0 else None,
                "rms_iq_a": math.sqrt(self.iq_sq_sum / dur) if dur > 0 else None,
                "peak_iq_a": self.iq_peak if self.n_status else None,
                "status_frames": self.n_status, "command_frames": self.n_cmd,
                "status_coverage_s": dur,
            }

    def stop(self):
        self.stop_event.set()


# ─────────────────────────────────────────────────────────────────────────────
# Display and save
# ─────────────────────────────────────────────────────────────────────────────

def f(v, fmt, width):
    return f"{v:{width}{fmt}}" if v is not None else " " * (width - 3) + "---"


HEADER = (f"{'Time s':>7}  {'Src':>5} {'St':>2}  |  {'Strides':>7} {'Steps~':>6} {'Cad/min':>7}  |  "
          f"{'Iq A':>6} {'Spd deg/s':>9} {'Tq Nm':>6} {'T C':>4}  |  {'P W':>6} {'P+avg W':>7} {'|P|avg W':>8}")


def line(now, gait, counter, can_l):
    if gait.error:
        src = "ERR"
    elif gait.last_t is None or now - gait.last_t > STALE_S:
        src = "wait" if gait.last_t is None else "stale"
    else:
        src = gait.label
    st = counter.stance
    s = can_l.status if can_l is not None else None
    can_fresh = s is not None and can_l.t_last is not None and now - can_l.t_last <= STALE_S
    avg = can_l.averages(now) if can_l is not None else {}
    out = (f"{now:7.1f}  {src:>5} {('-' if st is None else st):>2}  |  "
           f"{counter.strides:7d} {counter.steps:6d} {f(counter.cadence(), '.0f', 7)}  |  ")
    if can_fresh:
        out += (f"{s['iq_a']:6.2f} {s['omega_deg_s']:9.1f} {f(s['torque_nm'], '.2f', 6)} "
                f"{f(s['temp_c'], 'd', 4)}  |  {f(can_l.p_now, '.1f', 6)} ")
    else:
        out += f"{'no CAN' if can_l is None or can_l.n_status == 0 else 'stale':>28}  |  {'---':>6} "
    out += f"{f(avg.get('p_pos_avg_w'), '.2f', 7)} {f(avg.get('p_abs_avg_w'), '.2f', 8)}"
    return out


def save(t_start_wall, elapsed, gait, counter, can_l, args):
    stamp = datetime.fromtimestamp(t_start_wall).strftime("%Y%m%d_%H%M%S")
    out = RUNS_DIR / f"{stamp}_{counter.steps}steps"
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "gait.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t_s", "wall_time", "stance", "heel_fsr", "toe_fsr", "source"])
        w.writerows(list(gait.rows))
    with open(out / "heel_strikes.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["stride", "t_s", "period_s"])
        prev = None
        for i, t in enumerate(counter.strikes):
            w.writerow([i + 1, round(t, 4), "" if prev is None else round(t - prev, 4)])
            prev = t
    with open(out / "can.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t_s", "wall_time", "frame", "cmd_nm", "position_deg", "speed_erpm",
                    "current_a", "temp_c", "error", "speed_out_deg_s", "torque_nm", "power_w"])
        if can_l is not None:
            w.writerows(list(can_l.rows))

    walk = [b - a for a, b in zip(counter.strikes, counter.strikes[1:])
            if args.min_stride_s <= b - a <= args.max_stride_s]
    summary = {
        "started": datetime.fromtimestamp(t_start_wall).isoformat(timespec="seconds"),
        "duration_s": round(elapsed, 2),
        "gait_source": {"udp": "RunLogger UDP relay", "ws": "dashboard /ws/live",
                        "fsr": "ADS1115 FSRs (--sensors)"}.get(gait.label, gait.label),
        "controller_run_id": gait.run_id,
        "strides": counter.strides,
        "steps_est": counter.steps,
        "rejected_double_heel_strikes": counter.rejected,
        "walking_strides": len(walk),
        "median_stride_s": sorted(walk)[len(walk) // 2] if walk else None,
        "cadence_steps_min": 120.0 / sorted(walk)[len(walk) // 2] if walk else None,
        "gait_samples": len(gait.rows),
        "motor": can_l.averages(elapsed) if can_l is not None else None,
        "settings": {"min_stride_s": args.min_stride_s, "max_stride_s": args.max_stride_s,
                     "phase_resistance_ohm": args.phase_resistance, "can": args.can,
                     "kt_nm_per_a": KT_NM_PER_A, "pole_pairs": POLE_PAIRS, "gear_ratio": GEAR_RATIO,
                     "heel_fsr_threshold": HEEL_FSR_THRESHOLD, "toe_fsr_threshold": TOE_FSR_THRESHOLD},
        "notes": "steps_est = 2 x strides of the exo leg. Power is mechanical "
                 "(Kt*|Iq| x output speed, sign from the last command frame), plus copper "
                 "loss only if phase_resistance_ohm is set. No bus voltage is available.",
    }
    with open(out / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    return out, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sensors", action="store_true",
                    help="read the FSRs on I2C instead of the controller stream (no controller running)")
    ap.add_argument("--can", default=CAN_INTERFACE, help="CAN interface to listen on (default can0)")
    ap.add_argument("--no-can", action="store_true", help="steps only, do not open CAN")
    ap.add_argument("--ws-url", default=WS_URL, help=f"dashboard live WebSocket (default {WS_URL})")
    ap.add_argument("--i2c-bus", type=int, default=I2C_BUS, help="I2C bus for --sensors (default 1)")
    ap.add_argument("--min-stride-s", type=float, default=0.5,
                    help="ignore a heel strike this soon after the previous one (default 0.5)")
    ap.add_argument("--max-stride-s", type=float, default=2.5,
                    help="longer intervals are pauses, not used for cadence (default 2.5)")
    ap.add_argument("--phase-resistance", type=float, default=None,
                    help="motor phase resistance in ohm; adds copper loss 1.5*R*Iq^2 to power")
    args = ap.parse_args()

    t0 = time.monotonic()
    t_start_wall = time.time()
    counter = StepCounter(args.min_stride_s, args.max_stride_s)

    try:
        gait = FSRSource(t0, counter, args.i2c_bus) if args.sensors else StreamSource(t0, counter, args.ws_url)
    except (OSError, ImportError) as exc:
        sys.exit(f"Could not open the gait source: {exc}")

    can_l = None
    if not args.no_can:
        try:
            can_l = CanListener(t0, args.can, args.phase_resistance)
        except OSError as exc:
            print(f"[CAN] {args.can} not available ({exc.strerror or exc}); motor columns will show 'no CAN'.")

    src_txt = {"udp": f"controller stream (UDP {RELAY_HOST}:{RELAY_PORT})",
               "ws": f"dashboard WebSocket {args.ws_url}",
               "fsr": f"FSRs on I2C bus {args.i2c_bus} (0x48)"}[gait.label]
    print(f"Steps from: {src_txt}")
    print(f"Motor from: {'listening on ' + args.can + ' (receive only)' if can_l else 'off'}")
    if gait.label == "udp":
        print("Note: this monitor now holds the UDP relay port; start the dashboard before it, not after.")
    print("Ctrl+C to stop and save.\n")
    print(HEADER)
    print("-" * len(HEADER))

    gait.start()
    if can_l is not None:
        can_l.start()

    try:
        while True:
            now = time.monotonic() - t0
            sys.stdout.write("\r" + line(now, gait, counter, can_l) + "  ")
            sys.stdout.flush()
            time.sleep(DISP_INTERVAL)
    except KeyboardInterrupt:
        pass
    finally:
        elapsed = time.monotonic() - t0
        gait.stop()
        if can_l is not None:
            can_l.stop()
        gait.join(timeout=2.0)
        if can_l is not None:
            can_l.join(timeout=2.0)

    out, s = save(t_start_wall, elapsed, gait, counter, can_l, args)
    m = s["motor"] or {}

    def show(v, unit, nd=2):
        return f"{v:.{nd}f} {unit}" if v is not None else "n/a"

    print(f"\n\nStopped after {elapsed:.1f} s")
    print(f"  Strides: {s['strides']}   Steps (est.): {s['steps_est']}   "
          f"Cadence: {show(s['cadence_steps_min'], 'steps/min', 0)}")
    if gait.error:
        print(f"  Gait source error: {gait.error}")
    if can_l is not None:
        print(f"  Motor: {m.get('status_frames', 0)} status / {m.get('command_frames', 0)} command frames   "
              f"mean |Iq| {show(m.get('mean_abs_iq_a'), 'A')}   peak {show(m.get('peak_iq_a'), 'A')}")
        print(f"  Avg power: P+ {show(m.get('p_pos_avg_w'), 'W')}   |P| {show(m.get('p_abs_avg_w'), 'W')}   "
              f"net {show(m.get('p_net_avg_w'), 'W')}"
              + ("" if args.phase_resistance is None else
                 f"   (copper {show(m.get('copper_loss_avg_w'), 'W')})"))
        if can_l.error:
            print(f"  CAN error: {can_l.error}")
    print(f"  Saved {out.relative_to(TEST_DIR.parents[1])}/")


if __name__ == "__main__":
    main()
