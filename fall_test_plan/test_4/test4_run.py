#!/usr/bin/env python3
"""
test4_run.py — Test 4: control loop and CAN command rate (R-05), on the bench.

Runs the TCN controller loop the way the powered runner
(ML_model/scripts/jetson_deploy.py) does — read both IMUs, the FSRs and the
encoder, run the TCN, apply the safety checks (session ramp, loop-overrun
watchdog, torque clamp / slew), send one MIT torque frame over CAN, log the row,
stream Teleplot — and records for every loop its start and end, the time spent
in each stage, and every CAN send.

No walking is needed. The work a loop does does not depend on the gait:
the IMUs report at a fixed rate, the FSR/encoder reads are the same I2C
transactions, the TCN does the same arithmetic on any window, and the runner
sends one CAN frame per loop in stance and swing alike. Gait injection
(on by default) feeds a recorded walk to the controller while the real sensors
are still read every loop, so the stance gate, ramps and torque path behave as
in walking. The motor gets 0 N·m frames by default (--torque-mode zero): the
CAN traffic and the motor's replies are identical, but the shaft does not move.

Logging is buffered: rows go into preallocated RAM arrays and are written to
disk only after the timed window. --flush-every N makes the marked run that
writes + fsyncs the log every N seconds inside the loop.

Phases
------
    check   preflight: CAN link, I2C sensors, motor reply, TCN load
    idle    10 s reference: the same loop, scheduler and logger with no work
    load    the full-load run (default 600 s)
    all     check -> idle -> load [-> flush run] -> analysis, each in a fresh
            process, into one session folder

    python3 fall_test_plan/test_4/test4_run.py --phase all
    python3 fall_test_plan/test_4/test4_run.py --phase all --sim --duration 30

See README.md in this folder.
"""

import argparse
import csv
import inspect
import json
import os
import platform
import re
import select
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

TEST_DIR = Path(__file__).resolve().parent
REPO = TEST_DIR.parents[1]
DEFAULT_GAIT = (REPO / "Data collection" / "data"
                / "data_collection_20260404_215743_tightshoes_200Hz_2kmph.csv")

# raw-frame keys the SensorAdapter expects (ML_model/src/exo/deploy/jetson_io.py)
FRAME_KEYS = [
    "foot_ax", "foot_ay", "foot_az", "foot_gx", "foot_gy", "foot_gz",
    "shank_ax", "shank_ay", "shank_az", "shank_gx", "shank_gy", "shank_gz",
    "ankle_encoder_deg", "heel_fsr_raw", "toe_fsr_raw",
]

MOTOR_ID = 0x68
CMD_ARB_ID = (8 << 8) | MOTOR_ID      # MIT mode (8) torque frame, extended id
CAN_BITRATE = 1_000_000

# I2C devices on the exo (bus 1): ADS1115 FSR ADC, AS5600 encoder, two BNO085s
I2C_DEVICES = {0x48: "ADS1115 (FSR ADC)", 0x36: "AS5600 (ankle encoder)",
               0x4A: "BNO085 foot IMU (IMU-A)", 0x4B: "BNO085 shank IMU (IMU-B)"}

OOM_PATTERN = re.compile(r"out of memory|oom-kill|oom_reaper|killed process", re.I)

_STOP = False


def _handle_signal(signum, frame):
    """First Ctrl-C ends the run cleanly; a second one interrupts whatever is
    blocking (e.g. an IMU reconnect loop) — logs are still written."""
    global _STOP
    if _STOP:
        raise KeyboardInterrupt
    _STOP = True


# ── system / link information ────────────────────────────────────────────────

def _run(cmd, timeout=5.0):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout + r.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return -1, str(exc)


def system_info():
    def read(path):
        try:
            return Path(path).read_text().strip("\x00\n ")
        except OSError:
            return None

    cpus = sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*"))
    info = {
        "hostname": socket.gethostname(),
        "model": read("/proc/device-tree/model"),
        "l4t": read("/etc/nv_tegra_release"),
        "python": sys.version.split()[0],
        "python_exe": sys.executable,
        "numpy": np.__version__,
        "nvpmodel": _run(["nvpmodel", "-q"])[1],
        "governors": sorted({read(c / "cpufreq/scaling_governor") for c in cpus} - {None}),
        "cpu_max_khz": sorted({read(c / "cpufreq/scaling_max_freq") for c in cpus} - {None}),
        "online_cpus": os.cpu_count(),
        "loadavg_at_start": os.getloadavg(),
        "git_commit": _run(["git", "-C", str(REPO), "rev-parse", "HEAD"])[1],
        "git_branch": _run(["git", "-C", str(REPO), "rev-parse", "--abbrev-ref", "HEAD"])[1],
        "git_dirty_files": _run(["git", "-C", str(REPO), "status", "--porcelain"])[1].splitlines(),
    }
    for mod in ("can", "torch", "onnxruntime"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:                                  # noqa: BLE001
            info[mod] = None
    return info


def can_link(channel):
    """`ip -details -statistics link show` as a dict, or {"error": ...}."""
    rc, out = _run(["ip", "-json", "-details", "-statistics", "link", "show", channel])
    if rc != 0:
        return {"error": out}
    try:
        return json.loads(out)[0]
    except (ValueError, IndexError) as exc:
        return {"error": f"{exc}: {out[:200]}"}


def check_can_link(channel, bitrate=CAN_BITRATE):
    """None if the link is up at the right bitrate, else the problem + the fix."""
    fix = (f"sudo ip link set {channel} down; "
           f"sudo ip link set {channel} up type can bitrate {bitrate} restart-ms 100")
    link = can_link(channel)
    if "error" in link:
        return f"CAN interface {channel} not found ({link['error']})"
    if "UP" not in link.get("flags", []):
        return f"CAN interface {channel} is down. Run: {fix}"
    actual = link.get("linkinfo", {}).get("info_data", {}).get("bittiming", {}).get("bitrate")
    if actual != bitrate:
        return f"CAN interface {channel} bitrate is {actual}, motor expects {bitrate}. Run: {fix}"
    return None


def probe_i2c(bus_num, addrs):
    """Addresses on `bus_num` that do not acknowledge a read."""
    import smbus2
    missing = []
    with smbus2.SMBus(bus_num) as bus:
        for a in addrs:
            try:
                bus.read_byte(a)
            except OSError:
                missing.append(a)
    return missing


def kernel_oom_check(wall_start, wall_end):
    """Search the kernel log for OOM events between the two wall-clock times."""
    rc, out = _run(["journalctl", "-k", "--no-pager", "-o", "short-iso",
                    f"--since=@{int(wall_start) - 1}", f"--until=@{int(wall_end) + 2}"],
                   timeout=20.0)
    if rc != 0:
        return {"method": "journalctl -k", "ok": None, "error": out[-300:]}
    hits = [ln for ln in out.splitlines() if OOM_PATTERN.search(ln)]
    return {"method": "journalctl -k", "ok": not hits, "matches": hits[:20]}


# ── recording ───────────────────────────────────────────────────────────────

LOOP_COLUMNS = [
    # name, dtype
    ("seq", np.int64), ("timed", np.int8),
    ("t_start", np.float64), ("t_end", np.float64), ("sleep_req_ms", np.float32),
    ("imu_ms", np.float32), ("fsr_enc_ms", np.float32), ("step_ms", np.float32),
    ("predict_ms", np.float32), ("send_ms", np.float32), ("log_ms", np.float32),
    ("telemetry_ms", np.float32), ("flush_ms", np.float32),
    ("buffer_ready", np.int8), ("stance", np.int8), ("watchdog", np.int8),
    ("pred_nm_per_kg", np.float32), ("assist_cmd_nm", np.float32),
    ("session_ramp", np.float32), ("target_nm", np.float32), ("sent_nm", np.float32),
    ("can_cmd_ok", np.int8), ("can_err", np.int8),
]


class LoopRecorder:
    """Preallocated per-loop columns in RAM — the buffered logger."""

    def __init__(self, capacity):
        self.cols = {name: np.zeros(capacity, dtype=dt) for name, dt in LOOP_COLUMNS}
        self.raw = np.zeros((capacity, len(FRAME_KEYS)), dtype=np.float32)
        self.n = 0
        self.grow_events = 0

    def reserve(self, i):
        if i >= len(self.cols["seq"]):          # never drop a row: double the buffers
            for k, v in self.cols.items():
                self.cols[k] = np.concatenate([v, np.zeros_like(v)])
            self.raw = np.concatenate([self.raw, np.zeros_like(self.raw)])
            self.grow_events += 1
        self.n = i + 1

    def rows(self, start, stop):
        cols = [self.cols[name][start:stop].tolist() for name, _ in LOOP_COLUMNS]
        cols += [self.raw[start:stop, j].tolist() for j in range(len(FRAME_KEYS))]
        return zip(*cols)

    @staticmethod
    def header():
        return [name for name, _ in LOOP_COLUMNS] + [f"raw_{k}" for k in FRAME_KEYS]

    def write_csv(self, path):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(self.header())
            w.writerows(self.rows(0, self.n))
            f.flush()
            os.fsync(f.fileno())


class SendLog:
    """Wraps a python-can bus's send() and records the time of every send."""

    def __init__(self, bus, capacity):
        self.t0 = np.zeros(capacity)
        self.t1 = np.zeros(capacity)
        self.arb = np.zeros(capacity, dtype=np.uint32)
        self.ok = np.zeros(capacity, dtype=np.int8)
        self.loop = np.full(capacity, -1, dtype=np.int64)
        self.errors = []            # (send index, repr) — rare
        self.n = 0
        self.current_loop = -1
        self._send = bus.send
        bus.send = self.send        # instance attribute shadows the method

    def send(self, msg, timeout=None):
        t0 = time.perf_counter()
        err = None
        try:
            self._send(msg, timeout)
        except Exception as exc:                           # noqa: BLE001
            err = exc
        t1 = time.perf_counter()
        i = self.n
        if i == len(self.t0):
            for name in ("t0", "t1", "arb", "ok", "loop"):
                a = getattr(self, name)
                setattr(self, name, np.concatenate([a, np.zeros_like(a)]))
        self.t0[i], self.t1[i], self.arb[i] = t0, t1, msg.arbitration_id
        self.ok[i] = err is None
        self.loop[i] = self.current_loop
        self.n = i + 1
        if err is not None:
            self.errors.append((i, repr(err)))
        # The deploy runner would crash on a send error; here it is recorded and
        # the run continues so the whole window is measured. The analysis fails
        # the test on any send error.

    def write_csv(self, path):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["idx", "loop_seq", "t_send_start", "t_send_end", "arb_id", "ok"])
            for i in range(self.n):
                w.writerow([i, int(self.loop[i]), repr(float(self.t0[i])),
                            repr(float(self.t1[i])), f"0x{int(self.arb[i]):X}", int(self.ok[i])])


class DeployLogger:
    """The per-loop logging of jetson_deploy.py, so the loop carries the same
    load: one csv.DictWriter row, the dashboard RunLogger (in-memory buffer, UDP
    relay, Parquet writer thread every 500 rows) and a console line every 50 loops."""

    FIELDS = ["t_s", *FRAME_KEYS, "predicted_nm_per_kg", "predicted_nm",
              "assist_cmd_nm", "sent_torque_nm", "session_ramp", "stance",
              "ramp", "buffer_ready", "ankle_angle_deg", "calibrated",
              "assist_on", "assist_scale", "loop_ms"]

    def __init__(self, ml_root, csv_path, meta, print_every=50):
        root = Path(ml_root).expanduser().resolve().parent
        sys.path.insert(0, str(root))
        try:
            from dashboard.backend.run_logger import RunLogger
        except ModuleNotFoundError as exc:
            raise SystemExit(
                f"--logger deploy needs the dashboard RunLogger at {root}/dashboard/backend/"
                f"run_logger.py ({exc}). Use the ML branch checkout as --ml-root (README), "
                "or --logger buffered to log to RAM only.") from exc
        self.run_logger = RunLogger(meta=meta)
        self.run_logger.start()
        self.file = open(csv_path, "w", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=self.FIELDS)
        self.writer.writeheader()
        self.print_every = print_every
        self.i = 0

    def log(self, loop_start, t_now, frame, res, mass, sent, session_ramp, ankle_deg,
            assist_scale, loop_ms):
        pred_kg = res["predicted_nm_per_kg"]
        row = {"t_s": round(t_now, 4),
               **{k: round(frame[k], 5) for k in FRAME_KEYS},
               "predicted_nm_per_kg": round(pred_kg, 5),
               "predicted_nm": round(pred_kg * mass, 4),
               "assist_cmd_nm": round(res["command_nm"], 4),
               "sent_torque_nm": round(sent, 4),
               "session_ramp": round(session_ramp, 3),
               "stance": int(res["stance"]), "ramp": round(res["ramp"], 3),
               "buffer_ready": int(res["buffer_ready"]),
               "ankle_angle_deg": round(ankle_deg, 2),
               "calibrated": 1, "assist_on": 1,
               "assist_scale": round(assist_scale, 3),
               "loop_ms": round(loop_ms, 2)}
        self.writer.writerow(row)
        self.run_logger.log_sample(t=loop_start, **row)
        if self.print_every and self.i % self.print_every == 0:
            print(f"[{t_now:6.1f}s] pred={pred_kg:+.3f} Nm/kg  assist={res['command_nm']:6.2f}  "
                  f"sent={sent:+.2f} Nm  sess_ramp={session_ramp:.2f}  "
                  f"stance={int(res['stance'])}  loop={loop_ms:.1f}ms")
        self.i += 1

    def close(self, status):
        self.file.close()
        self.run_logger.stop(status=status)


class StreamFlusher:
    """--flush-every: write + fsync the rows logged so far, inside the loop."""

    def __init__(self, path):
        self.f = open(path, "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(LoopRecorder.header())
        self.done = 0

    def flush(self, rec, upto):
        self.w.writerows(rec.rows(self.done, upto))
        self.f.flush()
        os.fsync(self.f.fileno())
        self.done = upto

    def close(self):
        self.f.close()


# ── hardware / controller set-up ─────────────────────────────────────────────

class HardwareSensors:
    """FastDualIMUReader + TBE SensorData, read exactly like the ML runner's
    JetsonSensors.read(), with each half timed."""

    def __init__(self, imu_bus):
        sys.path.insert(0, str(REPO / "Sensor_Tests" / "IMUs" / "BNO085"))
        sys.path.insert(0, str(REPO / "TBE_controller"))
        from bno085_live_dual_fast import FastDualIMUReader
        from data_obtainer import SensorData
        from utilities import Logger

        self._imus = FastDualIMUReader(bus=imu_bus)
        self._fsr_enc = SensorData(Logger())
        self._last = {k: 0.0 for k in FRAME_KEYS}

    def read(self):
        t0 = time.perf_counter()
        sa, sb = self._imus.read()
        t1 = time.perf_counter()
        self._fsr_enc.readSensors()
        t2 = time.perf_counter()

        f = self._last
        if sa is not None:
            f["foot_ax"], f["foot_ay"], f["foot_az"] = sa["accel"]
            f["foot_gx"], f["foot_gy"], f["foot_gz"] = sa["gyro"]
        if sb is not None:
            f["shank_ax"], f["shank_ay"], f["shank_az"] = sb["accel"]
            f["shank_gx"], f["shank_gy"], f["shank_gz"] = sb["gyro"]
        f["ankle_encoder_deg"] = float(self._fsr_enc.encoder_data)
        f["heel_fsr_raw"] = float(self._fsr_enc.filtered_heel_fsr)
        f["toe_fsr_raw"] = float(self._fsr_enc.filtered_toe_fsr)
        return dict(f), t1 - t0, t2 - t1

    def shutdown(self):
        # SensorData.shutdown() also zeroes + exits its own motor-mode handle
        try:
            self._fsr_enc.shutdown()
        except Exception:                                  # noqa: BLE001
            pass


class GaitPlayer:
    """A recorded walk resampled to the control rate, played in real time."""

    def __init__(self, csv_path, rate_hz):
        import pandas as pd
        df = pd.read_csv(csv_path).rename(columns={"time_s": "timestamp_s"})
        df = df.sort_values("timestamp_s").interpolate().ffill().bfill()
        t = df["timestamp_s"].to_numpy()
        t = t - t[0]
        grid = np.arange(0.0, t[-1], 1.0 / rate_hz)
        cols = {k: np.interp(grid, t, df[k].to_numpy()) for k in FRAME_KEYS}
        self.frames = [{k: float(cols[k][i]) for k in FRAME_KEYS} for i in range(len(grid))]
        self.rate_hz = rate_hz
        self.length_s = len(grid) / rate_hz

    def at(self, t_s):
        return dict(self.frames[int(t_s * self.rate_hz) % len(self.frames)])


def load_ml(ml_root):
    src = Path(ml_root).expanduser().resolve() / "src"
    sys.path.insert(0, str(src))
    try:
        from exo.config import Config, replace
        from exo.deploy.jetson_io import Teleplot
        from exo.deploy.motor import MIT_T_MAX, MotorInterface
        from exo.deploy.runtime import ExoController
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"\nCannot import the ML controller from {src}: {exc}\n"
            "This checkout's ML_model/src is incomplete (exo/data/*.py is missing on main).\n"
            "Point --ml-root at a checkout of the ML controller branch, for example:\n"
            "    git worktree add ../exo_ws_ml axilles-59-ml-control-model\n"
            "    ... --ml-root ../exo_ws_ml/ML_model\n") from exc
    return {"Config": Config, "replace": replace, "Teleplot": Teleplot,
            "MIT_T_MAX": MIT_T_MAX, "MotorInterface": MotorInterface,
            "ExoController": ExoController, "src": str(src)}


def resolve_run_dir(ml_root, run_dir):
    if run_dir:
        return Path(run_dir).expanduser().resolve()
    runs = Path(ml_root).expanduser().resolve() / "runs"
    if (runs / "deploy" / "deploy_metadata.json").exists():
        return runs / "deploy"
    found = sorted(p.parent for p in runs.glob("*/deploy_metadata.json"))
    if not found:
        raise SystemExit(f"no exported TCN run (deploy_metadata.json) under {runs}; pass --run-dir")
    return found[-1]


def build_controller(ml, run_dir, backend, mass):
    cfg = ml["Config"].load(run_dir / "config.yaml")
    if backend:
        cfg = ml["replace"](cfg, **{"deploy.backend": backend})
    controller = ml["ExoController"](run_dir, cfg.deploy, subject_mass_kg=mass)
    controller.reset()
    return controller, cfg


def build_motor(ml, args):
    import can
    MotorInterface = ml["MotorInterface"]
    kwargs = {"torque_limit_nm": min(args.torque_cap, ml["MIT_T_MAX"]),
              "command_sign": args.command_sign}
    if "slew_nm_per_s" in inspect.signature(MotorInterface.__init__).parameters:
        kwargs["slew_nm_per_s"] = args.slew
    if args.sim:
        from sim_hw import make_sim_motor
        motor = make_sim_motor(MotorInterface, can, **kwargs)
    else:
        motor = MotorInterface(channel=args.can_channel, motor_id=MOTOR_ID, **kwargs)
    takes_dt = "dt" in inspect.signature(motor.send_torque).parameters
    return motor, takes_dt, "slew_nm_per_s" in kwargs


class Monitor:
    """can_monitor.py in its own process."""

    def __init__(self, args, out_npz):
        if args.sim:
            from sim_hw import SIM_CAN_CHANNEL, SIM_CAN_INTERFACE
            iface, chan = SIM_CAN_INTERFACE, SIM_CAN_CHANNEL
        else:
            iface, chan = "socketcan", args.can_channel
        cmd = [sys.executable, str(TEST_DIR / "can_monitor.py"), "--interface", iface,
               "--channel", chan, "--out", str(out_npz), "--parent-pid", str(os.getpid())]
        if args.sim:
            cmd.append("--sim-motor-reply")
        self.out_npz = out_npz
        # own session: a Ctrl-C in the terminal must not stop the monitor before
        # the controller's last frames (zero torque, exit MIT mode) are sent
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, start_new_session=True)
        ready, _, _ = select.select([self.proc.stdout], [], [], 15.0)
        line = self.proc.stdout.readline().strip() if ready else ""
        if line != "READY":
            self.proc.kill()
            rest = self.proc.stdout.read()
            raise SystemExit(f"CAN monitor failed to start: {line}\n{rest}")

    def stop(self):
        time.sleep(0.2)                 # let the last frames' echoes arrive
        self.proc.send_signal(signal.SIGINT)
        try:
            out, _ = self.proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            out, _ = self.proc.communicate()
        return self.proc.returncode, out.strip()


# ── phases ──────────────────────────────────────────────────────────────────

def phase_check(args):
    """Preflight. Exit code 0 only if everything needed for a hardware run works."""
    results = []

    def report(name, ok, detail):
        results.append(ok)
        print(f"  [{'OK' if ok else 'FAIL'}] {name}: {detail}")

    print("Test 4 preflight" + (" (SIM)" if args.sim else ""))
    ml = load_ml(args.ml_root)
    run_dir = resolve_run_dir(args.ml_root, args.run_dir)
    controller, cfg = build_controller(ml, run_dir, args.backend, args.mass)
    w = np.zeros((controller.meta["num_input_channels"], controller.meta["window_length"]),
                 dtype=np.float32)
    ts = []
    for _ in range(200):
        t0 = time.perf_counter()
        controller.backend.predict(w)
        ts.append(time.perf_counter() - t0)
    ts = np.array(ts[20:]) * 1e3
    report("TCN", True, f"{run_dir.name} backend={cfg.deploy.backend} predict "
           f"p50={np.median(ts):.2f} ms p99={np.percentile(ts, 99):.2f} ms")

    if not args.no_gait_inject:
        gait = GaitPlayer(args.gait_csv, args.rate)
        report("gait CSV", True, f"{Path(args.gait_csv).name} ({gait.length_s:.1f} s loop)")

    if args.sim:
        report("hardware", True, "skipped (--sim)")
    else:
        problem = check_can_link(args.can_channel)
        report("CAN link", problem is None, problem or f"{args.can_channel} up at {CAN_BITRATE} bit/s")

        for bus_num, addrs in ((1, [0x48, 0x36]), (args.imu_bus, [0x4A, 0x4B])):
            try:
                missing = probe_i2c(bus_num, addrs)
            except OSError as exc:
                report(f"I2C bus {bus_num}", False, str(exc))
                continue
            for a in addrs:
                report(f"I2C bus {bus_num} 0x{a:02X}", a not in missing,
                       I2C_DEVICES[a] + (" not responding" if a in missing else ""))

        if problem is None:
            motor, _, _ = build_motor(ml, args)
            bus = motor._bus
            replies = {}
            try:
                motor.enter()
                t_end = time.perf_counter() + 0.3                # drain the enter reply
                while bus.recv(timeout=0.05) is not None and time.perf_counter() < t_end:
                    pass
                for _ in range(20):                            # 0 N·m frames at 100 Hz
                    motor.send_torque(0.0)
                    t_end = time.perf_counter() + 0.01
                    while (left := t_end - time.perf_counter()) > 0:
                        msg = bus.recv(timeout=left)
                        if msg is not None:
                            key = f"0x{msg.arbitration_id:X}"
                            replies[key] = replies.get(key, 0) + 1
            except Exception as exc:                           # noqa: BLE001
                report("motor", False, f"{exc!r}")
            else:
                n = sum(replies.values())
                report("motor reply", n >= 15,
                       f"{n} replies to 20 zero-torque frames, ids {replies or 'none'}"
                       + ("" if n >= 15 else " — motor powered? CAN H/L, 120 ohm "
                          "termination, motor id 0x68?"))
            finally:
                motor.shutdown()
            link = can_link(args.can_channel)
            report("CAN bus state", True, link.get("linkinfo", {}).get("info_data", {})
                   .get("state", "?"))

    ok = all(results)
    print("preflight " + ("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


def phase_run(args):
    """idle or load: one timed run into args.out."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    idle = args.phase == "idle"
    rate, dt = args.rate, 1.0 / args.rate
    duration = args.duration
    preroll = args.preroll if args.preroll is not None else (1.0 if idle else 5.0)

    meta = {"phase": args.phase, "label": args.label, "sim": args.sim,
            "args": vars(args), "argv": sys.argv, "pid": os.getpid(),
            "started": datetime.now().isoformat(timespec="seconds"),
            "system": system_info(), "status": "starting"}
    if not args.sim:
        meta["can_link_before"] = can_link(args.can_channel)
    print(f"Test 4 [{args.phase}] {'SIM ' if args.sim else ''}-> {out}")

    if not args.sim and not idle:
        problem = check_can_link(args.can_channel)
        if problem:
            raise SystemExit(problem)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    capacity = int((duration + preroll + 30.0) * rate * 1.5) + 1000
    rec = LoopRecorder(capacity)
    monitor = None if args.no_can_monitor else Monitor(args, out / "can_monitor.npz")

    sensors = controller = motor = tp = flusher = gait = sends = deploy_log = None
    takes_dt = False
    timing = {"predict": 0.0}
    t0 = time.perf_counter()
    wall0 = time.time()
    timed_start = None
    error = None
    try:
        if not idle:
            ml = load_ml(args.ml_root)
            run_dir = resolve_run_dir(args.ml_root, args.run_dir)
            controller, cfg = build_controller(ml, run_dir, args.backend, args.mass)
            meta["tcn"] = {"run_dir": str(run_dir), "backend": cfg.deploy.backend,
                           "ml_src": ml["src"], "window_length": controller.meta["window_length"],
                           "control_rate_hz_config": cfg.deploy.control_rate_hz}

            predict = controller.backend.predict

            def timed_predict(window):
                p0 = time.perf_counter()
                r = predict(window)
                timing["predict"] = time.perf_counter() - p0
                return r

            controller.backend.predict = timed_predict

            if args.sim:
                from sim_hw import SimSensors
                sensors = SimSensors(FRAME_KEYS)
            else:
                if not args.skip_i2c_probe:
                    missing = probe_i2c(1, [0x48, 0x36]) + probe_i2c(args.imu_bus, [0x4A, 0x4B])
                    if missing:
                        raise SystemExit("I2C devices not responding: " + ", ".join(
                            f"0x{a:02X} {I2C_DEVICES[a]}" for a in missing)
                            + " — check wiring/power (or --skip-i2c-probe)")
                sensors = HardwareSensors(args.imu_bus)

            if not args.no_gait_inject:
                gait = GaitPlayer(args.gait_csv, rate)
            motor, takes_dt, slew_on = build_motor(ml, args)
            meta["motor"] = {"torque_mode": args.torque_mode, "torque_cap_nm": motor.torque_limit_nm,
                             "command_sign": args.command_sign, "slew_active": slew_on,
                             "cmd_arb_id": f"0x{CMD_ARB_ID:X}"}
            sends = SendLog(motor._bus, capacity * 2)
            tp = ml["Teleplot"](enabled=not args.no_teleplot)
            if args.logger == "deploy":
                deploy_log = DeployLogger(
                    args.ml_root, out / "deploy_log.csv",
                    {"controller": "Test 4 harness (ML_control TCN)", "test": "fall_test_plan/test_4",
                     "phase": args.phase, "label": args.label, "sim": args.sim})
                meta["deploy_logger"] = {"run_logger_id": deploy_log.run_logger.run_id}
            acfg = controller.adapter.cfg
            enc_neutral = getattr(acfg, "ankle_encoder_neutral_deg", 0.0)
            enc_sign = getattr(acfg, "ankle_encoder_sign", 1.0)
            if args.torque_mode == "computed":
                print(f"!! --torque-mode computed: the motor WILL be driven (cap "
                      f"{motor.torque_limit_nm} N·m). Starting in 3 s, Ctrl-C to abort.")
                time.sleep(3.0)
            motor.enter()

        if args.flush_every > 0:
            flusher = StreamFlusher(out / "loops_stream.csv")

        print(f"  {rate:.0f} Hz target, scheduler={args.scheduler}, pre-roll {preroll:.0f} s, "
              f"timed window {duration:.0f} s"
              + ("" if idle else f", gait injection {'off' if gait is None else 'on'}, "
                 f"torque mode {args.torque_mode}")
              + ("" if idle else f", logger {args.logger}")
              + (f", disk flush every {args.flush_every} s" if flusher else ""))

        t0 = time.perf_counter()
        wall0 = time.time()
        meta["clock_pair"] = [wall0, t0]
        t_stop = float("inf")
        next_deadline = t0
        next_flush = t0 + args.flush_every
        next_print = t0 + 5.0
        prev_start = None
        sent = session_ramp = 0.0
        i = 0
        while not _STOP:
            loop_start = time.perf_counter()
            if loop_start >= t_stop:
                meta["final_loop_start"] = loop_start   # closes the last timed period
                break
            t_now = loop_start - t0
            tick_dt = min(loop_start - prev_start, 0.05) if prev_start else dt
            prev_start = loop_start
            rec.reserve(i)
            c = rec.cols

            if timed_start is None and t_now >= preroll and (
                    idle or controller.buffer.ready):
                timed_start = loop_start
                t_stop = loop_start + duration

            if not idle:
                raw, imu_s, fse_s = sensors.read()
                frame = gait.at(t_now) if gait is not None else raw

                timing["predict"] = 0.0
                s0 = time.perf_counter()
                res = controller.step(frame, dt)
                step_s = time.perf_counter() - s0
                assist = res["command_nm"]

                # safety checks, as in jetson_deploy.py
                session_ramp = min(1.0, t_now / args.arm_ramp)
                target = assist * session_ramp
                work_ms = (time.perf_counter() - loop_start) * 1000.0
                watchdog = work_ms > args.watchdog_ms
                if watchdog:
                    target = 0.0
                cmd = target if args.torque_mode == "computed" else 0.0

                sends.current_loop = i
                n_before, err_before = sends.n, len(sends.errors)
                q0 = time.perf_counter()
                sent = motor.send_torque(cmd, dt=tick_dt) if takes_dt else motor.send_torque(cmd)
                send_s = time.perf_counter() - q0
                n_sent = sends.n - n_before
                n_err = len(sends.errors) - err_before

                l0 = time.perf_counter()
                c["imu_ms"][i] = imu_s * 1e3
                c["fsr_enc_ms"][i] = fse_s * 1e3
                c["step_ms"][i] = step_s * 1e3
                c["predict_ms"][i] = timing["predict"] * 1e3
                c["send_ms"][i] = send_s * 1e3
                c["buffer_ready"][i] = res["buffer_ready"]
                c["stance"][i] = res["stance"]
                c["watchdog"][i] = watchdog
                c["pred_nm_per_kg"][i] = res["predicted_nm_per_kg"]
                c["assist_cmd_nm"][i] = assist
                c["session_ramp"][i] = session_ramp
                c["target_nm"][i] = target
                c["sent_nm"][i] = sent
                c["can_cmd_ok"][i] = n_sent - n_err
                c["can_err"][i] = n_err
                rec.raw[i] = [raw[k] for k in FRAME_KEYS]
                if deploy_log is not None:
                    deploy_log.log(loop_start, t_now, frame, res, args.mass, sent, session_ramp,
                                   (frame["ankle_encoder_deg"] - enc_neutral) * enc_sign,
                                   controller.assistance.scale, work_ms)
            else:
                l0 = time.perf_counter()
            c["seq"][i] = i
            c["timed"][i] = timed_start is not None
            c["t_start"][i] = loop_start
            log_s = time.perf_counter() - l0

            tele_s = 0.0
            if tp is not None:
                e0 = time.perf_counter()
                t_ms = int(time.time() * 1000)
                tp.send("ml_pred_nm_per_kg,ML", res["predicted_nm_per_kg"], t_ms)
                tp.send("assist_cmd_nm,ML", assist, t_ms)
                tp.send("sent_torque_nm,Motor", sent, t_ms)
                tp.send("session_ramp,ML", session_ramp, t_ms)
                tp.send("stance,ML", res["stance"], t_ms)
                tp.send("loop_ms,Perf", (e0 - loop_start) * 1000.0, t_ms)
                tele_s = time.perf_counter() - e0

            flush_s = 0.0
            if flusher is not None and loop_start >= next_flush:
                f0 = time.perf_counter()
                flusher.flush(rec, i)              # rows before this one are complete
                flush_s = time.perf_counter() - f0
                next_flush += args.flush_every

            loop_end = time.perf_counter()
            c["t_end"][i] = loop_end
            c["log_ms"][i] = log_s * 1e3
            c["telemetry_ms"][i] = tele_s * 1e3
            c["flush_ms"][i] = flush_s * 1e3

            if args.scheduler == "relative":       # as jetson_deploy.py / TBE main.py
                sleep = dt - (time.perf_counter() - loop_start)
            else:
                # absolute deadlines: lateness under one period is made up on the
                # next loop; further behind than that, restart the grid (no burst)
                next_deadline += dt
                if loop_end - next_deadline > dt:
                    next_deadline = loop_end
                sleep = next_deadline - time.perf_counter()
            c["sleep_req_ms"][i] = max(sleep, 0.0) * 1e3

            if loop_end >= next_print:
                next_print += 5.0
                span = c["t_start"][max(0, i - 499):i + 1]
                hz = (len(span) - 1) / (span[-1] - span[0]) if len(span) > 1 else 0.0
                state = ("timed %5.0f/%.0f s" % (loop_end - timed_start, duration)
                         if timed_start else "pre-roll")
                print(f"  [{t_now:6.1f} s] {state}  loops={i + 1}  ~{hz:5.1f} Hz (last 500)",
                      flush=True)

            i += 1
            if sleep > 0:
                time.sleep(sleep)
        meta["clock_pair_end"] = [time.time(), time.perf_counter()]
        meta["status"] = ("completed" if timed_start is not None and
                          time.perf_counter() >= t_stop else "interrupted")
    except BaseException as exc:                           # noqa: BLE001
        error = exc
        meta["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "crashed"
        meta["exception"] = repr(exc)
        print(f"\n!! {exc!r} -> zeroing torque, writing logs")
    finally:
        # cleanup (zero torque, exit MIT mode, write logs) must not be cut short;
        # every wait in it is bounded
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: print("  (finishing shutdown and writing logs...)"))
        wall_end = time.time()
        if sends is not None:
            sends.current_loop = -1                        # shutdown frames belong to no loop
        for what, fn in (("motor stop", lambda: motor and motor.stop()),
                         ("motor shutdown", lambda: motor and motor.shutdown()),
                         ("sensors", lambda: sensors and sensors.shutdown()),
                         ("teleplot", lambda: tp and tp.close()),
                         ("deploy log", lambda: deploy_log and deploy_log.close(meta["status"])),
                         ("stream log", lambda: flusher and flusher.close())):
            try:
                fn()
            except Exception as exc:                       # noqa: BLE001
                meta.setdefault("shutdown_errors", []).append(f"{what}: {exc!r}")
        if not args.sim:
            meta["can_link_after"] = can_link(args.can_channel)
        if monitor is not None:
            rc, text = monitor.stop()
            meta["monitor"] = {"returncode": rc, "output": text[-500:],
                               "file": monitor.out_npz.name}

        # the buffered log goes to disk only now, after the timed window
        f0 = time.perf_counter()
        rec.write_csv(out / "loops.csv")
        if sends is not None:
            sends.write_csv(out / "can_sends.csv")
        meta["final_flush_s"] = time.perf_counter() - f0

        meta.update({
            "ended": datetime.now().isoformat(timespec="seconds"),
            "wall_start": wall0, "wall_end": wall_end, "rate_hz": rate,
            "requested_duration_s": duration, "preroll_s": preroll,
            "timed_start_mono": timed_start,
            "loops_recorded": rec.n, "recorder_grow_events": rec.grow_events,
            "can_sends_recorded": sends.n if sends else 0,
            "can_send_errors": (sends.errors[:50] if sends else []),
            "can_send_error_count": len(sends.errors) if sends else 0,
            "oom_check": kernel_oom_check(wall0, wall_end),
        })
        with open(out / "meta.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)
        print(f"  {meta['status']}: {rec.n} loops, "
              f"{meta['can_sends_recorded']} CAN sends -> {out}")
    if error is not None and not isinstance(error, KeyboardInterrupt):
        raise error
    return 0 if meta["status"] == "completed" else 2


def phase_all(args):
    session = Path(args.out) if args.out else (
        TEST_DIR / "results" / datetime.now().strftime("%Y%m%d_%H%M%S"))
    session.mkdir(parents=True, exist_ok=True)
    base = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
    # argparse keeps the last occurrence of an option, so these override the user's
    steps = []
    if not args.skip_check:
        steps.append(("check", ["--phase", "check"]))
    steps.append(("idle", ["--phase", "idle", "--out", str(session / "idle"),
                           "--duration", str(args.idle_duration), "--flush-every", "0"]))
    steps.append(("load", ["--phase", "load", "--out", str(session / "load"),
                           "--duration", str(args.duration), "--flush-every", "0"]))
    if args.flush_run:
        steps.append(("flush", ["--phase", "load", "--label", "flush-marked run",
                                "--out", str(session / "flush"),
                                "--duration", str(args.flush_duration),
                                "--flush-every", str(args.flush_every or 1.0)]))
    for name, extra in steps:
        print(f"\n=== {name} ===", flush=True)
        rc = subprocess.call(base + extra)
        if rc != 0:
            print(f"\n{name} step exited with code {rc}; stopping. Session: {session}")
            if name != "check":
                subprocess.call([sys.executable, str(TEST_DIR / "test4_analyze.py"), str(session)])
            return rc
    print("\n=== analysis ===", flush=True)
    return subprocess.call([sys.executable, str(TEST_DIR / "test4_analyze.py"), str(session)])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["check", "idle", "load", "all"], default="all")
    ap.add_argument("--out", default=None,
                    help="run folder (idle/load) or session folder (all); "
                         "default results/<timestamp>")
    ap.add_argument("--label", default="")
    ap.add_argument("--sim", action="store_true",
                    help="no hardware: simulated sensors + virtual CAN bus (pipeline check only)")

    g = ap.add_argument_group("timing")
    g.add_argument("--rate", type=float, default=100.0, help="control rate, Hz")
    g.add_argument("--duration", type=float, default=600.0,
                   help="timed window, s (load default 600 = the 10-minute run)")
    g.add_argument("--idle-duration", type=float, default=10.0)
    g.add_argument("--preroll", type=float, default=None,
                   help="s before the timed window (default 5 load / 1 idle); the load "
                        "window also waits for the TCN buffer to fill")
    g.add_argument("--scheduler", choices=["relative", "deadline"], default="relative",
                   help="relative = sleep(dt - elapsed) as the current runners do; "
                        "deadline = absolute 10 ms deadlines")
    g.add_argument("--flush-every", type=float, default=0.0,
                   help="s between in-loop disk flushes (0 = buffered, flush after the run)")
    g.add_argument("--flush-run", action="store_true",
                   help="with --phase all: add the marked disk-flush run")
    g.add_argument("--flush-duration", type=float, default=120.0)
    g.add_argument("--skip-check", action="store_true")

    g = ap.add_argument_group("controller")
    g.add_argument("--ml-root", default=str(REPO / "ML_model"),
                   help="ML_model folder whose src/ has the controller (see README)")
    g.add_argument("--run-dir", default=None, help="exported TCN run (default runs/deploy)")
    g.add_argument("--backend", choices=["jit", "onnx", "trt"], default=None,
                   help="default: the run's config.yaml")
    g.add_argument("--mass", type=float, default=72.0)
    g.add_argument("--gait-csv", default=str(DEFAULT_GAIT))
    g.add_argument("--no-gait-inject", action="store_true",
                   help="feed the controller the live sensor values instead of a recorded walk")
    g.add_argument("--arm-ramp", type=float, default=5.0)
    g.add_argument("--watchdog-ms", type=float, default=25.0)
    g.add_argument("--no-teleplot", action="store_true")
    g.add_argument("--logger", choices=["deploy", "buffered"], default="deploy",
                   help="deploy = also do jetson_deploy.py's per-loop logging (CSV row, "
                        "dashboard RunLogger, console line every 50 loops); buffered = RAM only")

    g = ap.add_argument_group("hardware")
    g.add_argument("--imu-bus", type=int, default=1)
    g.add_argument("--skip-i2c-probe", action="store_true")
    g.add_argument("--can-channel", default="can0")
    g.add_argument("--no-can-monitor", action="store_true")
    g.add_argument("--torque-mode", choices=["zero", "computed"], default="zero",
                   help="zero = 0 N·m in every frame (shaft does not move); "
                        "computed = send the controller's torque")
    g.add_argument("--torque-cap", type=float, default=2.0)
    g.add_argument("--command-sign", type=float, default=-1.0)
    g.add_argument("--slew", type=float, default=10.0, help="N·m/s (if MotorInterface supports it)")
    args = ap.parse_args()

    if args.phase == "check":
        return phase_check(args)
    if args.phase == "all":
        return phase_all(args)
    if args.out is None:
        args.out = str(TEST_DIR / "results" /
                       f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.phase}")
    return phase_run(args)


if __name__ == "__main__":
    sys.exit(main())
