#!/usr/bin/env python3
"""
can_monitor.py — independent CAN bus + system monitor for Test 4.

Runs as a SEPARATE process next to the control loop, so it adds no work to the
loop being measured. It records:

  * every frame on the CAN interface, with its kernel timestamp. SocketCAN
    loops locally sent frames back to every other socket on the interface, and
    the Jetson's mttcan driver (flag ECHO) does that only once the frame has
    actually left the controller. So the controller's torque commands show up
    here at the time they hit the wire, alongside the motor's replies.
  * once per second: MemAvailable, the control process's RSS, CPU temperature
    and frequency, load average — for the "no out-of-memory event" check.

Everything is kept in RAM and written to one .npz file when the process gets
SIGINT/SIGTERM. Prints "READY" on stdout once it is listening.

Normally started by test4_run.py. Standalone:
    python3 can_monitor.py --out /tmp/mon.npz            # Ctrl-C to stop
"""

import argparse
import os
import signal
import socket
import sys
import time
from pathlib import Path

import numpy as np

import can

CMD_ARB_ID = (8 << 8) | 0x68        # MIT-mode torque command to motor 0x68
SIM_REPLY_ID = (0x29 << 8) | 0x68   # status frame id used by the simulated motor

# setsockopt values to also receive CAN error frames (linux/can/raw.h)
_SOL_CAN_RAW = getattr(socket, "SOL_CAN_RAW", 101)
_CAN_RAW_ERR_FILTER = getattr(socket, "CAN_RAW_ERR_FILTER", 2)
_CAN_ERR_MASK = 0x1FFFFFFF

_STOP = False


def _handle_signal(signum, frame):
    global _STOP
    _STOP = True


class _Grow:
    """Append-only numpy column that doubles when full (never drops data)."""

    def __init__(self, dtype, cap=65536, width=None):
        shape = (cap,) if width is None else (cap, width)
        self.a = np.zeros(shape, dtype=dtype)
        self.n = 0

    def add(self, v):
        if self.n == len(self.a):
            self.a = np.concatenate([self.a, np.zeros_like(self.a)])
        self.a[self.n] = v
        self.n += 1

    def get(self):
        return self.a[: self.n]


def _read_kv(path, keys):
    out = {}
    try:
        with open(path) as f:
            for line in f:
                k, _, v = line.partition(":")
                if k in keys:
                    out[k] = int(v.split()[0])
    except OSError:
        pass
    return out


def _thermal_paths():
    paths = []
    for zone in sorted(Path("/sys/class/thermal").glob("thermal_zone*")):
        try:
            if "cpu" in (zone / "type").read_text().lower():
                paths.append(zone / "temp")
        except OSError:
            pass
    return paths


def _read_int(path, default=-1):
    try:
        return int(Path(path).read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return default


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--interface", default="socketcan")
    ap.add_argument("--channel", default="can0")
    ap.add_argument("--out", required=True, help=".npz output path")
    ap.add_argument("--parent-pid", type=int, default=None,
                    help="process whose RSS is sampled (the control loop)")
    ap.add_argument("--sim-motor-reply", action="store_true",
                    help="simulation only: answer every torque command like the motor would")
    args = ap.parse_args()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    bus = can.Bus(interface=args.interface, channel=args.channel,
                  receive_own_messages=False)
    if args.interface == "socketcan":
        try:
            bus.socket.setsockopt(_SOL_CAN_RAW, _CAN_RAW_ERR_FILTER, _CAN_ERR_MASK)
        except (AttributeError, OSError) as exc:
            print(f"[can_monitor] error frames not enabled: {exc}", file=sys.stderr)

    ts = _Grow(np.float64)          # kernel timestamp (CLOCK_REALTIME, s)
    arb = _Grow(np.uint32)
    flags = _Grow(np.uint8)         # bit0 extended, bit1 is_rx, bit2 error frame
    data = _Grow(np.uint8, width=8)

    s_wall = _Grow(np.float64, cap=4096)
    s_memavail = _Grow(np.int64, cap=4096)      # kB
    s_rss = _Grow(np.int64, cap=4096)           # kB, control process
    s_temp = _Grow(np.int32, cap=4096)          # milli-degC, max over CPU zones
    s_freq_min = _Grow(np.int32, cap=4096)      # kHz, slowest online core
    s_freq_max = _Grow(np.int32, cap=4096)      # kHz, fastest online core
    s_load = _Grow(np.float32, cap=4096)

    def record(msg, is_rx):
        ts.add(msg.timestamp)
        arb.add(msg.arbitration_id)
        flags.add(int(msg.is_extended_id) | (int(is_rx) << 1) | (int(msg.is_error_frame) << 2))
        row = np.zeros(8, dtype=np.uint8)
        row[: min(8, len(msg.data))] = list(msg.data)[:8]
        data.add(row)

    thermal = _thermal_paths()
    freq_paths = sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_cur_freq"))
    status_path = f"/proc/{args.parent_pid}/status" if args.parent_pid else None

    def sample_system():
        s_wall.add(time.time())
        s_memavail.add(_read_kv("/proc/meminfo", {"MemAvailable"}).get("MemAvailable", -1))
        s_rss.add(_read_kv(status_path, {"VmRSS"}).get("VmRSS", -1) if status_path else -1)
        s_temp.add(max((_read_int(p) for p in thermal), default=-1))
        freqs = [f for f in (_read_int(p) for p in freq_paths) if f > 0] or [-1]
        s_freq_min.add(min(freqs))
        s_freq_max.add(max(freqs))
        s_load.add(os.getloadavg()[0])

    clock_pair = (time.time(), time.perf_counter())
    sample_system()
    next_sample = time.monotonic() + 1.0
    print("READY", flush=True)

    recv_err = _Grow(np.float64, cap=1024)      # wall time of each receive error
    last_err = ""

    def recv(timeout):
        nonlocal last_err
        try:
            return bus.recv(timeout=timeout)
        except can.CanError as exc:     # e.g. "Network is down" after a bus-off
            recv_err.add(time.time())
            last_err = str(exc)
            time.sleep(0.05)
            return None

    while not _STOP:
        msg = recv(0.2)
        if msg is not None:
            if args.sim_motor_reply and msg.arbitration_id == SIM_REPLY_ID:
                continue                # our own reply, looped back by udp_multicast
            record(msg, msg.is_rx)
            if (args.sim_motor_reply and msg.arbitration_id == CMD_ARB_ID
                    and msg.is_extended_id):
                reply = can.Message(arbitration_id=SIM_REPLY_ID, is_extended_id=True,
                                    timestamp=time.time(), data=[0, 0, 0, 0, 0, 0, 30, 0])
                bus.send(reply)
                record(reply, True)
        if time.monotonic() >= next_sample:
            sample_system()
            next_sample += 1.0

    # read what is still queued (the controller's last frames) before closing;
    # bounded, in case another node keeps talking
    drain_end = time.monotonic() + 0.5
    while time.monotonic() < drain_end and (msg := recv(0.1)) is not None:
        record(msg, msg.is_rx)
    clock_pair_end = (time.time(), time.perf_counter())
    sample_system()
    bus.shutdown()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out,
             frame_ts=ts.get(), frame_arb=arb.get(), frame_flags=flags.get(),
             frame_data=data.get(),
             sys_wall=s_wall.get(), sys_memavail_kb=s_memavail.get(),
             sys_rss_kb=s_rss.get(), sys_cpu_temp_mc=s_temp.get(),
             sys_cpu_freq_min_khz=s_freq_min.get(), sys_cpu_freq_max_khz=s_freq_max.get(),
             sys_load1=s_load.get(),
             recv_error_wall=recv_err.get(), recv_last_error=np.array(last_err),
             clock_pair=np.array(clock_pair), clock_pair_end=np.array(clock_pair_end),
             interface=np.array(args.interface), channel=np.array(args.channel))
    print(f"[can_monitor] {ts.n} frames, {s_wall.n} system samples, "
          f"{recv_err.n} receive errors {last_err} -> {out}", flush=True)


if __name__ == "__main__":
    main()
