"""
AK80-9 Arrow-Key Torque Control with Trapezoidal Ramp
=======================================================
Uses raw terminal input (termios) — no X server or display required.
Works over SSH and on headless systems.

Up   / Down  arrow → target torque  +1 / -1 Nm
Right / Left arrow → ramp time      +0.1 / -0.1 s

The actual commanded torque ramps linearly to the new target over
_ramp_time seconds, producing a trapezoidal torque profile.

Usage:
    sudo ip link set can0 up type can bitrate 1000000
    python3 ak80_arrow_torque.py

Press 'q' or Ctrl+C to stop. Motor will be zeroed on exit.
"""

import can
import struct
import time
import sys
import math
import threading
import termios
import tty
import os

# ── Configuration ─────────────────────────────────────────────────────────────

CAN_INTERFACE  = "can0"
MOTOR_ID       = 0x68        # Your motor CAN ID

INITIAL_TORQUE = 0.0         # Nm on startup
TORQUE_STEP    = 0.8         # Nm added/subtracted per Up/Down press

INITIAL_RAMP_TIME = 1.0      # seconds on startup
RAMP_TIME_STEP    = 0.1      # seconds added/subtracted per Right/Left press
RAMP_TIME_MIN     = 0.1      # seconds — floor (never instantaneous)
RAMP_TIME_MAX     = 10.0     # seconds — ceiling

CONTROL_FREQ   = 200         # Hz
DT             = 1.0 / CONTROL_FREQ

PRINT_INTERVAL = 0.1         # seconds between status prints (10 Hz)

# MIT mode limits for AK80-9
MIT_P_MIN  = -12.56;  MIT_P_MAX  =  12.56
MIT_V_MIN  = -65.0;   MIT_V_MAX  =  65.0
MIT_T_MIN  = -22.0;   MIT_T_MAX  =  22.0
MIT_KP_MIN =   0.0;   MIT_KP_MAX = 500.0
MIT_KD_MIN =   0.0;   MIT_KD_MAX =   5.0
MODE_MIT   = 8

# ── Shared State ──────────────────────────────────────────────────────────────

_lock           = threading.Lock()
_target_torque  = INITIAL_TORQUE
_current_torque = INITIAL_TORQUE
_ramp_time      = INITIAL_RAMP_TIME   # live — modified by Left/Right keys
_running        = True

def get_target():
    with _lock:
        return _target_torque

def nudge_target(delta):
    with _lock:
        global _target_torque
        _target_torque = max(MIT_T_MIN, min(MIT_T_MAX, _target_torque + delta))
        return _target_torque

def get_ramp_rate():
    """Returns current Nm/s derived from the live ramp time."""
    with _lock:
        return TORQUE_STEP / _ramp_time

def get_ramp_time():
    with _lock:
        return _ramp_time

def nudge_ramp_time(delta):
    with _lock:
        global _ramp_time
        _ramp_time = round(
            max(RAMP_TIME_MIN, min(RAMP_TIME_MAX, _ramp_time + delta)), 2
        )
        return _ramp_time

def stop():
    global _running
    _running = False

# ── Arrow key escape sequences ─────────────────────────────────────────────────

_ARROW_UP    = b'\x1b[A'
_ARROW_DOWN  = b'\x1b[B'
_ARROW_RIGHT = b'\x1b[C'
_ARROW_LEFT  = b'\x1b[D'
_ALL_ARROWS  = (_ARROW_UP, _ARROW_DOWN, _ARROW_RIGHT, _ARROW_LEFT)

# ── Input Thread ──────────────────────────────────────────────────────────────

def input_thread():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    try:
        # Raw mode: keypresses arrive immediately, no echo
        tty.setraw(fd)

        # Re-enable output processing so \n → \r\n still works for print()
        attrs = termios.tcgetattr(fd)
        attrs[1] |= termios.OPOST | termios.ONLCR
        # VMIN=1, VTIME=0: blocking read returns as soon as 1 byte is available
        attrs[6][termios.VMIN]  = 1
        attrs[6][termios.VTIME] = 0
        termios.tcsetattr(fd, termios.TCSANOW, attrs)

        # Debounce: ignore repeated presses of the same arrow key
        # within this window. Prevents auto-repeat from stacking nudges.
        DEBOUNCE_S = 0.20        # seconds — tune up if still bouncing
        last_fired = {}          # key -> time.monotonic() of last accepted press

        while _running:
            # Blocking read — wakes instantly when any byte arrives
            ch = os.read(fd, 1)

            if ch == b'\x1b':
                # Arrow keys send ESC [ A/B/C/D — read the next 2 bytes
                # VTIME=1 (0.1s) so a bare ESC doesn't block forever
                attrs2 = termios.tcgetattr(fd)
                attrs2[6][termios.VMIN]  = 0
                attrs2[6][termios.VTIME] = 1
                termios.tcsetattr(fd, termios.TCSANOW, attrs2)

                ch2 = os.read(fd, 1)
                ch3 = os.read(fd, 1) if ch2 else b''

                # Restore blocking mode
                attrs[6][termios.VMIN]  = 1
                attrs[6][termios.VTIME] = 0
                termios.tcsetattr(fd, termios.TCSANOW, attrs)

                key = ch + ch2 + ch3
            else:
                key = ch

            # ── Debounce ──────────────────────────────────────────────────
            if key in _ALL_ARROWS:
                now = time.monotonic()
                if now - last_fired.get(key, 0.0) < DEBOUNCE_S:
                    continue         # too soon — discard
                last_fired[key] = now
            # ──────────────────────────────────────────────────────────────

            if key == _ARROW_UP:
                new = nudge_target(+TORQUE_STEP)
                print(f"  [UP]    torque target -> {new:+.1f} Nm")

            elif key == _ARROW_DOWN:
                new = nudge_target(-TORQUE_STEP)
                print(f"  [DOWN]  torque target -> {new:+.1f} Nm")

            elif key == _ARROW_RIGHT:
                new_rt = nudge_ramp_time(+RAMP_TIME_STEP)
                print(f"  [RIGHT] ramp time     -> {new_rt:.2f} s  "
                      f"({TORQUE_STEP/new_rt:.2f} Nm/s)")

            elif key == _ARROW_LEFT:
                new_rt = nudge_ramp_time(-RAMP_TIME_STEP)
                print(f"  [LEFT]  ramp time     -> {new_rt:.2f} s  "
                      f"({TORQUE_STEP/new_rt:.2f} Nm/s)")

            elif key in (b'q', b'Q'):
                print("  [q] quit requested.")
                stop()
                break

            elif key in (b'\x03', b'\x04'):   # Ctrl+C or Ctrl+D
                stop()
                break

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

# ── CAN Helpers ───────────────────────────────────────────────────────────────

def _clamp(val, lo, hi):
    return max(lo, min(hi, val))

def _float_to_uint(x, x_min, x_max, bits):
    x = _clamp(x, x_min, x_max)
    return int((x - x_min) / (x_max - x_min) * ((1 << bits) - 1))

def send_torque(bus, torque_nm):
    try:
        torque_nm = _clamp(torque_nm, MIT_T_MIN, MIT_T_MAX)

        kp_int = _float_to_uint(0.0,       MIT_KP_MIN, MIT_KP_MAX, 12)
        kd_int = _float_to_uint(0.0,       MIT_KD_MIN, MIT_KD_MAX, 12)
        p_int  = _float_to_uint(0.0,       MIT_P_MIN,  MIT_P_MAX,  16)
        v_int  = _float_to_uint(0.0,       MIT_V_MIN,  MIT_V_MAX,  12)
        t_int  = _float_to_uint(torque_nm, MIT_T_MIN,  MIT_T_MAX,  12)

        buf    = [0] * 8
        buf[0] =  kp_int >> 4
        buf[1] = ((kp_int & 0xF) << 4) | (kd_int >> 8)
        buf[2] =  kd_int & 0xFF
        buf[3] =  p_int  >> 8
        buf[4] =  p_int  & 0xFF
        buf[5] =  v_int  >> 4
        buf[6] = ((v_int  & 0xF) << 4) | (t_int >> 8)
        buf[7] =  t_int  & 0xFF

        arb_id = (MODE_MIT << 8) | MOTOR_ID
        msg    = can.Message(arbitration_id=arb_id, data=buf, is_extended_id=True)
        bus.send(msg)
        return True

    except (can.CanError, OSError):
        return False

def read_feedback(bus):
    try:
        msg = bus.recv(timeout=0.001)
    except (can.CanError, OSError):
        return None
    if msg is None or len(msg.data) < 8:
        return None
    return {
        "position":    struct.unpack(">h", bytes(msg.data[0:2]))[0] * 0.1,
        "speed":       struct.unpack(">h", bytes(msg.data[2:4]))[0] * 10.0,
        "current":     struct.unpack(">h", bytes(msg.data[4:6]))[0] * 0.01,
        "temperature": msg.data[6],
        "error":       msg.data[7],
    }

# ── Trapezoidal Ramp ──────────────────────────────────────────────────────────

def step_toward_target(current, target, rate, dt):
    max_step = rate * dt
    error    = target - current
    if abs(error) <= max_step:
        return target
    return current + math.copysign(max_step, error)

# ── Control Loop ──────────────────────────────────────────────────────────────

#      time   target  cmd     ramp    pos      cur    tmp
HDR = "  {:>7}  {:>6}  {:>7}  {:>6}  {:>8}  {:>6}  {:>3}".format(
          "Time", "Target", "Cmd(Nm)", "Ramp(s)", "Pos(deg)", "Cur(A)", "Tmp")
SEP = "  " + "-" * 58

def control_loop(bus):
    global _current_torque

    t_start       = time.perf_counter()
    last_print    = 0.0
    can_err_count = 0
    CAN_ERR_LIMIT = 20

    print(HDR)
    print(SEP)

    while _running:
        loop_start = time.perf_counter()
        t          = loop_start - t_start

        target    = get_target()
        ramp_rate = get_ramp_rate()   # re-read every cycle — updates live
        ramp_time = get_ramp_time()

        _current_torque = step_toward_target(_current_torque, target, ramp_rate, DT)

        ok = send_torque(bus, _current_torque)
        if not ok:
            can_err_count += 1
            if can_err_count == 1:
                print(f"  [WARN] CAN send error - retrying...")
            if can_err_count >= CAN_ERR_LIMIT:
                print(f"  [ERROR] CAN down for {CAN_ERR_LIMIT} cycles. "
                      f"Run: sudo ip link set {CAN_INTERFACE} up type can bitrate 1000000")
                stop()
                break
        else:
            if can_err_count > 0:
                print(f"  [OK] CAN recovered after {can_err_count} errors.")
            can_err_count = 0

        fb = read_feedback(bus)

        if t - last_print >= PRINT_INTERVAL:
            last_print = t
            if fb is not None:
                print("  {:>6.2f}s  {:>+6.2f}  {:>+7.3f}  {:>6.2f}  {:>8.1f}  {:>+6.2f}  {:>3}".format(
                    t, target, _current_torque, ramp_time,
                    fb["position"], fb["current"], fb["temperature"]))
            else:
                print("  {:>6.2f}s  {:>+6.2f}  {:>+7.3f}  {:>6.2f}  {:>8}  {:>6}  {:>3}".format(
                    t, target, _current_torque, ramp_time, "---", "---", "---"))

        elapsed = time.perf_counter() - loop_start
        sleep_t = DT - elapsed
        if sleep_t > 0:
            time.sleep(sleep_t)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global _running

    try:
        bus = can.interface.Bus(channel=CAN_INTERFACE, interface="socketcan")
    except Exception as e:
        print(f"ERROR: Could not open CAN interface '{CAN_INTERFACE}': {e}")
        print(f"Run:  sudo ip link set {CAN_INTERFACE} up type can bitrate 1000000")
        sys.exit(1)

    print("AK80-9 Arrow-Key Torque Control")
    print(f"  [UP]    / [DOWN]  : torque  +/- {TORQUE_STEP} Nm")
    print(f"  [RIGHT] / [LEFT]  : ramp time +/- {RAMP_TIME_STEP} s  "
          f"(range {RAMP_TIME_MIN}–{RAMP_TIME_MAX} s)")
    print(f"  Torque limits : {MIT_T_MIN} to {MIT_T_MAX} Nm")
    print(f"  Initial ramp  : {INITIAL_RAMP_TIME} s  "
          f"({TORQUE_STEP/INITIAL_RAMP_TIME:.2f} Nm/s)")
    print(f"  'q' or Ctrl+C to quit\n")

    kb_thread = threading.Thread(target=input_thread, daemon=True)
    kb_thread.start()
    time.sleep(0.05)   # let keyboard thread set raw mode before we start printing

    try:
        control_loop(bus)

    except KeyboardInterrupt:
        print("\nCtrl+C received -- stopping.")

    finally:
        _running = False
        kb_thread.join(timeout=0.5)

        print("Zeroing motor...")
        for _ in range(10):
            try:
                send_torque(bus, 0.0)
            except Exception:
                pass
            time.sleep(0.005)
        print("Motor zeroed.")

        try:
            bus.shutdown()
        except Exception:
            pass
        print("CAN bus closed. Done.")


if __name__ == "__main__":
    main()