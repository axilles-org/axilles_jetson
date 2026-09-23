#!/usr/bin/env python3
"""
ak80_bno_project.py — IMU-driven motor position control (fast)
===============================================================

Reads the BNO085 IMU rotation about the X-axis (roll) and maps it
to position-control the CubeMars AK80-9 motor over the range
-90° to +90°.

Speed optimisations over the naïve approach:
  1. Only enables Game Rotation Vector — no accel / gyro overhead.
  2. Single _process_available_packets() call per loop (fast IMU path
     from sensor_parse.py) instead of 3 separate property reads.
  3. CAN frames are built and sent inline — no per-frame print().
  4. Roll is computed inline (skip pitch / yaw math).
  5. Console output is throttled to ~10 Hz so the hot path never blocks
     on stdout.

──────────────────────────────────────────────────────────────────────────
USAGE:
    python3 ak80_bno_project.py                      # position mode
    python3 ak80_bno_project.py --mode impedance      # impedance mode
    python3 ak80_bno_project.py --kp 50 --kd 1.5      # custom gains
    python3 ak80_bno_project.py --rate 500             # cap at 500 Hz

PREREQUISITES:
    
──────────────────────────────────────────────────────────────────────────
"""

import sys
import os
import math
import struct
import time
import warnings
import argparse

import can

# ── Suppress harmless Blinka warning ─────────────────────────────────────────
warnings.filterwarnings("ignore", category=RuntimeWarning,
                        message="I2C frequency is not settable")

# ── Make sibling directories importable ──────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BNO_DIR    = os.path.join(os.path.dirname(_SCRIPT_DIR),
                           "Sensor_Tests", "IMUs", "BNO085")
sys.path.insert(0, _BNO_DIR)
sys.path.insert(0, _SCRIPT_DIR)

from adafruit_extended_bus import ExtendedI2C as ExtI2C
from adafruit_bno08x.i2c import BNO08X_I2C
from adafruit_bno08x import BNO_REPORT_GAME_ROTATION_VECTOR

from ak_80_test import (
    _send, _pack32, _float_to_uint, _clamp,
    MODE_POSITION, MODE_MIT, MODE_VELOCITY,
    MIT_P_MIN, MIT_P_MAX, MIT_V_MIN, MIT_V_MAX,
    MIT_T_MIN, MIT_T_MAX, MIT_KP_MIN, MIT_KP_MAX, MIT_KD_MIN, MIT_KD_MAX,
    CAN_INTERFACE, MOTOR_ID,
)

# ── Hardware constants ───────────────────────────────────────────────────────
I2C_BUS         = 7
IMU_ADDR        = 0x4A
RECONNECT_DELAY = 1.0
BOOT_DELAY      = 0.8
FEATURE_RETRIES = 5

# ── Motor mapping ────────────────────────────────────────────────────────────
ANGLE_MIN = -90.0   # degrees
ANGLE_MAX =  90.0   # degrees

# Default impedance gains
DEFAULT_KP = 30.0
DEFAULT_KD = 1.0

# Console print interval (seconds) — keeps stdout from bottlenecking the loop
PRINT_INTERVAL = 0.1   # ~10 Hz display updates

# Pre-computed constant
_DEG2RAD = math.pi / 180.0
_RAD2DEG = 180.0 / math.pi


# ── Fast quaternion-only IMU reader ──────────────────────────────────────────
class _QuatIMU:
    """
    Minimal BNO085 reader — enables *only* Game Rotation Vector and uses
    one _process_available_packets() call per read (same trick as
    sensor_parse._FastIMU) to drain all buffered SHTP data in a single
    I2C burst.  Returns the quaternion tuple directly (no dict).
    """

    def __init__(self, bus: int = I2C_BUS, address: int = IMU_ADDR):
        self._bus_num = bus
        self._address = address
        self._i2c = None
        self._bno = None
        self._connect()

    def _connect(self):
        while True:
            try:
                if self._i2c is not None:
                    try:
                        self._i2c.deinit()
                    except Exception:
                        pass
                    time.sleep(0.3)
                self._i2c = ExtI2C(self._bus_num)
                self._bno = BNO08X_I2C(self._i2c, address=self._address)
                time.sleep(BOOT_DELAY)
                for attempt in range(FEATURE_RETRIES):
                    try:
                        self._bno.enable_feature(BNO_REPORT_GAME_ROTATION_VECTOR)
                        break
                    except Exception:
                        if attempt == FEATURE_RETRIES - 1:
                            raise
                        time.sleep(0.2)
                return
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                print(f"[IMU] Connect failed ({exc}), retrying in "
                      f"{RECONNECT_DELAY}s …", flush=True)
                time.sleep(RECONNECT_DELAY)

    def _reconnect(self):
        print("\n[IMU] Reconnecting …", flush=True)
        self._connect()
        print("[IMU] Reconnected.\n", flush=True)

    def read_quat(self):
        """Return (qi, qj, qk, qr) or None.  One I2C burst, no dict."""
        try:
            self._bno._process_available_packets()
            quat = self._bno._readings.get(BNO_REPORT_GAME_ROTATION_VECTOR)
        except (OSError, RuntimeError, AttributeError, KeyError) as exc:
            print(f"\n[IMU] Error ({type(exc).__name__}: {exc}), reconnecting …",
                  flush=True)
            self._reconnect()
            return None
        return quat   # tuple (i, j, k, real) or None if not yet available

    def close(self):
        try:
            self._i2c.deinit()
        except Exception:
            pass


# ── Inline fast roll extraction ──────────────────────────────────────────────
def _quat_to_roll_deg(qi, qj, qk, qr):
    """Quaternion (i,j,k,real) → roll in degrees.  Skips pitch/yaw math."""
    sinr = 2.0 * (qr * qi + qj * qk)
    cosr = 1.0 - 2.0 * (qi * qi + qj * qj)
    return math.atan2(sinr, cosr) * _RAD2DEG


# ── Inline fast CAN senders (no print) ──────────────────────────────────────
def _fast_position(bus, motor_id, degrees):
    """Position-mode CAN frame — no stdout."""
    degrees = max(-36000.0, min(36000.0, degrees))
    _send(bus, MODE_POSITION, motor_id, _pack32(degrees * 10000.0))


def _fast_impedance(bus, motor_id, pos_rad, kp, kd):
    """Impedance CAN frame (vel=0, torque=0) — no stdout."""
    kp_int = _float_to_uint(kp,      MIT_KP_MIN, MIT_KP_MAX, 12)
    kd_int = _float_to_uint(kd,      MIT_KD_MIN, MIT_KD_MAX, 12)
    p_int  = _float_to_uint(pos_rad, MIT_P_MIN,  MIT_P_MAX,  16)
    v_int  = _float_to_uint(0.0,     MIT_V_MIN,  MIT_V_MAX,  12)
    t_int  = _float_to_uint(0.0,     MIT_T_MIN,  MIT_T_MAX,  12)
    buf = bytes([
        kp_int >> 4,
        ((kp_int & 0xF) << 4) | (kd_int >> 8),
        kd_int & 0xFF,
        p_int >> 8,
        p_int & 0xFF,
        v_int >> 4,
        ((v_int & 0xF) << 4) | (t_int >> 8),
        t_int & 0xFF,
    ])
    _send(bus, MODE_MIT, motor_id, buf)


def _fast_stop(bus, motor_id):
    _send(bus, MODE_VELOCITY, motor_id, _pack32(0))


# ── Argument parser ──────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ak80_bno_project.py",
        description="Map BNO085 X-axis (roll) to AK80-9 position [-90°, +90°].",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--mode", choices=["position", "impedance"], default="position",
        help="Motor control strategy (default: position)",
    )
    p.add_argument(
        "--kp", type=float, default=DEFAULT_KP,
        help=f"Position gain for impedance mode (default: {DEFAULT_KP})",
    )
    p.add_argument(
        "--kd", type=float, default=DEFAULT_KD,
        help=f"Damping gain for impedance mode (default: {DEFAULT_KD})",
    )
    p.add_argument(
        "--rate", type=float, default=0,
        help="Max loop rate in Hz.  0 = as fast as possible (default: 0)",
    )
    p.add_argument(
        "--id", type=lambda x: int(x, 0), default=MOTOR_ID,
        help=f"CAN motor ID (default: 0x{MOTOR_ID:02X})",
    )
    return p


# ── Main loop ────────────────────────────────────────────────────────────────

def main():
    args = build_parser().parse_args()
    motor_id = args.id
    min_dt = 1.0 / args.rate if args.rate > 0 else 0.0
    use_impedance = (args.mode == "impedance")
    kp = args.kp
    kd = args.kd

    # ── Open CAN bus ─────────────────────────────────────────────────────────
    try:
        bus = can.interface.Bus(channel=CAN_INTERFACE, interface="socketcan")
    except Exception as e:
        print(f"ERROR: Could not open CAN interface '{CAN_INTERFACE}': {e}")
        print("Make sure you ran:  sudo ip link set can0 up type can bitrate 1000000")
        sys.exit(1)

    # ── Open IMU (quat-only, fast path) ──────────────────────────────────────
    print("Connecting to BNO085 IMU …")
    try:
        imu = _QuatIMU()
    except KeyboardInterrupt:
        bus.shutdown()
        print("\nAborted during IMU init.")
        return

    print(f"IMU connected.  Control mode: {args.mode}")
    if use_impedance:
        print(f"  Kp = {kp}   Kd = {kd}")
    print(f"  Motor range: [{ANGLE_MIN}°, {ANGLE_MAX}°]")
    print(f"  Loop rate limit: {'unlimited' if args.rate == 0 else f'{args.rate} Hz'}")
    print("  Console updates: ~10 Hz  |  Press Ctrl+C to stop.\n")

    print(f"{'Roll(°)':>9} {'Target(°)':>10} {'Hz':>7}")
    print("-" * 30)

    # ── Local-variable aliases (avoid repeated dict / attr lookups) ───────
    _read     = imu.read_quat
    _roll     = _quat_to_roll_deg
    _clampf   = _clamp
    _time     = time.time
    _sleep    = time.sleep
    _amin     = ANGLE_MIN
    _amax     = ANGLE_MAX

    count      = 0
    t_start    = _time()
    t_print    = t_start        # next allowed print time
    last_roll  = 0.0
    last_tgt   = 0.0

    try:
        while True:
            t_loop = _time()

            quat = _read()
            if quat is None:
                continue

            # ── Roll only (inline, skip pitch/yaw) ───────────────────────
            qi, qj, qk, qr = quat
            roll = _roll(qi, qj, qk, qr)

            # ── Clamp to ±90° ────────────────────────────────────────────
            target_deg = _clampf(roll, _amin, _amax)

            # ── Send CAN frame (no print) ────────────────────────────────
            if use_impedance:
                _fast_impedance(bus, motor_id, target_deg * _DEG2RAD, kp, kd)
            else:
                _fast_position(bus, motor_id, target_deg)

            count += 1
            last_roll = roll
            last_tgt  = target_deg

            # ── Throttled console output ─────────────────────────────────
            if t_loop >= t_print:
                elapsed = t_loop - t_start
                hz = count / elapsed if elapsed > 0 else 0
                print(f"\r{last_roll:>9.2f} {last_tgt:>10.2f} {hz:>7.0f}",
                      end="", flush=True)
                t_print = t_loop + PRINT_INTERVAL

            # ── Rate-limit (if requested) ────────────────────────────────
            if min_dt > 0:
                dt = _time() - t_loop
                if dt < min_dt:
                    _sleep(min_dt - dt)

    except KeyboardInterrupt:
        pass
    finally:
        try:
            _fast_stop(bus, motor_id)
        except Exception:
            pass
        imu.close()
        bus.shutdown()

        elapsed = _time() - t_start
        if elapsed > 0 and count > 0:
            print(f"\nStopped.  {count} commands in {elapsed:.1f}s "
                  f"({count / elapsed:.1f} Hz)")
        else:
            print("\nStopped.")


if __name__ == "__main__":
    main()
