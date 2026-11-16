"""
CubeMars AK80 Motor Control - CLI Interface
============================================

Usage:
  python3 ak_80_test.py <mode> [options]

Modes:
  zero        Set current position as zero
  pos         Move to a position (degrees) at max speed
  pos_vel     Move to a position at a controlled speed
  vel         Spin at a target velocity (ERPM)
  impedance   Impedance/force control (MIT mode)
  brake       Hold current position with a braking current
  stop        Stop the motor (coast to zero RPM)

Run with --help on any mode to see its options:
  python3 ak_80_test.py pos --help
"""

import can
import struct
import time
import argparse
import sys

# ──────────────────────────────────────────────────────────────────
# CONFIGURATION — edit these if needed
# ──────────────────────────────────────────────────────────────────

CAN_INTERFACE = "can0"
MOTOR_ID      = 0x68   # 104 decimal. Change if your motor ID differs.

# MIT mode limits for AK80-9
#   Motor   | Max speed (rad/s) | Max torque (Nm)
#   AK80-9  |       65.0        |      18.0
#   AK10-9  |       28.0        |      54.0
#   AK60-6  |       60.0        |      12.0
#   AK70-9  |       30.0        |      32.0
#   AKE60-8 |       40.0        |      15.0
#   AKE80-8 |       20.0        |      35.0
MIT_P_MIN  = -12.56
MIT_P_MAX  =  12.56
MIT_V_MIN  = -65.0
MIT_V_MAX  =  65.0
MIT_T_MIN  = -22.0
MIT_T_MAX  =  22.0
MIT_KP_MIN =   0.0
MIT_KP_MAX = 500.0
MIT_KD_MIN =   0.0
MIT_KD_MAX =   5.0

# CAN mode IDs
MODE_POSITION = 4
MODE_VELOCITY = 3
MODE_BRAKE    = 2
MODE_ORIGIN   = 5
MODE_POS_VEL  = 6
MODE_MIT      = 8


# ──────────────────────────────────────────────────────────────────
# LOW-LEVEL HELPERS
# ──────────────────────────────────────────────────────────────────

def _send(bus, control_mode, motor_id, data):
    arb_id = (control_mode << 8) | motor_id
    msg = can.Message(arbitration_id=arb_id, data=data, is_extended_id=True)
    bus.send(msg)

def _pack32(value):
    return list(struct.pack(">i", int(value)))

def _pack16(value):
    return list(struct.pack(">h", int(value)))

def _clamp(val, lo, hi):
    return max(lo, min(hi, val))

def _float_to_uint(x, x_min, x_max, bits):
    x = _clamp(x, x_min, x_max)
    span = x_max - x_min
    return int((x - x_min) * (((1 << bits) - 1) / span))


# ──────────────────────────────────────────────────────────────────
# MOTOR COMMANDS
# ──────────────────────────────────────────────────────────────────

def cmd_zero(bus, motor_id, permanent):
    _send(bus, MODE_ORIGIN, motor_id, [1 if permanent else 0])
    print(f"[zero] Current position set as zero.  permanent={permanent}")

def cmd_stop(bus, motor_id):
    _send(bus, MODE_VELOCITY, motor_id, _pack32(0))
    print("[stop] Motor stopped.")

def cmd_brake(bus, motor_id, current_a):
    _send(bus, MODE_BRAKE, motor_id, _pack32(current_a * 1000.0))
    print(f"[brake] Holding with {current_a} A braking current.")

def cmd_position(bus, motor_id, degrees):
    degrees = _clamp(degrees, -36000, 36000)
    _send(bus, MODE_POSITION, motor_id, _pack32(degrees * 10000.0))
    print(f"[pos] Moving to {degrees}°")

def cmd_position_velocity(bus, motor_id, degrees, speed_erpm, accel_erpm_s2):
    degrees = _clamp(degrees, -36000, 36000)
    data = (
        _pack32(degrees * 10000.0) +
        _pack16(speed_erpm / 10.0) +
        _pack16(accel_erpm_s2 / 10.0)
    )
    _send(bus, MODE_POS_VEL, motor_id, data)
    print(f"[pos_vel] Moving to {degrees}°  speed={speed_erpm} ERPM  accel={accel_erpm_s2} ERPM/s²")

def cmd_velocity(bus, motor_id, erpm):
    _send(bus, MODE_VELOCITY, motor_id, _pack32(erpm))
    print(f"[vel] Spinning at {erpm} ERPM")

def cmd_impedance(bus, motor_id, pos_rad, vel_rad_s, torque_nm, kp, kd):
    kp_int = _float_to_uint(kp,        MIT_KP_MIN, MIT_KP_MAX, 12)
    kd_int = _float_to_uint(kd,        MIT_KD_MIN, MIT_KD_MAX, 12)
    p_int  = _float_to_uint(pos_rad,   MIT_P_MIN,  MIT_P_MAX,  16)
    v_int  = _float_to_uint(vel_rad_s, MIT_V_MIN,  MIT_V_MAX,  12)
    t_int  = _float_to_uint(torque_nm, MIT_T_MIN,  MIT_T_MAX,  12)

    buf = [0] * 8
    buf[0] =  kp_int >> 4
    buf[1] = ((kp_int & 0xF) << 4) | (kd_int >> 8)
    buf[2] =  kd_int & 0xFF
    buf[3] =  p_int  >> 8
    buf[4] =  p_int  & 0xFF
    buf[5] =  v_int  >> 4
    buf[6] = ((v_int  & 0xF) << 4) | (t_int >> 8)
    buf[7] =  t_int  & 0xFF

    _send(bus, MODE_MIT, motor_id, buf)
    print(f"[impedance] pos={pos_rad} rad  vel={vel_rad_s} rad/s  "
          f"torque={torque_nm} Nm  kp={kp}  kd={kd}")

def read_feedback(bus, duration=2.0):
    print(f"\n[feedback] Listening for {duration}s ...\n"
          f"  {'position':>10}  {'speed':>10}  {'current':>9}  {'temp':>6}  {'error':>5}")
    print("  " + "-" * 55)
    t_end = time.time() + duration
    while time.time() < t_end:
        msg = bus.recv(timeout=0.05)
        if msg is None or len(msg.data) < 8:
            continue
        pos = struct.unpack(">h", bytes(msg.data[0:2]))[0] * 0.1
        spd = struct.unpack(">h", bytes(msg.data[2:4]))[0] * 10.0
        cur = struct.unpack(">h", bytes(msg.data[4:6]))[0] * 0.01
        tmp = msg.data[6]
        err = msg.data[7]
        print(f"  {pos:>9.1f}°  {spd:>9.0f} ERPM  {cur:>7.2f} A  {tmp:>5}°C  {err:>5}")


# ──────────────────────────────────────────────────────────────────
# ARGUMENT PARSER
# ──────────────────────────────────────────────────────────────────

def build_parser():
    parser = argparse.ArgumentParser(
        prog="ak_80_test.py",
        description="CubeMars AK80 motor control over CAN bus.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES
--------
  Set zero (temporary):
    python3 ak_80_test.py zero

  Set zero (saved to flash):
    python3 ak_80_test.py zero --permanent

  Move to 180 degrees at max speed:
    python3 ak_80_test.py pos --degrees 180

  Move to 90 degrees at controlled speed:
    python3 ak_80_test.py pos_vel --degrees 90 --speed 3000 --accel 15000

  Spin at 5000 ERPM:
    python3 ak_80_test.py vel --erpm 5000

  Spin backwards:
    python3 ak_80_test.py vel --erpm -5000

  Stop the motor:
    python3 ak_80_test.py stop

  Hold position with 3A brake:
    python3 ak_80_test.py brake --current 3.0

  Stiff position hold at 0 rad (kp=100, kd=2):
    python3 ak_80_test.py impedance --pos 0.0 --kp 100 --kd 2

  Compliant position hold at 1.57 rad (soft spring):
    python3 ak_80_test.py impedance --pos 1.57 --kp 20 --kd 0.5

  Pure velocity in MIT mode:
    python3 ak_80_test.py impedance --vel 3.0 --kd 2.0

  Pure torque (e.g. gravity compensation):
    python3 ak_80_test.py impedance --torque 1.5

  Fully backdriveable (zero torque, free to move):
    python3 ak_80_test.py impedance

  Use a different motor ID (e.g. 0x01):
    python3 ak_80_test.py pos --degrees 90 --id 1

  Read motor feedback for 3 seconds after a command:
    python3 ak_80_test.py vel --erpm 2000 --feedback 3
        """
    )

    sub = parser.add_subparsers(dest="mode", metavar="<mode>")
    sub.required = True

    # ── Shared optional args added to every subcommand ─────────────
    def add_common(p):
        p.add_argument("--id",       type=lambda x: int(x, 0), default=MOTOR_ID,
                       metavar="ID",
                       help=f"CAN motor ID in decimal or hex (default: {MOTOR_ID} = 0x{MOTOR_ID:02X})")
        p.add_argument("--feedback", type=float, default=0.0,
                       metavar="SECS",
                       help="After sending the command, listen and print motor feedback for N seconds")

    # ── zero ───────────────────────────────────────────────────────
    p_zero = sub.add_parser("zero", help="Set current position as zero")
    p_zero.add_argument("--permanent", action="store_true",
                        help="Save zero to flash (persists after power off). "
                             "Default: temporary (lost on power off)")
    add_common(p_zero)

    # ── stop ───────────────────────────────────────────────────────
    p_stop = sub.add_parser("stop", help="Stop the motor (coast to 0 RPM)")
    add_common(p_stop)

    # ── brake ──────────────────────────────────────────────────────
    p_brake = sub.add_parser("brake", help="Hold position using a braking current")
    p_brake.add_argument("--current", type=float, default=3.0,
                         metavar="AMPS",
                         help="Braking current in Amps (default: 3.0). "
                              "Higher = stiffer hold but more heat. Suggested: 1-10")
    add_common(p_brake)

    # ── pos ────────────────────────────────────────────────────────
    p_pos = sub.add_parser("pos", help="Move to absolute position at max speed")
    p_pos.add_argument("--degrees", type=float, required=True,
                       metavar="DEG",
                       help="Target position in degrees. Range: -36000 to +36000 "
                            "(i.e. ±100 full rotations from zero)")
    add_common(p_pos)

    # ── pos_vel ────────────────────────────────────────────────────
    p_pv = sub.add_parser("pos_vel",
                          help="Move to position at a controlled speed and acceleration")
    p_pv.add_argument("--degrees", type=float, required=True,
                      metavar="DEG",
                      help="Target position in degrees (-36000 to +36000)")
    p_pv.add_argument("--speed",   type=float, default=5000,
                      metavar="ERPM",
                      help="Max speed in electrical RPM (default: 5000). "
                           "AK80-9 has 14 pole pairs: 5000 ERPM ≈ 357 mech. RPM")
    p_pv.add_argument("--accel",   type=float, default=30000,
                      metavar="ERPM/S2",
                      help="Acceleration in ERPM/s² (default: 30000). "
                           "Lower = smoother ramp up")
    add_common(p_pv)

    # ── vel ────────────────────────────────────────────────────────
    p_vel = sub.add_parser("vel", help="Spin at a constant electrical RPM")
    p_vel.add_argument("--erpm", type=float, required=True,
                       metavar="ERPM",
                       help="Target electrical RPM. Positive or negative for direction. "
                            "0 = stop.  AK80-9: ~65000 ERPM max")
    add_common(p_vel)

    # ── impedance ──────────────────────────────────────────────────
    p_imp = sub.add_parser("impedance",
                           help="MIT impedance control: blend position, velocity, torque")
    p_imp.add_argument("--pos",    type=float, default=0.0,
                       metavar="RAD",
                       help="Desired position in radians (default: 0.0). "
                            "Range: -12.56 to +12.56  (~±2 full turns)")
    p_imp.add_argument("--vel",    type=float, default=0.0,
                       metavar="RAD/S",
                       help="Desired velocity in rad/s (default: 0.0)")
    p_imp.add_argument("--torque", type=float, default=0.0,
                       metavar="NM",
                       help="Feedforward torque in Nm (default: 0.0). "
                            "AK80-9 max: ±18 Nm")
    p_imp.add_argument("--kp",     type=float, default=0.0,
                       metavar="KP",
                       help="Position gain / spring stiffness (default: 0.0). "
                            "Range: 0-500.  Higher = stiffer")
    p_imp.add_argument("--kd",     type=float, default=0.0,
                       metavar="KD",
                       help="Damping gain (default: 0.0). "
                            "Range: 0-5.  Higher = more damping")
    add_common(p_imp)

    return parser


# ──────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────

def main():
    parser = build_parser()
    args   = parser.parse_args()

    try:
        bus = can.interface.Bus(channel=CAN_INTERFACE, interface="socketcan")
    except Exception as e:
        print(f"ERROR: Could not open CAN interface '{CAN_INTERFACE}': {e}")
        print("Make sure you ran:  sudo ip link set can0 up type can bitrate 1000000")
        sys.exit(1)

    try:
        motor_id = args.id

        if args.mode == "zero":
            cmd_zero(bus, motor_id, args.permanent)

        elif args.mode == "stop":
            cmd_stop(bus, motor_id)

        elif args.mode == "brake":
            cmd_brake(bus, motor_id, args.current)

        elif args.mode == "pos":
            cmd_position(bus, motor_id, args.degrees)

        elif args.mode == "pos_vel":
            cmd_position_velocity(bus, motor_id, args.degrees, args.speed, args.accel)

        elif args.mode == "vel":
            cmd_velocity(bus, motor_id, args.erpm)

        elif args.mode == "impedance":
            cmd_impedance(bus, motor_id,
                          args.pos, args.vel, args.torque,
                          args.kp,  args.kd)

        # Optional: read feedback after any command
        if args.feedback > 0:
            read_feedback(bus, duration=args.feedback)

    finally:
        bus.shutdown()


if __name__ == "__main__":
    main()