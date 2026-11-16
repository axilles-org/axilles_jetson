import can
import struct
import time
import math
import sys

# ---------------- Configuration ----------------

CAN_INTERFACE = "can0"
MOTOR_ID = 0x68
MODE_MIT = 8

CONTROL_FREQ = 200.0
DT = 1.0 / CONTROL_FREQ

# Position bounds (deg) - strict
POS_MIN_DEG = -15.0
POS_MAX_DEG = +15.0
SWITCH_BAND_DEG = 0.4      # switch target when close to endpoint
HARD_GUARD_DEG = 0.05      # absolute guard margin

# Trapezoid torque profile params
TORQUE_PEAK = 0.5          # Nm
DECEL_ZONE_DEG = 10.0       # start ramp-down this far before target
RAMP_RATE = 100.0           # Nm/s slew limit (trapezoid edges)

# Safety braking
LIMIT_BRAKE_TORQUE = 0.5   # Nm opposite torque at/beyond limit
NO_FB_TIMEOUT = 0.10       # s -> zero torque if feedback lost

# MIT mode limits for AK80-9
MIT_P_MIN, MIT_P_MAX = -12.56, 12.56
MIT_V_MIN, MIT_V_MAX = -65.0, 65.0
MIT_T_MIN, MIT_T_MAX = -22.0, 22.0
MIT_KP_MIN, MIT_KP_MAX = 0.0, 500.0
MIT_KD_MIN, MIT_KD_MAX = 0.0, 5.0

# ---------------- CAN Helpers ----------------

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))

def _float_to_uint(x, x_min, x_max, bits):
    x = _clamp(x, x_min, x_max)
    return int((x - x_min) * (((1 << bits) - 1) / (x_max - x_min)))

def send_torque(bus, torque_nm):
    torque_nm = _clamp(torque_nm, MIT_T_MIN, MIT_T_MAX)

    kp_int = _float_to_uint(0.0,       MIT_KP_MIN, MIT_KP_MAX, 12)
    kd_int = _float_to_uint(0.0,       MIT_KD_MIN, MIT_KD_MAX, 12)
    p_int  = _float_to_uint(0.0,       MIT_P_MIN,  MIT_P_MAX,  16)
    v_int  = _float_to_uint(0.0,       MIT_V_MIN,  MIT_V_MAX,  12)
    t_int  = _float_to_uint(torque_nm, MIT_T_MIN,  MIT_T_MAX,  12)

    buf = [0] * 8
    buf[0] = kp_int >> 4
    buf[1] = ((kp_int & 0xF) << 4) | (kd_int >> 8)
    buf[2] = kd_int & 0xFF
    buf[3] = p_int >> 8
    buf[4] = p_int & 0xFF
    buf[5] = v_int >> 4
    buf[6] = ((v_int & 0xF) << 4) | (t_int >> 8)
    buf[7] = t_int & 0xFF

    arb_id = (MODE_MIT << 8) | MOTOR_ID
    msg = can.Message(arbitration_id=arb_id, data=buf, is_extended_id=True)
    bus.send(msg)

def read_feedback(bus):
    msg = bus.recv(timeout=0.001)
    if msg is None or len(msg.data) < 8:
        return None
    return {
        "position":    struct.unpack(">h", bytes(msg.data[0:2]))[0] * 0.1,   # deg
        "speed":       struct.unpack(">h", bytes(msg.data[2:4]))[0] * 10.0,  # ERPM (sign useful)
        "current":     struct.unpack(">h", bytes(msg.data[4:6]))[0] * 0.01,  # A
        "temperature": msg.data[6],
        "error":       msg.data[7],
    }

def slew_to(current, target, max_step):
    if target > current + max_step:
        return current + max_step
    if target < current - max_step:
        return current - max_step
    return target

# ---------------- Main ----------------

def main():
    try:
        bus = can.interface.Bus(channel=CAN_INTERFACE, interface="socketcan")
    except Exception as e:
        print(f"ERROR: Could not open CAN interface '{CAN_INTERFACE}': {e}")
        print("Run: sudo ip link set can0 up type can bitrate 1000000")
        sys.exit(1)

    print("AK80-9 trapezoidal torque profile with hard +/-15 deg limits")
    print("Press Ctrl+C to stop.\n")
    print(f"{'t[s]':>7} {'pos[deg]':>10} {'cmd[Nm]':>9} {'target':>8} {'cur[A]':>8}")

    target = POS_MAX_DEG
    cmd_torque = 0.0

    last_pos = 0.0
    last_fb_t = time.perf_counter()
    t0 = time.perf_counter()

    try:
        while True:
            loop_t = time.perf_counter()
            fb = read_feedback(bus)

            if fb is not None:
                pos = fb["position"]
                last_pos = pos
                last_fb_t = loop_t
            else:
                pos = last_pos

            # Fail-safe: no feedback -> stop commanding torque
            if (loop_t - last_fb_t) > NO_FB_TIMEOUT:
                cmd_torque = 0.0
                send_torque(bus, 0.0)
                time.sleep(DT)
                continue

            # Flip direction near target endpoint
            if target == POS_MAX_DEG and pos >= (POS_MAX_DEG - SWITCH_BAND_DEG):
                target = POS_MIN_DEG
            elif target == POS_MIN_DEG and pos <= (POS_MIN_DEG + SWITCH_BAND_DEG):
                target = POS_MAX_DEG

            # Desired torque (trapezoid-like): ramp up/down with distance-to-target
            direction = 1.0 if target > pos else -1.0
            dist = abs(target - pos)

            if dist >= DECEL_ZONE_DEG:
                desired_abs = TORQUE_PEAK
            else:
                desired_abs = TORQUE_PEAK * (dist / DECEL_ZONE_DEG)  # linear ramp-down to 0

            desired_torque = direction * desired_abs

            # Slew-rate limit => trapezoid edges in time
            cmd_torque = slew_to(cmd_torque, desired_torque, RAMP_RATE * DT)

            # -------- Hard safety guard (do not push beyond +/-15 deg) --------
            if pos >= (POS_MAX_DEG - HARD_GUARD_DEG):
                # Never allow positive torque at/near upper limit
                cmd_torque = min(cmd_torque, 0.0)
                if pos > POS_MAX_DEG:
                    cmd_torque = -abs(LIMIT_BRAKE_TORQUE)

            if pos <= (POS_MIN_DEG + HARD_GUARD_DEG):
                # Never allow negative torque at/near lower limit
                cmd_torque = max(cmd_torque, 0.0)
                if pos < POS_MIN_DEG:
                    cmd_torque = abs(LIMIT_BRAKE_TORQUE)

            cmd_torque = _clamp(cmd_torque, MIT_T_MIN, MIT_T_MAX)
            send_torque(bus, cmd_torque)

            t = loop_t - t0
            cur = fb["current"] if fb else 0.0
            print(f"{t:7.2f} {pos:10.3f} {cmd_torque:9.3f} {target:8.1f} {cur:8.3f}")

            elapsed = time.perf_counter() - loop_t
            sleep_t = DT - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    except KeyboardInterrupt:
        print("\nStopping...")
        for _ in range(10):
            send_torque(bus, 0.0)
            time.sleep(0.005)
        print("Motor zeroed.")
    finally:
        bus.shutdown()
        print("CAN bus closed.")

if __name__ == "__main__":
    main()