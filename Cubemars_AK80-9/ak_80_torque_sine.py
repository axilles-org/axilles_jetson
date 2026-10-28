"""
AK80-9 Sinusoidal Torque Control
=================================
Sends a sinusoidal torque command to the AK80-9 motor via CAN bus
using MIT impedance mode (pure torque: kp=0, kd=0).

Usage:
    sudo ip link set can0 up type can bitrate 1000000
    python3 ak80_sine_torque.py

Press Ctrl+C to stop. Motor will be zeroed on exit.

Adjust AMPLITUDE, FREQUENCY, and OFFSET below to change the wave.
"""

import can
import struct
import time
import math
import sys

# ── Configuration ─────────────────────────────────────────────────────────────

CAN_INTERFACE = "can0"
MOTOR_ID      = 0x68          # Your motor CAN ID

# Sine wave parameters
AMPLITUDE     = 0.5           # Peak torque in Nm (wave goes from -22 to +22)
FREQUENCY     = 2.0           # Hz (one full cycle every 2 seconds)
OFFSET        = 0.0           # Nm (shifts the wave up/down, e.g., 5.0 makes it 0 to 10 Nm)

# Loop rate
CONTROL_FREQ  = 200           # Hz
DT            = 1.0 / CONTROL_FREQ

# MIT mode limits for AK80-9
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

MODE_MIT   = 8

# ── CAN Helpers ───────────────────────────────────────────────────────────────

def _clamp(val, lo, hi):
    return max(lo, min(hi, val))

def _float_to_uint(x, x_min, x_max, bits):
    x = _clamp(x, x_min, x_max)
    span = x_max - x_min
    return int((x - x_min) * ((1 << bits) / span))

def send_torque(bus, torque_nm):
    """Send pure torque command via MIT mode (kp=0, kd=0, pos=0, vel=0)."""

    torque_nm = _clamp(torque_nm, MIT_T_MIN, MIT_T_MAX)

    kp_int = _float_to_uint(0.0,       MIT_KP_MIN, MIT_KP_MAX, 12)
    kd_int = _float_to_uint(0.0,       MIT_KD_MIN, MIT_KD_MAX, 12)
    p_int  = _float_to_uint(0.0,       MIT_P_MIN,  MIT_P_MAX,  16)
    v_int  = _float_to_uint(0.0,       MIT_V_MIN,  MIT_V_MAX,  12)
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

    arb_id = (MODE_MIT << 8) | MOTOR_ID
    msg = can.Message(arbitration_id=arb_id, data=buf, is_extended_id=True)
    bus.send(msg)

def read_feedback(bus):
    """Read one feedback frame from motor. Returns dict or None."""
    msg = bus.recv(timeout=0.01)
    if msg is None or len(msg.data) < 8:
        return None
    
    return {
        "position":    struct.unpack(">h", bytes(msg.data[0:2]))[0] * 0.1,     # degrees
        "speed":       struct.unpack(">h", bytes(msg.data[2:4]))[0] * 10.0,    # ERPM
        "current":     struct.unpack(">h", bytes(msg.data[4:6]))[0] * 0.01,    # Amps
        "temperature": msg.data[6],                                              # °C
        "error":       msg.data[7]
    }

# ── Main ──────────────────────────────────────────────────────────────────────

def main():

    # Open CAN bus
    try:
        bus = can.interface.Bus(channel=CAN_INTERFACE, interface="socketcan")
    except Exception as e:
        print(f"ERROR: Could not open CAN interface '{CAN_INTERFACE}': {e}")
        print("Run:  sudo ip link set can0 up type can bitrate 1000000")
        sys.exit(1)

    print(f"AK80-9 Sinusoidal Torque Control")
    print(f"  Amplitude:  {AMPLITUDE} Nm")
    print(f"  Frequency:  {FREQUENCY} Hz")
    print(f"  Offset:     {OFFSET} Nm")
    print(f"  Loop rate:  {CONTROL_FREQ} Hz")
    print(f"  Torque range: {OFFSET - AMPLITUDE:.1f} to {OFFSET + AMPLITUDE:.1f} Nm")
    print(f"\nPress Ctrl+C to stop.\n")
    print(f"  {'Time':>8}  {'Cmd Torque':>10}  {'Position':>10}  {'Current':>9}  {'Temp':>6}")
    print(f"  " + "-" * 55)

    t_start = time.perf_counter()

    try:
        while True:
            loop_start = time.perf_counter()

            # Compute elapsed time
            t = time.perf_counter() - t_start

            # Generate sinusoidal torque command
            torque_cmd = OFFSET + AMPLITUDE * math.sin(2 * math.pi * FREQUENCY * t)
            print(f"DEBUG: t={t:.3f}s, torque_cmd={torque_cmd:.2f} Nm")
            # Send to motor
            send_torque(bus, torque_cmd)

            # Read feedback
            fb = read_feedback(bus)

            # Print status
            if fb is not None:
                print(f"  {t:>7.2f}s  {torque_cmd:>+9.2f} Nm  {fb['position']:>9.1f}°"
                      f"  {fb['current']:>+7.2f} A  {fb['temperature']:>5}°C")
            else:
                print(f"  {t:>7.2f}s  {torque_cmd:>+9.2f} Nm  {'(no feedback)':>30}")

            # Maintain loop rate
            elapsed = time.perf_counter() - loop_start
            sleep_time = DT - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n\nStopping...")
        # Send zero torque several times to make sure motor gets it
        for _ in range(10):
            send_torque(bus, 0.0)
            time.sleep(0.005)
        print("Motor zeroed.")

    finally:
        bus.shutdown()
        print("CAN bus closed. Done.")


if __name__ == "__main__":
    main()