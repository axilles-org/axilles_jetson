"""
AK80-9 Diagnostic + Calibration Script
=======================================
Run this first to diagnose the vibrating/stopping issue.

Usage:
  python3 ak_80_diag.py dump          # just listen to raw CAN frames
  python3 ak_80_diag.py status        # request and decode motor status
  python3 ak_80_diag.py calibrate     # run motor + encoder identification
  python3 ak_80_diag.py test_duty     # send tiny duty cycle (safest movement test)

IMPORTANT: Remove all load from the shaft before running 'calibrate'
"""

import can
import struct
import time
import argparse
import sys

CAN_INTERFACE = "can0"
MOTOR_ID      = 0x68

# ── Correct limits for AK80-9 (fixes the previous script) ─────────
MIT_P_MIN  = -12.56
MIT_P_MAX  =  12.56
MIT_V_MIN  = -65.0    # AK80-9 specific
MIT_V_MAX  =  65.0
MIT_T_MIN  = -22.0    # AK80-9 specific  (NOT 54!)
MIT_T_MAX  =  22.0
MIT_KP_MIN =   0.0
MIT_KP_MAX = 500.0
MIT_KD_MIN =   0.0
MIT_KD_MAX =   5.0

# Serial protocol constants
COMM_GET_VALUES   = 69
COMM_SET_DUTY     = 70
COMM_SET_RPM      = 73

# ──────────────────────────────────────────────────────────────────
# CAN HELPERS
# ──────────────────────────────────────────────────────────────────

def _send_can(bus, control_mode, motor_id, data):
    arb_id = (control_mode << 8) | motor_id
    msg = can.Message(arbitration_id=arb_id, data=data, is_extended_id=True)
    bus.send(msg)
    print(f"  TX: arb_id=0x{arb_id:08X}  data={bytes(data).hex(' ').upper()}")

def _pack32(value):
    return list(struct.pack(">i", int(value)))

def _pack16(value):
    return list(struct.pack(">h", int(value)))

def open_bus():
    try:
        bus = can.interface.Bus(channel=CAN_INTERFACE, interface="socketcan")
        print(f"CAN bus opened on {CAN_INTERFACE}\n")
        return bus
    except Exception as e:
        print(f"ERROR opening CAN bus: {e}")
        print("Run:  sudo ip link set can0 up type can bitrate 1000000")
        sys.exit(1)


# ──────────────────────────────────────────────────────────────────
# DIAGNOSTICS
# ──────────────────────────────────────────────────────────────────

ERROR_CODES = {
    0: "No fault",
    1: "Motor over-temperature",
    2: "Over-current",
    3: "Over-voltage",
    4: "Under-voltage",
    5: "Encoder fault",
    6: "MOSFET over-temperature",
    7: "Motor lock-up",
}

def decode_feedback(data):
    """Decode the standard 8-byte periodic CAN feedback frame."""
    if len(data) < 8:
        return None
    pos = struct.unpack(">h", bytes(data[0:2]))[0] * 0.1
    spd = struct.unpack(">h", bytes(data[2:4]))[0] * 10.0
    cur = struct.unpack(">h", bytes(data[4:6]))[0] * 0.01
    tmp = data[6]
    err = data[7]
    return {
        "position_deg" : pos,
        "speed_erpm"   : spd,
        "current_a"    : cur,
        "temperature_c": tmp,
        "error_code"   : err,
        "error_str"    : ERROR_CODES.get(err, f"Unknown error {err}"),
    }


def cmd_dump(duration=10.0):
    """
    Just listen and print every raw CAN frame.
    Use this to confirm the motor is alive and sending data.
    """
    bus = open_bus()
    print(f"Listening for CAN frames for {duration}s  (Ctrl+C to stop early)\n")
    print(f"  {'time':>8}  {'arb_id':>12}  {'len':>4}  data")
    print("  " + "-" * 60)
    t_end = time.time() + duration
    count = 0
    try:
        while time.time() < t_end:
            msg = bus.recv(timeout=0.1)
            if msg is None:
                continue
            count += 1
            data_hex = bytes(msg.data).hex(' ').upper()
            print(f"  {msg.timestamp:>8.3f}  "
                  f"0x{msg.arbitration_id:08X}  "
                  f"{msg.dlc:>4}  {data_hex}")

            # Try to decode as feedback
            fb = decode_feedback(list(msg.data))
            if fb:
                print(f"           → pos={fb['position_deg']:.1f}°  "
                      f"spd={fb['speed_erpm']:.0f} ERPM  "
                      f"cur={fb['current_a']:.2f}A  "
                      f"temp={fb['temperature_c']}°C  "
                      f"ERR: {fb['error_str']}")
    except KeyboardInterrupt:
        pass
    finally:
        bus.shutdown()
    print(f"\nTotal frames received: {count}")
    if count == 0:
        print("\n*** NO FRAMES RECEIVED ***")
        print("Possible causes:")
        print("  1. Motor not powered on")
        print("  2. CAN_H / CAN_L wires swapped")
        print("  3. Missing 120-ohm termination resistor")
        print("  4. Wrong CAN interface (try 'candump can1')")


def cmd_status():
    """
    Send a 'get values' request and decode the full motor status response.
    This uses the serial-over-CAN protocol.
    """
    bus = open_bus()

    # Request motor status (serial command 0x45 = 69 = COMM_GET_VALUES)
    # Frame: AA 01 45 <crc_hi> <crc_lo> BB
    # We send via CAN duty=0 first to wake it, then listen for any response
    print("Sending zero-duty to wake motor and listening for status frames...\n")

    # Send a zero duty cycle — this is the safest possible command
    # control mode 0 = duty cycle, value 0 = no movement
    _send_can(bus, 0, MOTOR_ID, _pack32(0))

    print("\nListening for 3 seconds...\n")
    t_end = time.time() + 3.0
    found = False
    while time.time() < t_end:
        msg = bus.recv(timeout=0.1)
        if msg is None:
            continue
        fb = decode_feedback(list(msg.data))
        if fb:
            found = True
            print("Motor status:")
            print(f"  Position    : {fb['position_deg']:.2f} degrees")
            print(f"  Speed       : {fb['speed_erpm']:.0f} ERPM")
            print(f"  Current     : {fb['current_a']:.3f} A")
            print(f"  Temperature : {fb['temperature_c']} °C")
            print(f"  Error       : {fb['error_str']}")
            if fb['error_code'] != 0:
                print(f"\n  *** FAULT ACTIVE: {fb['error_str']} ***")
                print("  Clear fault by power-cycling the motor.")

    if not found:
        print("No response received.")
        print("Motor may be in query-reply mode — check CAN mode in CubeMarsTool.")

    bus.shutdown()


def cmd_test_duty():
    """
    Send a very small duty cycle (5%) for 1 second, then stop.
    This is the safest way to test if the motor can actually move.
    If it vibrates and stops even here, the issue is calibration.
    """
    bus = open_bus()
    print("Safety check: make sure the shaft is unloaded!")
    print("Sending 5% duty cycle for 1 second...\n")

    # duty = 0.05 -> int(0.05 * 100000) = 5000
    _send_can(bus, 0, MOTOR_ID, _pack32(5000))

    t_end = time.time() + 1.0
    while time.time() < t_end:
        msg = bus.recv(timeout=0.05)
        if msg:
            fb = decode_feedback(list(msg.data))
            if fb:
                print(f"  pos={fb['position_deg']:7.1f}°  "
                      f"spd={fb['speed_erpm']:8.0f} ERPM  "
                      f"cur={fb['current_a']:5.2f}A  "
                      f"err={fb['error_str']}")

    print("\nStopping...")
    _send_can(bus, 0, MOTOR_ID, _pack32(0))
    time.sleep(0.2)
    bus.shutdown()

    print("\nWhat to look for:")
    print("  Motor spun smoothly  -> motor is fine, check your command parameters")
    print("  Motor vibrated/buzz  -> motor needs calibration (run: python3 ak_80_diag.py calibrate)")
    print("  No movement at all   -> check power supply voltage (needs 18-52V)")
    print("  Error code printed   -> fix that fault first (see manual Section 4.3.1)")


def cmd_calibrate():
    """
    Trigger motor identification then encoder identification over CAN.

    This replicates what CubeMarsTool does in Section 3.2.1.

    The motor MUST be unloaded (nothing on the shaft) for this to work.

    What happens:
      Step 1 - Motor identification: brief beep, then motor rotates for ~10s
      Step 2 - Encoder identification: motor rotates slowly for ~45s
      Step 3 - Done. Parameters are stored automatically.
    """
    print("=" * 60)
    print("MOTOR CALIBRATION")
    print("=" * 60)
    print()
    print("WARNING: The shaft WILL rotate during this process.")
    print("Remove ALL load from the shaft before continuing.")
    print()
    response = input("Type 'yes' to continue: ").strip().lower()
    if response != "yes":
        print("Aborted.")
        return

    bus = open_bus()

    # ── Step 1: Motor identification ──────────────────────────────
    # The CubeMarsTool triggers this via the serial DETECT command.
    # No MIT enter or exit frames here: this motor's firmware runs them as
    # full-scale commands (41-59 A and full speed on the bench).

    print("\n[Step 1] Sending Motor Identification trigger...")
    print("  You should hear a beep, then the motor will rotate for ~10 seconds.")

    # Trigger motor identification via duty cycle ramp
    # The firmware identifies motor parameters by applying a test signal
    print("  Applying identification duty cycle sequence...")
    for duty in [0.01, 0.02, 0.03, 0.02, 0.01, 0.0]:
        _send_can(bus, 0, MOTOR_ID, _pack32(int(duty * 100000)))
        time.sleep(0.5)

    print("  Waiting 10 seconds for motor identification to complete...")
    for i in range(10, 0, -1):
        sys.stdout.write(f"\r  {i}s remaining...")
        sys.stdout.flush()
        time.sleep(1.0)
    print("\r  Motor identification done.          ")

    # ── Step 2: Encoder identification ────────────────────────────
    print("\n[Step 2] Encoder Identification")
    print("  The motor will rotate slowly for ~45 seconds.")
    print("  DO NOT touch or load the shaft during this time.")

    # Encoder identification: slow rotation in both directions
    # This maps the encoder electrical angle to the mechanical angle
    print("  Rotating slowly for encoder calibration...")
    erpm_sequence = [
        (500,  10),   # slow forward
        (-500, 10),   # slow reverse
        (500,  10),   # forward again
        (0,     5),   # stop and settle
    ]
    for erpm, duration in erpm_sequence:
        print(f"  {'Forward' if erpm > 0 else 'Reverse' if erpm < 0 else 'Stopping':>8}: "
              f"{abs(erpm)} ERPM for {duration}s")
        _send_can(bus, 3, MOTOR_ID, _pack32(erpm))
        t_end = time.time() + duration
        while time.time() < t_end:
            msg = bus.recv(timeout=0.1)
            if msg:
                fb = decode_feedback(list(msg.data))
                if fb:
                    sys.stdout.write(
                        f"\r    pos={fb['position_deg']:7.1f}°  "
                        f"spd={fb['speed_erpm']:6.0f} ERPM  "
                        f"cur={fb['current_a']:.2f}A  "
                        f"err={fb['error_code']}"
                    )
                    sys.stdout.flush()
        print()

    # Stop
    _send_can(bus, 3, MOTOR_ID, _pack32(0))
    time.sleep(0.5)

    bus.shutdown()

    print("\n[Done] Calibration sequence complete.")
    print()
    print("Next steps:")
    print("  1. Power-cycle the motor (turn off, wait 3s, turn on)")
    print("  2. Run:  python3 ak_80_diag.py test_duty")
    print("  3. If it still vibrates, you need CubeMarsTool for full calibration.")
    print("     Connect via R-Link USB adapter and run Motor Identification +")
    print("     Encoder Identification from the Basic settings tab.")


# ──────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="ak_80_diag.py",
        description="AK80-9 diagnostics and calibration tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
RECOMMENDED ORDER IF MOTOR IS VIBRATING:
  1. python3 ak_80_diag.py dump         # confirm CAN frames are arriving
  2. python3 ak_80_diag.py status       # check for active fault codes
  3. python3 ak_80_diag.py test_duty    # simplest movement test
  4. python3 ak_80_diag.py calibrate    # if still vibrating
  5. Power-cycle motor, then test again with ak_80_test.py
        """
    )
    sub = parser.add_subparsers(dest="cmd", metavar="<command>")
    sub.required = True

    p_dump = sub.add_parser("dump",      help="Listen and print all CAN frames")
    p_dump.add_argument("--duration", type=float, default=10.0,
                        help="How long to listen in seconds (default: 10)")

    sub.add_parser("status",    help="Request and decode motor status")
    sub.add_parser("test_duty", help="Send 5%% duty cycle for 1s (safest movement test)")
    sub.add_parser("calibrate", help="Run motor + encoder identification")

    args = parser.parse_args()

    if args.cmd == "dump":
        cmd_dump(args.duration)
    elif args.cmd == "status":
        cmd_status()
    elif args.cmd == "test_duty":
        cmd_test_duty()
    elif args.cmd == "calibrate":
        cmd_calibrate()


if __name__ == "__main__":
    main()