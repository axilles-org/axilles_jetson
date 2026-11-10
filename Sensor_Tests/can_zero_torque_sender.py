#!/usr/bin/env python3
"""
can_zero_torque_sender.py — keep the AK80-9 driver active with 0 Nm commands,
and nothing else. No I2C, no sensors, so a sensor monitor
(true_dropout_monitor.py) can run at the same time and see what an ACTIVE
motor driver does to the sensors.

Sends the same frame as Fall_tests/fall_logger.py --actuator --torque 0:
MIT mode, position 0, velocity 0, Kp 0, Kd 0, torque 0 (decodes to about
-0.005 Nm), extended ID (8 << 8) | 0x68, at --hz (default 100). The torque is
fixed at zero; there is no option to change it. On exit (end of --seconds,
Ctrl+C or SIGTERM) it sends three more zero frames and closes the bus.

Needs can0 up and the motor powered; keep the emergency stop in reach.
    python3 can_zero_torque_sender.py --seconds 75
"""
import argparse
import signal
import time

import can

CAN_INTERFACE = "can0"
MOTOR_ID = 0x68
MODE_MIT = 8
MIT_P, MIT_V, MIT_T = (-12.56, 12.56), (-65.0, 65.0), (-22.0, 22.0)
MIT_KP, MIT_KD = (0.0, 500.0), (0.0, 5.0)
_stop = False


def _f2u(x, lo_hi, bits):
    lo, hi = lo_hi
    x = max(lo, min(hi, x))
    return int((x - lo) / (hi - lo) * ((1 << bits) - 1))


def zero_frame():
    kp, kd = _f2u(0.0, MIT_KP, 12), _f2u(0.0, MIT_KD, 12)
    p, v, t = _f2u(0.0, MIT_P, 16), _f2u(0.0, MIT_V, 12), _f2u(0.0, MIT_T, 12)
    return [kp >> 4, ((kp & 0xF) << 4) | (kd >> 8), kd & 0xFF, p >> 8, p & 0xFF,
            v >> 4, ((v & 0xF) << 4) | (t >> 8), t & 0xFF]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--hz", type=float, default=100.0)
    a = ap.parse_args()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: globals().__setitem__("_stop", True))

    bus = can.interface.Bus(channel=CAN_INTERFACE, interface="socketcan")
    msg = can.Message(arbitration_id=(MODE_MIT << 8) | MOTOR_ID, data=zero_frame(),
                      is_extended_id=True)
    sent = fail = rx = 0
    period = 1.0 / a.hz
    t0 = time.monotonic()
    nxt = t0
    print(f"[can0] sending 0 Nm MIT frames at {a.hz:g} Hz for {a.seconds:g} s", flush=True)
    try:
        while not _stop and time.monotonic() - t0 < a.seconds:
            try:
                bus.send(msg)
                sent += 1
            except (can.CanError, OSError):
                fail += 1
            while bus.recv(timeout=0) is not None:
                rx += 1
            nxt += period
            time.sleep(max(0.0, nxt - time.monotonic()))
    finally:
        for _ in range(3):
            try:
                bus.send(msg)
            except (can.CanError, OSError):
                pass
            time.sleep(0.005)
        bus.shutdown()
        dur = time.monotonic() - t0
        print(f"[can0] done: {sent} sent ({sent / dur:.0f}/s), {fail} send errors, "
              f"{rx} replies ({rx / dur:.0f}/s) in {dur:.1f} s", flush=True)


if __name__ == "__main__":
    main()
