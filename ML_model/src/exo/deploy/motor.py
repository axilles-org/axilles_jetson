"""CAN interface to the CubeMars AK80-9 in MIT mode, feed-forward torque only.

Every command is hard-clamped before packing. ``python-can`` is imported lazily,
so this module loads without CAN; only ``MotorInterface`` (non-dry-run) needs it.
Protocol constants and the conservative ±5 N·m range follow the axilles_jetson
repo (TBE_controller/data_obtainer.py, Cubemars_AK80-9/ak_80_test.py).
"""
from __future__ import annotations

import struct
import time

CAN_INTERFACE = "can0"
MOTOR_ID = 0x68

# MIT-mode field ranges (match axilles_jetson's conservative AK80-9 setup)
MIT_P_MIN, MIT_P_MAX = -12.56, 12.56
MIT_V_MIN, MIT_V_MAX = -65.0, 65.0
MIT_T_MIN, MIT_T_MAX = -5.0, 5.0
MIT_KP_MIN, MIT_KP_MAX = 0.0, 500.0
MIT_KD_MIN, MIT_KD_MAX = 0.0, 5.0

MODE_MIT = 8

_ENTER_MIT = [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFC]
_EXIT_MIT = [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFD]
_ZERO_MIT = [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFE]


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _float_to_uint(x: float, lo: float, hi: float, bits: int) -> int:
    x = _clamp(x, lo, hi)
    return int((x - lo) * ((1 << bits) / (hi - lo)))


def _pack_mit(pos: float, vel: float, torque: float, kp: float, kd: float) -> list[int]:
    kp_i = _float_to_uint(kp, MIT_KP_MIN, MIT_KP_MAX, 12)
    kd_i = _float_to_uint(kd, MIT_KD_MIN, MIT_KD_MAX, 12)
    p_i = _float_to_uint(pos, MIT_P_MIN, MIT_P_MAX, 16)
    v_i = _float_to_uint(vel, MIT_V_MIN, MIT_V_MAX, 12)
    t_i = _float_to_uint(torque, MIT_T_MIN, MIT_T_MAX, 12)
    return [
        kp_i >> 4,
        ((kp_i & 0xF) << 4) | (kd_i >> 8),
        kd_i & 0xFF,
        p_i >> 8,
        p_i & 0xFF,
        v_i >> 4,
        ((v_i & 0xF) << 4) | (t_i >> 8),
        t_i & 0xFF,
    ]


class MotorInterface:
    """Feed-forward torque to the AK80-9 over CAN.

    ``torque_limit_nm`` is clamped to at most ``MIT_T_MAX`` (5). ``command_sign``
    maps a positive assist magnitude to the plantarflexion-assist direction —
    verify on the bench.
    """

    def __init__(self, torque_limit_nm: float = 5.0, command_sign: float = 1.0,
                 channel: str = CAN_INTERFACE, motor_id: int = MOTOR_ID,
                 dry_run: bool = False):
        self.torque_limit_nm = min(float(torque_limit_nm), MIT_T_MAX)
        self.command_sign = float(command_sign)
        self.motor_id = motor_id
        self.dry_run = dry_run
        self.last_torque_nm = 0.0
        self._bus = None

        if dry_run:
            return
        import can  # lazy: Jetson only

        self._can = can
        self._bus = can.interface.Bus(channel=channel, interface="socketcan")

    # -- lifecycle -------------------------------------------------
    def enter(self) -> None:
        self._raw(_ENTER_MIT)
        time.sleep(0.05)

    def zero_position(self) -> None:
        self._raw(_ZERO_MIT)
        time.sleep(0.05)

    def exit(self) -> None:
        self.send_torque(0.0)
        self._raw(_EXIT_MIT)

    def stop(self) -> None:
        """Zero torque immediately — the e-stop action."""
        self.send_torque(0.0)

    # -- command --------------------------------------------------
    def send_torque(self, torque_nm: float) -> float:
        """Command a feed-forward torque (N·m). Returns the value actually sent
        after sign and clamp."""
        t = _clamp(self.command_sign * torque_nm,
                   -self.torque_limit_nm, self.torque_limit_nm)
        self.last_torque_nm = t
        if self.dry_run or self._bus is None:
            return t
        self._raw(_pack_mit(0.0, 0.0, t, 0.0, 0.0))
        return t

    def read_feedback(self) -> dict | None:
        if self.dry_run or self._bus is None:
            return None
        msg = self._bus.recv(timeout=0.001)
        if msg is None or len(msg.data) < 6:
            return None
        return {
            "position_deg": struct.unpack(">h", bytes(msg.data[0:2]))[0] * 0.1,
            "speed_erpm": struct.unpack(">h", bytes(msg.data[2:4]))[0] * 10.0,
            "current_a": struct.unpack(">h", bytes(msg.data[4:6]))[0] * 0.01,
            "temp_c": msg.data[6] if len(msg.data) > 6 else None,
            "error": msg.data[7] if len(msg.data) > 7 else None,
        }

    def shutdown(self) -> None:
        try:
            self.exit()
        except Exception:
            pass
        if self._bus is not None:
            try:
                self._bus.shutdown()
            except Exception:
                pass

    # -- internal ------------------------------------------------
    def _raw(self, data: list[int]) -> None:
        if self.dry_run or self._bus is None:
            return
        arb = (MODE_MIT << 8) | self.motor_id
        self._bus.send(self._can.Message(arbitration_id=arb, data=data,
                                         is_extended_id=True))