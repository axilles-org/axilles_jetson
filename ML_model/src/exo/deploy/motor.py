"""CAN interface to the CubeMars AK80-9 in MIT mode, feed-forward torque only.

Every command is hard-clamped before packing. ``python-can`` is imported lazily,
so this module loads without CAN; only ``MotorInterface`` (non-dry-run) needs it.
Protocol constants follow the axilles_jetson repo (TBE_controller/data_obtainer.py,
Cubemars_AK80-9/ak_80_test.py): frames are packed with the motor's ±22 N·m
encoding range, and every command is clamped to the separate ±5 N·m SAFE_T_MAX.
"""
from __future__ import annotations

import json
import struct
import subprocess
import time

CAN_INTERFACE = "can0"
CAN_BITRATE = 1_000_000  # AK80-9 default
MOTOR_ID = 0x68

# MIT-mode field ranges: the encoding the motor firmware decodes with, NOT a
# limit. Packing with a narrower range than the firmware's scales every torque
# up (±5 vs ±22 -> 4.4x), so the torque range is ±22 as in TBE_controller.
MIT_P_MIN, MIT_P_MAX = -12.56, 12.56
MIT_V_MIN, MIT_V_MAX = -65.0, 65.0
MIT_T_MIN, MIT_T_MAX = -22.0, 22.0
MIT_KP_MIN, MIT_KP_MAX = 0.0, 500.0
MIT_KD_MIN, MIT_KD_MAX = 0.0, 5.0

# Safety ceiling on any commanded torque, independent of the encoding range
# (TBE_controller/data_obtainer.py uses the same SAFE_T_MAX).
SAFE_T_MAX = 5.0

MODE_MIT = 8
MODE_SET_ORIGIN = 5     # servo-mode "set origin" (same as Cubemars_AK80-9/ak_80_test.py zero)

# No MIT "special" frames (FF FF FF FF FF FF FF FC / FD / FE). The exo's AK80-9 runs
# CubeMars servo-mode firmware (status id 0x2968), which decodes them as ordinary
# MIT commands with every field at maximum (kp~500, kd~5, p=+12.56 rad, v=+65 rad/s,
# near-max torque). Bench recordings (fall_test_plan/test_4, 2026-10-27): 41-59 A
# and full speed, held until the motor's ~2 s CAN timeout. MIT commands work
# without them, so enter()/exit() only send 0 N·m.


def check_can_link(channel: str = CAN_INTERFACE, bitrate: int = CAN_BITRATE) -> None:
    """Raise with the fix command if ``channel`` is down or at the wrong bitrate
    (python-can only fails later, at the first send, with "Network is down")."""
    fix = (f"sudo ip link set {channel} down && "
           f"sudo ip link set {channel} up type can bitrate {bitrate} restart-ms 100")
    try:
        out = subprocess.run(["ip", "-json", "-details", "link", "show", channel],
                             capture_output=True, text=True, check=True).stdout
        link = json.loads(out)[0]
    except (OSError, subprocess.CalledProcessError, ValueError, IndexError) as exc:
        raise RuntimeError(f"CAN interface {channel} not found ({exc})") from exc

    info = link.get("linkinfo", {}).get("info_data", {})
    actual = info.get("bittiming", {}).get("bitrate")
    if "UP" not in link.get("flags", []):
        raise RuntimeError(f"CAN interface {channel} is down. Run: {fix}")
    if actual != bitrate:
        raise RuntimeError(f"CAN interface {channel} bitrate is {actual}, "
                           f"motor expects {bitrate}. Run: {fix}")


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _float_to_uint(x: float, lo: float, hi: float, bits: int) -> int:
    x = _clamp(x, lo, hi)
    return int((x - lo) * ((1 << bits) / (hi - lo)))


def _pack_mit(pos: float, vel: float, torque: float, kp: float, kd: float) -> list[int]:

    # clamp torque
    torque = _clamp(torque, -SAFE_T_MAX, SAFE_T_MAX)
    
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

    ``torque_limit_nm`` is clamped to at most ``SAFE_T_MAX`` (5). ``command_sign``
    maps a positive assist magnitude to the plantarflexion-assist direction —
    -1 on the exo (TBE_controller/data_obtainer.py negates its profile too).
    ``slew_nm_per_s`` bounds how fast the sent torque may change when
    ``send_torque`` is given ``dt``; ``stop()`` always zeros immediately.
    """

    def __init__(self, torque_limit_nm: float = 5.0, command_sign: float = 1.0,
                 channel: str = CAN_INTERFACE, motor_id: int = MOTOR_ID,
                 dry_run: bool = False, slew_nm_per_s: float | None = None):
        self.torque_limit_nm = min(float(torque_limit_nm), SAFE_T_MAX)
        self.command_sign = float(command_sign)
        self.slew_nm_per_s = slew_nm_per_s
        self.motor_id = motor_id
        self.dry_run = dry_run
        self.last_torque_nm = 0.0
        self._bus = None

        if dry_run:
            return
        import can  # lazy: Jetson only

        check_can_link(channel)
        self._can = can
        self._bus = can.interface.Bus(channel=channel, interface="socketcan")

    # -- lifecycle -------------------------------------------------
    def enter(self) -> None:
        """Start at 0 N·m (no MIT enter frame — see the note at the top)."""
        self.send_torque(0.0)
        time.sleep(0.05)

    def zero_position(self) -> None:
        """Make the current position the origin until power-off (servo-mode
        command, not the MIT zero frame — see the note at the top)."""
        self._raw([0], mode=MODE_SET_ORIGIN)
        time.sleep(0.05)

    def exit(self) -> None:
        """End at 0 N·m (no MIT exit frame — see the note at the top)."""
        self.send_torque(0.0)

    def stop(self) -> None:
        """Zero torque immediately — the e-stop action."""
        self.send_torque(0.0)

    # -- command --------------------------------------------------
    def send_torque(self, torque_nm: float, dt: float | None = None) -> float:
        """Command a feed-forward torque (N·m). Returns the value actually sent
        after sign, clamp and (with ``dt``) slew limit."""
        t = _clamp(self.command_sign * torque_nm,
                   -self.torque_limit_nm, self.torque_limit_nm)
        if self.slew_nm_per_s and dt:
            step = self.slew_nm_per_s * dt
            t = _clamp(t, self.last_torque_nm - step, self.last_torque_nm + step)
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
    def _raw(self, data: list[int], mode: int = MODE_MIT) -> None:
        if self.dry_run or self._bus is None:
            return
        arb = (mode << 8) | self.motor_id
        self._bus.send(self._can.Message(arbitration_id=arb, data=data,
                                         is_extended_id=True))