"""CAN interface to the CubeMars AK80-9 in MIT mode, feed-forward torque only.

Every command is hard-clamped before packing. ``python-can`` is imported lazily,
so this module loads without CAN; only ``MotorInterface`` (non-dry-run) needs it.
Protocol constants follow the axilles_jetson repo (TBE_controller/data_obtainer.py,
Cubemars_AK80-9/ak_80_test.py): frames are packed with the motor's ±22 N·m
encoding range, and every command is clamped to the separate ±5 N·m SAFE_T_MAX.
"""
from __future__ import annotations

import json
import math
import struct
import subprocess
import threading
import time

CAN_INTERFACE = "can0"
CAN_BITRATE = 1_000_000  # AK80-9 default
MOTOR_ID = 0x68

# MIT-mode field ranges: the encoding the motor firmware decodes with, NOT a
# limit. Packing with a narrower range than the firmware's scales every torque
# up (±5 vs ±22 -> 4.4x), so the torque range is ±22 as in TBE_controller.
# The V3 manual (p.31) lists ±25, but the saved Test 4 frames measured about
# 31.2 A per full scale, which is ±22 with the manual's Kt.
MIT_P_MIN, MIT_P_MAX = -12.56, 12.56
MIT_V_MIN, MIT_V_MAX = -65.0, 65.0
MIT_T_MIN, MIT_T_MAX = -22.0, 22.0
MIT_KP_MIN, MIT_KP_MAX = 0.0, 500.0
MIT_KD_MIN, MIT_KD_MAX = 0.0, 5.0

# Safety ceiling on any commanded torque, independent of the encoding range
# (TBE_controller/data_obtainer.py uses the same SAFE_T_MAX).
SAFE_T_MAX = 10.0

MODE_MIT = 8
MODE_SET_ORIGIN = 5     # servo-mode "set origin" (same as Cubemars_AK80-9/ak_80_test.py zero)
MODE_DISABLE = 15       # V3 manual 2.1.8, see MotorInterface.disable
STATUS_FUNCTION = 0x29  # status frames arrive on (0x29 << 8) | motor id, about 50 Hz

KT_NM_PER_A = 0.7044    # AK80-9 torque constant, V3 manual p.31

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


class MotorFault(RuntimeError):
    """The motor or its command path failed; torque is zero and stays zero."""


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _float_to_uint(x: float, lo: float, hi: float, bits: int) -> int:
    x = _clamp(x, lo, hi)
    return int((x - lo) / (hi - lo) * ((1 << bits) - 1))    # V3 manual p.34


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
    +1 on the exo (-1 gave dorsiflexion in a worn test on 2026-10-31).
    ``slew_nm_per_s`` bounds how fast the sent torque may change when
    ``send_torque`` is given ``dt``; ``stop()`` always zeros immediately.

    After ``start_sender`` a background thread transmits every torque frame and
    watches the motor. Any failure latches: torque goes to zero and every later
    ``send_torque`` raises ``MotorFault``.
    """

    def __init__(self, torque_limit_nm: float = 5.0, command_sign: float = 1.0,
                 channel: str = CAN_INTERFACE, motor_id: int = MOTOR_ID,
                 dry_run: bool = False, slew_nm_per_s: float | None = None,
                 disable_on_shutdown: bool = False):
        limit = float(torque_limit_nm)
        if not math.isfinite(limit) or limit <= 0.0:
            # a negative limit inverts _clamp, which then returns +|limit| for
            # every input, including the 0 N·m that stop() sends
            raise ValueError(f"torque limit must be a positive number, got {torque_limit_nm!r}")
        self.torque_limit_nm = min(limit, SAFE_T_MAX)
        self.command_sign = float(command_sign)
        self.slew_nm_per_s = slew_nm_per_s
        self.motor_id = motor_id
        self.dry_run = dry_run
        self.disable_on_shutdown = disable_on_shutdown
        self.last_torque_nm = 0.0
        self.iq_limit_a = max(2.0, 1.5 * self.torque_limit_nm / KT_NM_PER_A)
        self.last_tx_at: float | None = None
        self.last_tx_torque_nm = 0.0
        self.last_status_at: float | None = None
        self.status: dict | None = None     # latest status frame (sender thread)
        self._bus = None
        self._fault: str | None = None
        self._target_nm = 0.0
        self._tx_lock = threading.Lock()
        self._sender: threading.Thread | None = None
        self._stop_sender = threading.Event()
        self._new_target = threading.Event()

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
        self.stop()

    def stop(self) -> None:
        """Zero torque immediately — the e-stop action. Works after a fault."""
        self._target_nm = 0.0
        self.last_torque_nm = 0.0
        self._transmit_target()

    def disable(self) -> None:
        """Motor disable command (V3 manual 2.1.8); the motor answers with 0x77
        in status byte 7. Not yet run on this motor, and the manual does not say
        how it re-enables, so ``shutdown`` only sends it on request."""
        self._raw([], mode=MODE_DISABLE)

    def start_sender(self, stale_s: float = 0.1, status_timeout_s: float = 0.1) -> None:
        """Move torque frames to a background thread that survives a hung loop.

        ``send_torque`` then only hands over the target. If none arrives for
        ``stale_s`` the thread sends 0 N·m on its own, so a hung sensor read or
        inference cannot leave torque on until the motor's CAN timeout (about
        3 s). The thread also checks every status frame. It cannot help if the
        whole process dies.
        """
        if self._sender is not None:
            return
        self._sender = threading.Thread(target=self._sender_loop, name="motor-sender",
                                        args=(stale_s, status_timeout_s), daemon=True)
        self._sender.start()

    # -- command --------------------------------------------------
    def send_torque(self, torque_nm: float, dt: float | None = None,
                    *, immediate_zero: bool = False) -> float:
        """Command a feed-forward torque (N·m). Returns the value actually sent
        after sign, clamp and (with ``dt``) slew limit. With a sender thread this
        is the queued target, not measured torque or a CAN transmission timestamp.
        ``immediate_zero`` bypasses slew only for a zero target."""
        if immediate_zero and torque_nm != 0.0:
            raise ValueError("immediate_zero is only valid for a zero command")
        if self._fault is None and not math.isfinite(torque_nm):
            self._latch(f"non-finite torque command {torque_nm!r}")
            self.stop()
        if self._fault is not None:
            raise MotorFault(self._fault)
        t = _clamp(self.command_sign * torque_nm,
                   -self.torque_limit_nm, self.torque_limit_nm)
        if self.slew_nm_per_s and dt and not immediate_zero:
            step = self.slew_nm_per_s * dt
            t = _clamp(t, self.last_torque_nm - step, self.last_torque_nm + step)
        self.last_torque_nm = t
        self._target_nm = t
        if self._sender is not None:
            self._new_target.set()
        else:
            self._transmit_target()
        return t

    def read_feedback(self) -> dict | None:
        if self.dry_run or self._bus is None:
            return None
        return self._decode_status(self._bus.recv(timeout=0.001))

    def shutdown(self) -> None:
        if self._sender is not None:
            self._stop_sender.set()
            self._new_target.set()
            self._sender.join(timeout=1.0)
            self._sender = None
        try:
            self.exit()
            if self.disable_on_shutdown:
                self.disable()
        except Exception:
            pass
        if self._bus is not None:
            try:
                self._bus.shutdown()
            except Exception:
                pass

    # -- internal ------------------------------------------------
    def _latch(self, reason: str) -> None:
        if self._fault is None:
            self._fault = reason
        self._target_nm = 0.0
        self.last_torque_nm = 0.0

    def _decode_status(self, msg) -> dict | None:
        if (msg is None or not msg.is_extended_id or len(msg.data) < 6
                or msg.arbitration_id != (STATUS_FUNCTION << 8) | self.motor_id):
            return None
        return {
            "position_deg": struct.unpack(">h", bytes(msg.data[0:2]))[0] * 0.1,
            "speed_erpm": struct.unpack(">h", bytes(msg.data[2:4]))[0] * 10.0,
            "current_a": struct.unpack(">h", bytes(msg.data[4:6]))[0] * 0.01,
            "temp_c": msg.data[6] if len(msg.data) > 6 else None,
            "error": msg.data[7] if len(msg.data) > 7 else None,
        }

    def _check_status(self, deadline_s: float, last_seen: float) -> float:
        """Drain status frames and latch on a bad one. Compares |Iq| only: the
        motor reports Iq with the opposite sign to the command. Position is not
        used because it clips at ±3200°. Returns when status was last seen."""
        now = time.monotonic()
        while (msg := self._bus.recv(timeout=0.0)) is not None:
            status = self._decode_status(msg)
            if status is None:
                continue
            self.status, last_seen = status, now
            self.last_status_at = time.perf_counter()
            if status["error"]:
                self._latch(f"motor error code {status['error']}")
            if abs(status["current_a"]) > self.iq_limit_a:
                self._latch(f"|Iq| {abs(status['current_a']):.1f} A over "
                            f"the {self.iq_limit_a:.1f} A limit")
        if now - last_seen > deadline_s:
            self._latch(f"no motor status for {now - last_seen:.2f} s")
        return last_seen

    def _sender_loop(self, stale_s: float, status_timeout_s: float) -> None:
        last_status = time.monotonic()
        while not self._stop_sender.is_set():
            fresh = self._new_target.wait(timeout=stale_s)
            self._new_target.clear()
            if self._stop_sender.is_set():
                return
            if not fresh and self._target_nm != 0.0:
                self._latch(f"no torque target for {stale_s:.2f} s")
            try:
                if not self.dry_run and self._bus is not None:
                    last_status = self._check_status(status_timeout_s, last_status)
                self._transmit_target()
            except Exception as exc:                       # noqa: BLE001
                self._latch(f"CAN failure: {exc!r}")

    def _transmit_target(self) -> None:
        # Read the latest target under the transmit lock. A queued nonzero frame
        # must not be sent after stop() has already transmitted zero.
        if self.dry_run or self._bus is None:
            return
        with self._tx_lock:
            torque = self._target_nm
            self._bus.send(self._can.Message(
                arbitration_id=(MODE_MIT << 8) | self.motor_id,
                data=_pack_mit(0.0, 0.0, torque, 0.0, 0.0), is_extended_id=True))
            self.last_tx_at = time.perf_counter()
            self.last_tx_torque_nm = torque

    def transmission_snapshot(self) -> tuple[float | None, float]:
        with self._tx_lock:
            return self.last_tx_at, self.last_tx_torque_nm

    def _raw(self, data: list[int], mode: int = MODE_MIT) -> None:
        if self.dry_run or self._bus is None:
            return
        arb = (mode << 8) | self.motor_id
        with self._tx_lock:
            self._bus.send(self._can.Message(arbitration_id=arb, data=data,
                                             is_extended_id=True))