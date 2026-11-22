"""Status LED logic: no serial port or Arduino involved (fake serial below)."""
import time

import pytest

from exo.deploy.leds import (LedMode, LedStatus, StanceWindow, config_packet,
                             foot_gyro_rad_s, state_packet, torque_color)


def test_torque_color_runs_blue_to_red():
    assert torque_color(0.0, 2.0) == (0, 0, 255)
    assert torque_color(2.0, 2.0) == (255, 0, 0)
    assert torque_color(1.0, 2.0) == (128, 0, 127)
    assert torque_color(-1.0, 2.0) == (128, 0, 127)      # dorsiflexion: magnitude
    assert torque_color(9.0, 2.0) == (255, 0, 0)
    assert torque_color(float("nan"), 2.0) == (0, 0, 255)


def test_packets():
    assert config_packet(12, "BGR", 0.5, 2) == b"DC" + bytes((12, 2, 1, 0, 16, 2))
    assert config_packet(12, "rgb", 1.0, 2) == b"DC" + bytes((12, 0, 1, 2, 31, 2))
    assert state_packet(LedMode.ASSIST, 3, (255, 0, 1)) == b"DP" + bytes((3, 3, 255, 0, 1))
    assert state_packet(LedMode.OFF) == b"DP" + bytes((0, 0xFF, 0, 0, 0))
    with pytest.raises(ValueError):
        config_packet(12, "RGG")


DT = 0.01                                                  # controller runs at 100 Hz


def _quiet(k):                                             # standing foot: small, changing
    return 0.05 + 0.01 * (k % 5)


def _swinging(k):                                          # real swing: ~6 rad/s
    return 6.0 + 0.01 * (k % 5)


class Walker:
    """Feeds a StanceWindow 100 Hz samples and keeps the time."""

    def __init__(self, window: StanceWindow):
        self.w = window
        self.t = 0.0

    def feed(self, seconds: float, stance: bool, gyro) -> list:
        out = []
        for k in range(round(seconds / DT)):
            g = gyro(k) if callable(gyro) else gyro
            out.append(self.w.update(stance, self.t, g))
            self.t += DT
        return out

    def stride(self, stance_s: float = 0.75, dropout_at_s: float | None = 0.5) -> list:
        """Real swing, then a stance with the push-off FSR dropout seen in the logs."""
        self.feed(0.5, False, _swinging)
        if dropout_at_s is None:
            return self.feed(stance_s, True, _quiet)
        shown = self.feed(dropout_at_s, True, _quiet)
        shown += self.feed(0.05, False, lambda k: 1.5 + 0.01 * k)
        shown += self.feed(stance_s - dropout_at_s - 0.05, True, _quiet)
        return shown


def test_window_waits_for_a_real_step():
    w = Walker(StanceWindow(12, 2))
    assert set(w.feed(1.0, True, _quiet)) == {None}        # run started mid-stance
    shown = w.stride(dropout_at_s=None)
    assert shown[0] == 0


def test_weight_shifts_while_standing_never_show_a_window():
    w = Walker(StanceWindow(12, 2))
    w.feed(2.0, True, _quiet)
    for _ in range(3):
        assert set(w.feed(0.3, False, lambda k: 0.3 + 0.01 * k)) == {None}   # foot unloads
        assert set(w.feed(1.0, True, _quiet)) == {None}
        assert set(w.feed(0.05, False, _quiet)) == {None}                     # FSR flicker
        assert set(w.feed(1.0, True, _quiet)) == {None}


def test_window_sweeps_through_the_push_off_dropout():
    w = Walker(StanceWindow(12, 2, first_stance_s=0.75))
    shown = w.stride(stance_s=0.75, dropout_at_s=0.5)
    assert None not in shown                               # dropout does not end the stance
    assert shown[0] == 0
    assert shown == sorted(shown)                          # never restarts at the first LED
    assert shown[-1] == 10                                 # last two LEDs by toe-off
    assert w.feed(0.01, False, _swinging) == [None]        # real swing: off at once


def test_window_off_once_standing():
    w = Walker(StanceWindow(12, 2))
    shown = w.stride(stance_s=3.0, dropout_at_s=None)      # step, then stand
    on = [p for p in shown if p is not None]
    assert len(on) == pytest.approx(1.2 / DT, abs=2)       # off after STANDING_STANCE_S
    assert shown[-1] is None
    assert set(w.feed(2.0, True, _quiet)) == {None}
    assert w.stride(dropout_at_s=None)[0] == 0             # next real step: back on


def test_window_learns_the_full_stance_time():
    w = Walker(StanceWindow(12, 2, first_stance_s=0.5))
    for _ in range(3):
        w.stride(stance_s=0.9, dropout_at_s=0.6)
    assert w.w.expected_stance_s == pytest.approx(0.9, abs=0.02)   # dropout merged
    w.feed(0.5, False, _swinging)
    assert w.feed(0.45, True, _quiet)[-1] == 5             # halfway through 0.9 s


def test_standing_bout_does_not_change_the_estimate():
    w = Walker(StanceWindow(12, 2, first_stance_s=0.75))
    w.stride(stance_s=10.0, dropout_at_s=None)
    w.feed(0.5, False, _swinging)
    assert w.w.expected_stance_s == 0.75


def test_frozen_gyro_falls_back_to_swing_duration():
    w = Walker(StanceWindow(12, 2))
    w.feed(1.0, True, 3.61)                                # IMU stuck at one value
    assert set(w.feed(0.1, False, 3.61)) == {None}         # short gap: not a step
    w.feed(0.5, True, 3.61)
    w.feed(0.3, False, 3.61)                               # 0.3 s swing: a step
    assert w.feed(0.01, True, 3.61) == [0]


def test_foot_gyro_magnitude():
    assert foot_gyro_rad_s({"foot_gx": 3.0, "foot_gy": 0.0, "foot_gz": -4.0}) == 5.0


class FakeSerial:
    def __init__(self):
        self.written = b""
        self.closed = False

    def write(self, data):
        self.written += data

    def reset_input_buffer(self):
        pass

    def flush(self):
        pass

    def close(self):
        self.closed = True


def _packets(data: bytes) -> list[bytes]:
    out = []
    while data:
        size = 8 if data[:2] == b"DC" else 7
        out.append(data[:size])
        data = data[size:]
    return out


def test_status_sends_config_states_and_final_event():
    fake = FakeSerial()
    leds = LedStatus(send_hz=500, serial_factory=lambda: fake)
    leds.set_state(LedMode.ASSIST, 4, (10, 0, 245))
    time.sleep(0.05)
    leds.event(LedMode.ESTOP)
    leds.close()

    packets = _packets(fake.written)
    assert packets[0] == config_packet()
    assert state_packet(LedMode.ASSIST, 4, (10, 0, 245)) in packets
    estop = packets.index(state_packet(LedMode.ESTOP))
    assert all(p[2] != LedMode.ASSIST for p in packets[estop:])
    assert fake.closed


def test_missing_arduino_never_reaches_the_caller():
    def no_port():
        raise OSError("no such port")

    leds = LedStatus(send_hz=500, serial_factory=no_port)
    leds.set_state(LedMode.CALIBRATING)
    time.sleep(0.02)
    leds.event(LedMode.STOP)
    leds.close()


def test_one_shots_and_steady_states_are_not_mixed_up():
    leds = LedStatus(send_hz=500, serial_factory=FakeSerial)
    with pytest.raises(ValueError):
        leds.set_state(LedMode.ESTOP)
    with pytest.raises(ValueError):
        leds.event(LedMode.ASSIST)
    leds.close()
