"""S6, S7, S10: the motor thread, the status check and the disable command, on a fake bus."""
import time
import types

import pytest

from exo.deploy.motor import MotorFault, MotorInterface
from fakes import FakeBus, is_special, torque_of, wait_for


def motor_on_fake_bus(**kwargs):
    m = MotorInterface(dry_run=True, **kwargs)
    bus = FakeBus()
    m.dry_run = False
    m._can = types.SimpleNamespace(Message=lambda **k: types.SimpleNamespace(**k))
    m._bus = bus
    return m, bus


@pytest.fixture
def motor():
    made = []

    def build(stale_s=0.05, **kwargs):
        m, bus = motor_on_fake_bus(torque_limit_nm=2.0, **kwargs)
        m.start_sender(stale_s=stale_s)
        made.append(m)
        return m, bus

    yield build
    for m in made:
        m.shutdown()


def faults_while_streaming(m, torque=1.0, seconds=1.0):
    """Send like the control loop; True once the motor refuses with MotorFault."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        try:
            m.send_torque(torque)
        except MotorFault:
            return True
        time.sleep(0.01)
    return False


def test_thread_sends_the_commanded_torque(motor):
    m, bus = motor()
    m.send_torque(1.0)
    assert wait_for(lambda: bus.frames()) is not None
    assert torque_of(bus.frames()[-1]) == pytest.approx(1.0, abs=0.02)


def test_hung_loop_is_zeroed_by_the_thread(motor):
    m, bus = motor(stale_s=0.05)
    m.send_torque(1.5)
    assert wait_for(lambda: bus.frames() and abs(torque_of(bus.frames()[-1])) > 1) is not None
    # the control loop now hangs: nothing calls send_torque any more
    took = wait_for(lambda: abs(torque_of(bus.frames()[-1])) < 0.02)
    assert took is not None and took < 0.2
    with pytest.raises(MotorFault):
        m.send_torque(1.0)


def test_steady_control_loop_does_not_trip(motor):
    m, bus = motor(stale_s=0.05)
    assert not faults_while_streaming(m, seconds=0.4)
    assert len(bus.frames()) > 20


def test_passive_run_keeps_sending_zero_without_a_fault(motor):
    m, bus = motor(stale_s=0.05)
    m.send_torque(0.0)
    time.sleep(0.3)
    assert len(bus.frames()) >= 3
    assert all(abs(torque_of(f)) < 0.02 for f in bus.frames())
    m.send_torque(0.0)


def test_motor_error_code_faults(motor):
    m, bus = motor()
    bus.error = 2
    assert faults_while_streaming(m)
    assert wait_for(lambda: abs(torque_of(bus.frames()[-1])) < 0.02) is not None


@pytest.mark.parametrize("iq_a", [41.0, -41.0])
def test_overcurrent_faults_in_either_sign(motor, iq_a):
    m, bus = motor()
    bus.iq_a = iq_a
    assert faults_while_streaming(m)


def test_normal_current_with_the_motors_flipped_sign_is_fine(motor):
    m, bus = motor(stale_s=0.05)
    bus.iq_a = 2.8                  # recorded value for about -2 N·m
    assert not faults_while_streaming(m, torque=2.0, seconds=0.3)
    assert m.status["current_a"] == pytest.approx(2.8)


def test_losing_status_faults(motor):
    m, bus = motor()
    bus.status_on = False
    assert faults_while_streaming(m)


def test_can_send_failure_faults_and_zero_goes_out_when_it_recovers(motor):
    m, bus = motor()
    bus.fail_sends = True
    assert faults_while_streaming(m)
    bus.fail_sends = False
    assert wait_for(lambda: bus.frames() and abs(torque_of(bus.frames()[-1])) < 0.02) is not None


def test_frames_from_other_ids_are_ignored(motor):
    m, bus = motor(stale_s=0.05)
    other = FakeBus()
    other.iq_a, other.error = 50.0, 3
    frame = other.status_frame()
    bus.rx.append(types.SimpleNamespace(arbitration_id=0x2969, is_extended_id=True, data=frame.data))
    bus.rx.append(types.SimpleNamespace(arbitration_id=0x68, is_extended_id=False, data=frame.data))
    assert not faults_while_streaming(m, seconds=0.2)


def test_read_feedback_accepts_only_status_frames():
    m, bus = motor_on_fake_bus()
    bus.status_on = False
    bus.rx.append(types.SimpleNamespace(arbitration_id=0x868, is_extended_id=True, data=bytes(8)))
    assert m.read_feedback() is None
    bus.status_on = True
    assert m.read_feedback()["position_deg"] == pytest.approx(87.5)


def test_stop_works_after_a_fault(motor):
    m, bus = motor()
    bus.error = 1
    assert faults_while_streaming(m)
    count = len(bus.frames())
    m.stop()
    assert len(bus.frames()) > count
    assert abs(torque_of(bus.frames()[-1])) < 0.02


def test_shutdown_ends_at_zero_and_stops_the_thread():
    m, bus = motor_on_fake_bus(torque_limit_nm=2.0)
    m.start_sender(stale_s=0.05)
    m.send_torque(1.0)
    assert wait_for(lambda: bus.frames()) is not None
    m.shutdown()
    count = len(bus.frames())
    assert abs(torque_of(bus.frames()[-1])) < 0.02
    time.sleep(0.15)
    assert len(bus.frames()) == count


def test_no_special_frame_in_a_full_session():
    m, bus = motor_on_fake_bus(torque_limit_nm=2.0)
    m.enter()
    m.start_sender(stale_s=0.05)
    assert not faults_while_streaming(m, seconds=0.1)
    m.stop()
    m.shutdown()
    assert bus.frames() and not [f for f in bus.frames() if is_special(f)]


def test_disable_frame_is_off_by_default():
    m, bus = motor_on_fake_bus()
    m.shutdown()
    assert not [f for f in bus.frames() if f.arbitration_id == 0xF68]


def test_disable_frame_matches_the_manual_when_requested():
    m, bus = motor_on_fake_bus(disable_on_shutdown=True)
    m.shutdown()
    last = bus.frames()[-1]
    assert (last.arbitration_id, list(last.data), last.is_extended_id) == (0xF68, [], True)
    assert abs(torque_of(bus.frames()[-2])) < 0.02


def test_dry_run_thread_needs_no_bus():
    m = MotorInterface(torque_limit_nm=2.0, dry_run=True)
    m.start_sender(stale_s=0.05)
    assert m.send_torque(1.0) == pytest.approx(1.0)
    m.shutdown()
