"""MotorInterface: torque clamp, sign, MIT packing, lifecycle (all dry-run)."""
import pytest

from exo.deploy.motor import (
    MIT_T_MAX,
    MIT_T_MIN,
    MotorInterface,
    _float_to_uint,
    _pack_mit,
)


def test_torque_cap_never_exceeds_mit_range():
    m = MotorInterface(torque_limit_nm=99.0, dry_run=True)
    assert m.torque_limit_nm == MIT_T_MAX


def test_clamp_to_configured_cap():
    m = MotorInterface(torque_limit_nm=2.0, dry_run=True)
    assert m.send_torque(10.0) == 2.0
    assert m.send_torque(-10.0) == -2.0
    assert m.send_torque(1.3) == pytest.approx(1.3)


def test_command_sign_applied():
    m = MotorInterface(torque_limit_nm=5.0, command_sign=-1.0, dry_run=True)
    assert m.send_torque(3.0) == -3.0


def test_pack_mit_zero_torque_is_midscale():
    buf = _pack_mit(0.0, 0.0, 0.0, 0.0, 0.0)
    assert len(buf) == 8
    t_int = ((buf[6] & 0xF) << 8) | buf[7]
    assert t_int == _float_to_uint(0.0, MIT_T_MIN, MIT_T_MAX, 12)
    assert abs(t_int - 2048) <= 1                 # centre of a 12-bit field


def test_pack_mit_sign_direction():
    pos = ((_pack_mit(0, 0, 2.0, 0, 0)[6] & 0xF) << 8) | _pack_mit(0, 0, 2.0, 0, 0)[7]
    neg = ((_pack_mit(0, 0, -2.0, 0, 0)[6] & 0xF) << 8) | _pack_mit(0, 0, -2.0, 0, 0)[7]
    assert pos > 2048 > neg


def test_lifecycle_no_crash_dry_run():
    m = MotorInterface(torque_limit_nm=2.0, dry_run=True)
    m.enter()
    m.zero_position()
    m.send_torque(1.5)
    m.stop()
    assert m.last_torque_nm == 0.0
    m.shutdown()


def test_read_feedback_none_without_bus():
    m = MotorInterface(dry_run=True)
    assert m.read_feedback() is None
