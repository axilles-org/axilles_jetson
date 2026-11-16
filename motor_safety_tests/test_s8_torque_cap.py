"""S8: unsafe torque caps are refused."""
import math

import pytest

from dashboard.backend import process_manager
from exo.deploy.motor import SAFE_T_MAX, MotorInterface


@pytest.mark.parametrize("cap", [0.0, -2.0, math.nan, math.inf, -math.inf])
def test_motor_refuses_a_bad_cap(cap):
    with pytest.raises(ValueError):
        MotorInterface(torque_limit_nm=cap, dry_run=True)


def test_motor_still_limits_a_high_cap():
    assert MotorInterface(torque_limit_nm=99.0, dry_run=True).torque_limit_nm == SAFE_T_MAX


def _live_argv(cap):
    return process_manager.build_argv(
        controller="ml-live", run_dir="runs/deploy", mass=72, replay_path=None, duration=10,
        backend=None, assist_scale=None, height=1.75, gender="M", torque_cap=cap)[0]


@pytest.mark.parametrize("cap", [0, 0.0, -2, 5.01, 99, None, "abc", "", math.nan, math.inf])
def test_backend_refuses_a_bad_cap(cap):
    with pytest.raises(ValueError):
        _live_argv(cap)


@pytest.mark.parametrize("cap, text", [(2, "2.0"), (0.5, "0.5"), (5, "5.0"), ("1.5", "1.5")])
def test_backend_passes_a_good_cap_through(cap, text):
    argv = _live_argv(cap)
    assert argv[argv.index("--torque-cap") + 1] == text


def test_backend_limit_matches_the_motor():
    assert process_manager.MAX_TORQUE_CAP_NM == SAFE_T_MAX
