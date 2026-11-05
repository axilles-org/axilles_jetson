"""Delay timing and final gating, without opening hardware."""
import dataclasses
import sys
from pathlib import Path

import pytest

from exo.deploy.delay import PredictionDelay
from exo.deploy.assistance import AssistanceController


def test_delay_uses_elapsed_time_at_irregular_rates():
    d = PredictionDelay(50)
    for t, value in [(0, -1), (.009, -2), (.031, +3), (.049, +4)]:
        assert d.update(value, t) == 0
        assert not d.ready
    assert d.update(+5, .050) == -1
    assert d.age_ms == pytest.approx(50)
    assert d.update(+6, .073) == -2
    assert d.age_ms == pytest.approx(64)
    assert d.update(+7, .081) == 3


def test_zero_delay_passthrough_and_bounded_history():
    d = PredictionDelay(0)
    for i in range(1000):
        assert d.update(i, i * .003) == i
        assert d.ready and d.age_ms == 0
    assert len(d._samples) == 1
    d = PredictionDelay(50)
    for i in range(10000):
        d.update(i, i * .01)
    assert len(d._samples) <= 7


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), 1001])
def test_invalid_delays_rejected(value):
    with pytest.raises(ValueError, match="assistance_delay_ms"):
        PredictionDelay(value)


def make_controller(cfg):
    return AssistanceController(dataclasses.replace(
        cfg.deploy, assistance_delay_ms=50, assistance_mode="bidirectional",
        assistance_scale=.02, ramp_in_s=.001, rate_limit_nm_per_s=10000,
        torque_limit_nm=2, dorsiflexion_limit_nm=.5))


def step(c, t, value=-1, **gates):
    args = dict(stance=True, ready=True, enabled=True, now=t)
    args.update(gates)
    return c.update(value, 70, 0, 0, .01, **args).torque_nm


@pytest.mark.parametrize("gate", ["stance", "enabled", "ready"])
def test_stop_is_immediate_and_restart_cannot_replay_old_torque(cfg, gate):
    c = make_controller(cfg)
    assert step(c, 0) == 0
    assert step(c, .05) == pytest.approx(1.4)
    assert step(c, .06, **{gate: False}) == 0
    assert not c.delay.ready
    assert step(c, .07, +1) == 0
    assert step(c, .11, +1) == 0
    assert step(c, .12, +1) == pytest.approx(-.5)


def test_fault_bypasses_delay_and_latches_until_reset(cfg):
    c = make_controller(cfg)
    step(c, 0)
    assert step(c, .05) > 0
    assert step(c, .06, float("nan")) == 0
    assert c.faulted and not c.delay.ready
    assert step(c, .12) == 0
    c.reset()
    assert step(c, .13) == 0
    assert step(c, .18) > 0


def test_scaling_and_direction_limits_apply_to_delayed_prediction(cfg):
    c = make_controller(cfg)
    step(c, 0, -10)
    assert step(c, .05, +10) == 2
    c.scale = .001
    assert step(c, .1, 0) == pytest.approx(-.5)  # dorsi cap, current scale
    c.scale = .0001
    assert step(c, .11, 0) == pytest.approx(-.07)


@pytest.mark.parametrize("mode", ["ml-live", "ml-mock", "ml-dry-run"])
def test_dashboard_passes_delay_to_each_ml_runner(mode):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from dashboard.backend.process_manager import build_argv
    argv, _ = build_argv(mode, "runs/deploy", 72, None, 10, "onnx", .02,
                         height=1.75, gender="M", torque_cap=2,
                         assistance_delay_ms=75)
    assert argv[argv.index("--assistance-delay-ms") + 1] == "75.0"


@pytest.mark.parametrize("value", [-1, "abc", float("nan"), float("inf"), 1001])
def test_dashboard_rejects_invalid_delay(value):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from dashboard.backend.process_manager import build_argv
    with pytest.raises(ValueError, match="assistance delay"):
        build_argv("ml-mock", "runs/deploy", 72, None, 10, "onnx", .02,
                   assistance_delay_ms=value)
