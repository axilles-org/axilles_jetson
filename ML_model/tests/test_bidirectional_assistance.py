"""Command-path regression tests, with no hardware connection."""
import importlib.util
from pathlib import Path
import threading
import time

import pytest

from exo.config import replace
from exo.deploy.assistance import AssistanceController
from exo.deploy.jetson_io import JetsonSensors, SensorFault
from exo.deploy.motor import MotorInterface


def controller(cfg, **changes):
    values = dict(assistance_mode="bidirectional", assistance_scale=0.02,
                  ramp_in_s=0.001, rate_limit_nm_per_s=10000,
                  dorsiflexion_limit_nm=0.5, exo_command_sign=1.0,
                  plantarflexion_sign=-1.0)
    values.update(changes)
    return AssistanceController(replace(cfg, **{f"deploy.{k}": v for k, v in values.items()}).deploy)


@pytest.mark.parametrize("sign", [-1, 1])
def test_both_directions_and_dorsi_cap(cfg, sign):
    c = controller(cfg, exo_command_sign=sign)
    assert c.update(-1, 70, 0, 0, .01, stance=True).torque_nm == pytest.approx(sign * 1.4)
    assert c.update(+1, 70, 0, 0, .01, stance=True).torque_nm == pytest.approx(-sign * .5)
    assert c.update(+1, 70, 99999, 99999, .01, stance=False).torque_nm == 0


def test_nonfinite_prediction_latches_off(cfg):
    c = controller(cfg)
    c.update(float("nan"), 70, 0, 0, .01, stance=True)
    assert c.update(-1, 70, 0, 0, .01, stance=True).torque_nm == 0
    assert c.faulted


def load_runner():
    path = Path(__file__).parents[1] / "scripts/jetson_deploy.py"
    spec = importlib.util.spec_from_file_location("jetson_deploy_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("disabled", ["enabled", "stance", "ready", "faulted", "overrun"])
@pytest.mark.parametrize("target", [-2.0, 2.0])
def test_final_gate_bypasses_motor_slew(disabled, target):
    send = load_runner().send_assistance
    m = MotorInterface(torque_limit_nm=5, dry_run=True, slew_nm_per_s=10)
    gates = dict(enabled=True, stance=True, ready=True, faulted=False, overrun=False)
    for _ in range(30):
        send(m, target, dt=.01, **gates)
    assert abs(m.last_torque_nm) == 2
    gates[disabled] = not gates[disabled]
    assert send(m, target, dt=.01, **gates) == 0
    assert m.last_torque_nm == 0


def test_stale_gyro_not_hidden_by_fresh_accel():
    # Instantiate without opening any bus. Numeric frame values are irrelevant:
    # report receipt times determine freshness even for a stationary wearer.
    s = JetsonSensors.__new__(JetsonSensors)
    s._lock = threading.Lock()
    s._error = None
    s._started = time.monotonic() - 10
    s.stale_s = .1
    s._updated = dict(foot_accel=time.monotonic(), foot_gyro=time.monotonic() - .5)
    with pytest.raises(SensorFault, match="foot_gyro"):
        s._check()
    s._updated["foot_gyro"] = time.monotonic()
    s._check()


def test_transmission_timestamp_only_on_bus_send():
    from types import SimpleNamespace
    m = MotorInterface(dry_run=True)
    m.send_torque(1)
    assert m.transmission_snapshot()[0] is None
    frames = []
    m.dry_run = False
    m._can = SimpleNamespace(Message=lambda **kw: kw)
    m._bus = SimpleNamespace(send=frames.append)
    m.send_torque(1)
    at, value = m.transmission_snapshot()
    assert at is not None and value == 1 and len(frames) == 1
    m.stop()
    assert m.transmission_snapshot()[1] == 0
