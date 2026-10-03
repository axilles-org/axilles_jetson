"""Assistance controller: stance gate, ramps, and safety limits."""
from exo.config import replace
from exo.deploy.assistance import AssistanceController


def _controller(cfg, **overrides):
    d = {
        "deploy.control_rate_hz": 100,
        "deploy.sensor_adapter.heel_fsr_threshold": 1000.0,
        "deploy.sensor_adapter.toe_fsr_threshold": 1000.0,
        **{f"deploy.{k}": v for k, v in overrides.items()},
    }
    return AssistanceController(replace(cfg, **d).deploy)


def test_zero_command_in_swing(cfg):
    ctrl = _controller(cfg)
    out = ctrl.update(predicted_moment_nm_per_kg=-1.5, subject_mass_kg=70,
                      heel_fsr=0.0, toe_fsr=0.0, dt=0.01)
    assert out.stance is False
    assert out.torque_nm == 0.0


def test_ramps_in_over_stance(cfg):
    ctrl = _controller(cfg, ramp_in_s=0.05, assistance_scale=0.2)
    ramps = []
    for _ in range(10):
        out = ctrl.update(-1.0, 70, heel_fsr=5000.0, toe_fsr=0.0, dt=0.01)
        ramps.append(out.ramp)
    assert ramps[0] < ramps[-1]
    assert ramps[-1] == 1.0


def test_torque_limit_saturates(cfg):
    ctrl = _controller(cfg, torque_limit_nm=15.0, rate_limit_nm_per_s=10_000, ramp_in_s=1e-3)
    for _ in range(5):
        out = ctrl.update(-5.0, 90, heel_fsr=5000.0, toe_fsr=0.0, dt=0.01)
    assert abs(out.torque_nm) <= 15.0 + 1e-6


def test_rate_limit_bounds_step(cfg):
    ctrl = _controller(cfg, rate_limit_nm_per_s=100.0, ramp_in_s=1e-3, torque_limit_nm=999)
    out = ctrl.update(-3.0, 80, heel_fsr=5000.0, toe_fsr=0.0, dt=0.01)
    assert abs(out.torque_nm) <= 100.0 * 0.01 + 1e-6


def test_wrong_direction_moment_gives_no_assist(cfg):
    ctrl = _controller(cfg)
    # positive predicted moment is dorsiflexion in the GaTech convention -> no assist
    out = ctrl.update(predicted_moment_nm_per_kg=+1.5, subject_mass_kg=70,
                      heel_fsr=5000.0, toe_fsr=0.0, dt=0.01)
    assert out.torque_nm == 0.0
