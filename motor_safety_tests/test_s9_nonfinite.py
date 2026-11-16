"""S9: a NaN or infinite value can never become torque."""
import math
from pathlib import Path

import pytest

from conftest import REPO
from exo.config import Config, DeployConfig
from exo.deploy.assistance import AssistanceController
from exo.deploy.jetson_io import FRAME_KEYS
from exo.deploy.motor import MotorFault, MotorInterface
from exo.deploy.runtime import ExoController

HEEL_ON, HEEL_OFF = 30000.0, 0.0            # above and below the default thresholds
RUN_DIR = REPO / "ML_model" / "runs" / "deploy"


def _controller():
    return AssistanceController(DeployConfig(assistance_scale=0.2, torque_limit_nm=5.0))


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize("heel", [HEEL_ON, HEEL_OFF])
def test_bad_prediction_gives_zero_in_stance_and_swing(bad, heel):
    c = _controller()
    for _ in range(50):
        c.update(-1.0, 72.0, HEEL_ON, HEEL_ON, 0.01)
    out = c.update(bad, 72.0, heel, heel, 0.01)
    assert out.torque_nm == 0.0 and c.faulted


def test_assistance_stays_off_after_a_bad_prediction():
    c = _controller()
    c.update(math.nan, 72.0, HEEL_ON, HEEL_ON, 0.01)
    for _ in range(50):
        assert c.update(-1.0, 72.0, HEEL_ON, HEEL_ON, 0.01).torque_nm == 0.0
    c.reset()
    assert c.update(-1.0, 72.0, HEEL_ON, HEEL_ON, 0.01).torque_nm > 0.0


def test_bad_scale_is_caught_too():
    c = _controller()
    c.scale = math.nan
    assert c.update(-1.0, 72.0, HEEL_ON, HEEL_ON, 0.01).torque_nm == 0.0


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_motor_refuses_a_bad_torque(bad):
    m = MotorInterface(torque_limit_nm=2.0, dry_run=True)
    m.send_torque(1.0)
    with pytest.raises(MotorFault):
        m.send_torque(bad)
    assert m.last_torque_nm == 0.0
    with pytest.raises(MotorFault):
        m.send_torque(1.0)


@pytest.mark.skipif(not (RUN_DIR / "best.onnx").exists(), reason="no deploy model")
def test_real_model_with_nan_sensor_values_gives_zero_torque():
    cfg = Config.load(Path(RUN_DIR) / "config.yaml")
    controller = ExoController(RUN_DIR, cfg.deploy, subject_mass_kg=72.0)
    controller.reset()
    frame = {k: 0.1 for k in FRAME_KEYS}
    frame["heel_fsr_raw"] = frame["toe_fsr_raw"] = 40000.0
    for _ in range(301):
        out = controller.step(frame, 0.01)
    assert out["buffer_ready"] == 1.0 and out["model_fault"] == 0.0

    frame["foot_ax"] = math.nan
    out = controller.step(frame, 0.01)
    assert not math.isfinite(out["predicted_nm_per_kg"])    # the model really does pass NaN on
    assert out["command_nm"] == 0.0 and out["model_fault"] == 1.0
