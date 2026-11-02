"""One stance detector: the torque gate uses the same debounced stance as the model."""
import pytest

from conftest import REPO

RUN_DIR = REPO / "ML_model" / "runs" / "deploy"


def test_assistance_has_no_stance_rule_of_its_own():
    src = (REPO / "ML_model" / "src" / "exo" / "deploy" / "assistance.py").read_text()
    assert "fsr_threshold" not in src and "StanceDetector" not in src


@pytest.mark.skipif(not (RUN_DIR / "best.onnx").exists(), reason="no deploy model")
def test_gate_and_model_switch_together_after_the_debounce():
    from exo.config import Config
    from exo.deploy.jetson_io import FRAME_KEYS
    from exo.deploy.runtime import ExoController

    cfg = Config.load(RUN_DIR / "config.yaml")
    ctrl = ExoController(RUN_DIR, cfg.deploy, subject_mass_kg=72.0)
    ctrl.reset()
    loaded = {k: 0.1 for k in FRAME_KEYS}
    loaded["heel_fsr_raw"] = loaded["toe_fsr_raw"] = 40000.0
    stance_ch = ctrl.feature_names.index("stance")

    for _ in range(5):                                  # 50 ms: shorter than the 0.1 s debounce
        out = ctrl.step(loaded, 0.01)
        assert out["stance"] == 0.0 and not ctrl.adapter.stance
        assert ctrl.buffer.window()[stance_ch, -1] == 0.0

    for _ in range(10):
        out = ctrl.step(loaded, 0.01)
        assert out["stance"] == float(ctrl.adapter.stance)
        assert ctrl.buffer.window()[stance_ch, -1] == float(ctrl.adapter.stance)
    assert out["stance"] == 1.0
