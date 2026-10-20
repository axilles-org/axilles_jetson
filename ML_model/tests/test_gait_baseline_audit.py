"""Guard against confusing waveform correlation with accuracy or future timing."""
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "audit_gait_baselines", Path(__file__).resolve().parents[1] / "scripts/audit_gait_baselines.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_squared_correlation_does_not_measure_amplitude_accuracy():
    truth = np.sin(np.linspace(0, 6 * np.pi, 1000))
    result = audit.scores(3 * truth + 2, truth)
    assert np.isclose(result["r2_corr"], 1)
    assert result["r2_pred"] < 0
    assert result["rmse_nm_per_kg"] > 1


def test_causal_template_is_invariant_to_future_heel_strikes():
    stance = (np.arange(600) % 100 < 60).astype(float)
    changed = stance.copy()
    changed[250:] = (np.arange(350) % 140 < 70).astype(float)
    template = np.sin(np.linspace(0, 2 * np.pi, 101))
    original = audit.causal_template(stance, template, 100, 100)
    modified = audit.causal_template(changed, template, 100, 100)
    assert np.array_equal(original[:250], modified[:250])
    assert np.array_equal(original[:250], audit.causal_template(stance[:250], template, 100, 100))


def test_scores_agree_with_independent_regression_metric():
    from sklearn.metrics import r2_score

    truth = np.array([0, -1, -2, -0.5, 0.2])
    pred = np.array([0.1, -0.8, -1.6, -0.9, 0])
    result = audit.scores(pred, truth)
    assert np.isclose(result["r2_pred"], r2_score(truth, pred))
