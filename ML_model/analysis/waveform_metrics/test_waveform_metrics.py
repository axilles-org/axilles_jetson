import numpy as np
import pytest

from . import agreement, features, shape, timing
from .runner import compute_all


@pytest.fixture
def sine_pair():
    t = np.linspace(0, 1, 100, endpoint=False)
    gatech = np.sin(2 * np.pi * t)
    exo = gatech.copy()
    return exo, gatech


def test_identical_signals_are_perfect(sine_pair):
    exo, gatech = sine_pair
    assert shape.rmse(exo, gatech) == pytest.approx(0.0, abs=1e-9)
    assert shape.r2_score(exo, gatech) == pytest.approx(1.0)
    assert shape.pearson_r(exo, gatech) == pytest.approx(1.0)
    assert features.peak_value_error(exo, gatech) == pytest.approx(0.0)
    assert features.rom_error(exo, gatech) == pytest.approx(0.0)


def test_known_shift_is_detected(sine_pair):
    _, gatech = sine_pair
    shifted = np.roll(gatech, 5)
    assert timing.xcorr_lag(shifted, gatech) == 5


def test_dtw_zero_for_identical(sine_pair):
    exo, gatech = sine_pair
    assert timing.dtw_distance(exo, gatech) == pytest.approx(0.0, abs=1e-9)


def test_scaled_signal_lowers_r2_but_keeps_correlation(sine_pair):
    _, gatech = sine_pair
    scaled = gatech * 2.0
    assert shape.pearson_r(scaled, gatech) == pytest.approx(1.0)
    assert shape.r2_score(scaled, gatech) < 0.5


def test_nrmse_handles_flat_reference():
    flat = np.ones(50)
    exo = np.ones(50) * 1.1
    assert np.isnan(shape.nrmse(exo, flat))


def test_bland_altman_bias_matches_mean_difference():
    exo_peaks = np.array([1.0, 2.0, 3.0, 4.0])
    gatech_peaks = np.array([0.9, 1.9, 2.8, 4.1])
    bias, loa = agreement.bland_altman(exo_peaks, gatech_peaks)
    assert bias == pytest.approx(np.mean(exo_peaks - gatech_peaks))
    assert loa[0] < bias < loa[1]


def test_icc_is_high_for_near_identical_raters():
    subjects = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    rater_a = subjects + np.array([0.1, -0.1, 0.05, -0.05, 0.0])
    rater_b = subjects.copy()
    assert agreement.icc(rater_a, rater_b) > 0.95


def test_cmc_is_high_for_consistent_strides():
    base = np.sin(np.linspace(0, 2 * np.pi, 50))
    strides_a = np.stack([base + np.random.default_rng(i).normal(0, 0.01, 50) for i in range(5)])
    strides_b = np.stack([base + np.random.default_rng(i + 100).normal(0, 0.01, 50) for i in range(5)])
    assert agreement.cmc(strides_a, strides_b) > 0.9


def test_compute_all_shape_mismatch_raises():
    with pytest.raises(ValueError):
        compute_all(np.zeros(10), np.zeros(20))


def test_compute_all_without_strides_leaves_agreement_fields_none(sine_pair):
    exo, gatech = sine_pair
    result = compute_all(exo, gatech)
    assert result.cmc is None
    assert result.bland_altman_bias is None
    assert result.icc is None


def test_compute_all_with_strides_fills_agreement_fields(sine_pair):
    exo, gatech = sine_pair
    strides = np.stack([gatech] * 5)
    result = compute_all(exo, gatech, exo_strides=strides, gatech_strides=strides)
    assert result.cmc is not None
    assert result.icc is not None
