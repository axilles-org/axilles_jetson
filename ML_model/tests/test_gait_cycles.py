"""Heel-strike detection and stride time-normalisation."""
import numpy as np

from exo.data.gait_cycles import (
    N_PHASE,
    heel_strikes_from_phase,
    heel_strikes_from_stance,
    stride_curves,
)

FS = 100


def _walk(n_strides: int = 5, stride_s: float = 1.0):
    n = int(stride_s * FS)
    phase = np.tile(np.linspace(0, 100, n, endpoint=False), n_strides)
    stance = (phase < 60).astype(float)
    signal = np.sin(2 * np.pi * phase / 100)
    return phase, stance, signal


def test_heel_strikes_agree_between_phase_and_stance():
    phase, stance, _ = _walk()
    assert np.array_equal(heel_strikes_from_phase(phase), heel_strikes_from_stance(stance))


def test_stride_curves_shape_and_content():
    phase, stance, signal = _walk()
    curves = stride_curves(signal, heel_strikes_from_stance(stance), FS)
    assert curves.shape == (3, N_PHASE)
    assert np.allclose(curves[0], curves[-1])


def test_multichannel_and_rejected_strides():
    phase, stance, signal = _walk(stride_s=0.3)
    assert stride_curves(signal, heel_strikes_from_stance(stance), FS).shape == (0, N_PHASE)
    _, stance, signal = _walk()
    two = np.stack([signal, -signal], 1)
    curves = stride_curves(two, heel_strikes_from_stance(stance), FS)
    assert curves.shape == (3, N_PHASE, 2)
    assert np.allclose(curves[..., 0], -curves[..., 1])
