"""adapter_fit: closed-form helpers recover a known synthetic transform."""
import numpy as np

from exo.deploy.adapter_fit import _kabsch, heel_strike_indices, phase_average


def _euler(rx, ry, rz):
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def test_kabsch_recovers_rotation():
    rng = np.random.default_rng(0)
    R_true = _euler(0.3, -0.5, 1.1)
    src = rng.normal(size=(200, 3))
    dst = src @ R_true.T
    R = _kabsch(src, dst, np.ones(200))
    assert np.allclose(R, R_true, atol=1e-6)


def test_kabsch_rank_deficient_falls_back_to_identity():
    # all points on one axis -> rotation about that axis is undetermined
    src = np.zeros((50, 3))
    src[:, 0] = np.linspace(-1, 1, 50)
    R = _kabsch(src, src.copy(), np.ones(50))
    assert np.allclose(R, np.eye(3))


def test_phase_average_of_sine_is_sine():
    rate = 100.0
    t = np.arange(0, 10, 1 / rate)
    period = 1.0
    sig = np.sin(2 * np.pi * t / period)
    events = np.arange(0, len(t), int(period * rate))
    avg = phase_average(sig, events)
    ref = np.sin(2 * np.pi * np.linspace(0, 1, 100))
    assert np.corrcoef(avg, ref)[0, 1] > 0.999


def test_phase_shift_rolls_the_axis():
    events = np.arange(0, 1000, 100)
    sig = np.tile(np.arange(100.0), 10)          # ramp 0..99 each cycle
    a = phase_average(sig, events)
    b = phase_average(sig, events, phase_shift=25)
    assert np.allclose(b, np.roll(a, -25))


def test_heel_strike_spacing_respected():
    rate = 100.0
    t = np.arange(0, 8, 1 / rate)
    sig = np.sin(2 * np.pi * t / 0.9)            # ~0.9 s cycles
    ev = heel_strike_indices(sig, rate, min_cycle_s=0.6)
    gaps = np.diff(ev)
    assert gaps.min() >= 0.6 * rate
    assert 7 <= len(ev) <= 10
