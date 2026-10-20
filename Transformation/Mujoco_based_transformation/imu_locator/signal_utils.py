"""Filtering, differentiation and resampling helpers."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt


def sample_rate(t: np.ndarray) -> float:
    return 1.0 / float(np.median(np.diff(t)))


def lowpass(x: np.ndarray, fs: float, fc: float | None, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth low-pass along axis 0 (no-op if fc is None)."""
    if fc is None or fc >= 0.49 * fs:
        return np.asarray(x, float)
    b, a = butter(order, fc / (0.5 * fs))
    return filtfilt(b, a, x, axis=0)


def fill_nans(x: np.ndarray) -> np.ndarray:
    """Linear interpolation over NaN gaps, column-wise."""
    x = np.array(x, float, copy=True)
    x2 = x.reshape(len(x), -1)
    idx = np.arange(len(x))
    for j in range(x2.shape[1]):
        bad = ~np.isfinite(x2[:, j])
        if bad.all():
            x2[:, j] = 0.0
        elif bad.any():
            x2[bad, j] = np.interp(idx[bad], idx[~bad], x2[~bad, j])
    return x2.reshape(x.shape)


def make_uniform(t: np.ndarray, x: np.ndarray, fs: float | None = None):
    """Resample to a uniform grid (keeps the median rate unless fs given)."""
    fs = fs or sample_rate(t)
    tu = np.arange(t[0], t[-1] + 0.5 / fs, 1.0 / fs)
    tu = tu[tu <= t[-1]]
    return tu, interp_rows(t, x, tu)


def interp_rows(t: np.ndarray, x: np.ndarray, tq: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    flat = x.reshape(len(t), -1)
    out = np.column_stack([np.interp(tq, t, flat[:, j]) for j in range(flat.shape[1])])
    return out.reshape((len(tq),) + x.shape[1:])


def differentiate(q: np.ndarray, t: np.ndarray, fc: float | None):
    """Filtered positions, velocities and accelerations (central differences)."""
    fs = sample_rate(t)
    qf = lowpass(q, fs, fc)
    qd = np.gradient(qf, t, axis=0)
    qdd = np.gradient(qd, t, axis=0)
    return qf, qd, qdd
