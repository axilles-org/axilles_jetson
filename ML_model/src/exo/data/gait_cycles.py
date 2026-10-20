"""Split continuous signals into gait cycles at heel strikes and time-normalise them."""
from __future__ import annotations

import numpy as np

N_PHASE = 101
MIN_STRIDE_S, MAX_STRIDE_S = 0.6, 2.2


def heel_strikes_from_stance(stance: np.ndarray) -> np.ndarray:
    """Sample indices where a 0/1 stance flag rises."""
    return np.flatnonzero(np.diff(np.asarray(stance).astype(int)) == 1) + 1


def heel_strikes_from_phase(phase_pct: np.ndarray) -> np.ndarray:
    """Sample indices where a 0-100 % gait-phase signal wraps back to 0."""
    return np.flatnonzero(np.diff(np.asarray(phase_pct, float)) < -50) + 1


def stride_curves(signal: np.ndarray, heel_strikes: np.ndarray, fs: float,
                  n_phase: int = N_PHASE) -> np.ndarray:
    """(n_strides, n_phase[, channels]) of ``signal`` between consecutive heel strikes.

    Strides outside 0.6-2.2 s or containing NaN are skipped.
    """
    signal = np.asarray(signal, float)
    phase = np.linspace(0.0, 1.0, n_phase)
    lo, hi = MIN_STRIDE_S * fs, MAX_STRIDE_S * fs
    out = []
    for s, e in zip(heel_strikes[:-1], heel_strikes[1:]):
        seg = signal[s:e + 1]
        if not lo <= e - s <= hi or np.isnan(seg).any():
            continue
        t = np.linspace(0.0, 1.0, len(seg))
        out.append(np.interp(phase, t, seg) if seg.ndim == 1 else
                   np.stack([np.interp(phase, t, seg[:, c]) for c in range(seg.shape[1])], 1))
    shape = (0, n_phase) + signal.shape[1:]
    return np.asarray(out) if out else np.empty(shape)
