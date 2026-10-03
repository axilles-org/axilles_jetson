"""Synthetic level-walking joint trajectories (for validating the pipeline)."""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline

# (gait-cycle phase, value in degrees), heel strike at phase 0
_KEYS = {
    "hip_flexion": [(0, 30), (0.2, 18), (0.5, -10), (0.62, -2), (0.85, 32), (1, 30)],
    "knee_angle": [(0, 5), (0.15, 18), (0.4, 5), (0.6, 35), (0.72, 62), (0.87, 30), (1, 5)],
    "ankle_angle": [(0, 0), (0.07, -6), (0.45, 10), (0.62, -18), (0.75, -5), (0.9, 2), (1, 0)],
    "hip_adduction": [(0, 0), (0.15, 7), (0.5, -3), (0.7, -6), (1, 0)],
    "hip_rotation": [(0, -3), (0.3, 3), (0.6, -5), (1, -3)],
    "subtalar_angle": [(0, 3), (0.2, -4), (0.6, 2), (1, 3)],
}


def _periodic(keys):
    x, y = np.array(keys, float).T
    return CubicSpline(x, y, bc_type="periodic")


def walking(duration=30.0, fs=1000.0, speed=1.25, stride_time=1.1, variability=0.08, seed=0):
    """Returns t and {coordinate: radians or metres} with slowly varying cadence
    and stride-to-stride amplitude jitter (keeps the motion non-degenerate)."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration, 1 / fs)
    slow = lambda: np.interp(t, np.linspace(0, duration, int(duration / 3) + 2),
                             rng.normal(0, 1, int(duration / 3) + 2))
    cadence = (1 / stride_time) * (1 + 0.5 * variability * slow())
    phase = np.cumsum(cadence) / fs
    amp = lambda: 1 + variability * slow()

    c = {}
    for side, shift in (("r", 0.0), ("l", 0.5)):
        ph = (phase + shift) % 1.0
        for name, keys in _KEYS.items():
            c[f"{name}_{side}"] = np.radians(_periodic(keys)(ph) * amp())
    tw = 2 * np.pi * phase
    c["pelvis_tilt"] = np.radians(2 * np.sin(2 * tw) * amp())
    c["pelvis_list"] = np.radians(4 * np.sin(tw + 0.3) * amp())
    c["pelvis_rotation"] = np.radians(5 * np.sin(tw - 0.4) * amp() + 10 * slow())
    c["lumbar_extension"] = np.radians(-3 + 1.5 * np.sin(2 * tw + 1.0) * amp())
    c["lumbar_bending"] = np.radians(-3 * np.sin(tw + 0.2) * amp())
    c["lumbar_rotation"] = np.radians(-4 * np.sin(tw - 0.2) * amp())
    v = speed * cadence * stride_time
    c["pelvis_tx"] = np.cumsum(v) / fs + 0.01 * np.sin(2 * tw)
    c["pelvis_ty"] = 0.93 + 0.02 * np.cos(2 * tw)
    c["pelvis_tz"] = 0.02 * np.sin(tw)
    return t, c
