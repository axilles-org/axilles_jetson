"""Peak and range features.

The peak is the extremum in the reference (``gatech``) waveform's dominant direction:
the sign of its largest-magnitude sample. Both waveforms are measured in that same
direction, so a negative push-off moment is compared with a negative peak.
"""
from __future__ import annotations

import numpy as np


def peak_sign(reference: np.ndarray) -> float:
    return 1.0 if reference[np.argmax(np.abs(reference))] >= 0 else -1.0


def peak_index(x: np.ndarray, sign: float) -> int:
    return int(np.argmax(sign * x))


def peak_values(x: np.ndarray, sign: float, axis: int = -1) -> np.ndarray:
    return sign * np.max(sign * x, axis=axis)


def peak_value_error(exo: np.ndarray, gatech: np.ndarray) -> float:
    s = peak_sign(gatech)
    return float(abs(peak_values(exo, s) - peak_values(gatech, s)))


def peak_value_pct_error(exo: np.ndarray, gatech: np.ndarray) -> float:
    s = peak_sign(gatech)
    ref = peak_values(gatech, s)
    if abs(ref) < 1e-9:
        return float("nan")
    return float(100.0 * abs(peak_values(exo, s) - ref) / abs(ref))


def peak_timing_error_pct(exo: np.ndarray, gatech: np.ndarray) -> float:
    """Difference in peak location, as a percentage of gait cycle (array length)."""
    s = peak_sign(gatech)
    return float(100.0 * abs(peak_index(exo, s) - peak_index(gatech, s)) / len(gatech))


def rom_error(exo: np.ndarray, gatech: np.ndarray) -> float:
    exo_rom = exo.max() - exo.min()
    gatech_rom = gatech.max() - gatech.min()
    return float(abs(exo_rom - gatech_rom))
