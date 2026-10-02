from __future__ import annotations

import numpy as np


def peak_value_error(exo: np.ndarray, gatech: np.ndarray) -> float:
    return float(np.abs(exo.max() - gatech.max()))


def peak_value_pct_error(exo: np.ndarray, gatech: np.ndarray) -> float:
    if abs(gatech.max()) < 1e-9:
        return float("nan")
    return float(100.0 * abs(exo.max() - gatech.max()) / abs(gatech.max()))


def peak_timing_error_pct(exo: np.ndarray, gatech: np.ndarray) -> float:
    """Difference in peak location, as a percentage of gait cycle (array length)."""
    exo_idx = int(np.argmax(exo))
    gatech_idx = int(np.argmax(gatech))
    return float(100.0 * abs(exo_idx - gatech_idx) / len(gatech))


def rom_error(exo: np.ndarray, gatech: np.ndarray) -> float:
    exo_rom = exo.max() - exo.min()
    gatech_rom = gatech.max() - gatech.min()
    return float(abs(exo_rom - gatech_rom))
