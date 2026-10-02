from __future__ import annotations

import numpy as np
from scipy.signal import correlate


def xcorr_lag(exo: np.ndarray, gatech: np.ndarray) -> int:
    """Sample offset that best aligns exo to gatech (positive => exo lags)."""
    exo_c = exo - exo.mean()
    gatech_c = gatech - gatech.mean()
    corr = correlate(exo_c, gatech_c, mode="full")
    lags = np.arange(-len(gatech) + 1, len(exo))
    return int(lags[np.argmax(corr)])


def dtw_distance(exo: np.ndarray, gatech: np.ndarray) -> float:
    """Dynamic time warping distance with a Euclidean local cost."""
    n, m = len(exo), len(gatech)
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            local = abs(exo[i - 1] - gatech[j - 1])
            cost[i, j] = local + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    return float(cost[n, m])
