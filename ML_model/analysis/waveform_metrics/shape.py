from __future__ import annotations

import numpy as np


def rmse(exo: np.ndarray, gatech: np.ndarray) -> float:
    return float(np.sqrt(np.mean((exo - gatech) ** 2)))


def mae(exo: np.ndarray, gatech: np.ndarray) -> float:
    return float(np.mean(np.abs(exo - gatech)))


def nrmse(exo: np.ndarray, gatech: np.ndarray) -> float:
    """RMSE normalized by the gatech signal's peak-to-peak range."""
    p2p = gatech.max() - gatech.min()
    if p2p < 1e-9:
        return float("nan")
    return rmse(exo, gatech) / p2p


def r2_score(exo: np.ndarray, gatech: np.ndarray) -> float:
    ss_res = np.sum((exo - gatech) ** 2)
    ss_tot = np.sum((gatech - gatech.mean()) ** 2)
    if ss_tot < 1e-9:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def pearson_r(exo: np.ndarray, gatech: np.ndarray) -> float:
    if exo.std() < 1e-9 or gatech.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(exo, gatech)[0, 1])
