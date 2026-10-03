"""Torque-prediction metrics, in physical units (N·m/kg).

All model outputs and targets are z-scored during training. These helpers invert
the z-score (via the fitted ``scaler_y``) and mass-normalise so errors are
comparable across subjects of different body mass.
"""
from __future__ import annotations

import numpy as np
import torch


def _to_nm_per_kg(t: torch.Tensor | np.ndarray, scaler_y, mass_kg: float) -> np.ndarray:
    """(C, T) z-scored tensor  ->  (T, C) array in N·m/kg."""
    arr = t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)
    arr = arr.T                                   # (T, C)
    arr_nm = scaler_y.inverse_transform(arr)
    return arr_nm / mass_kg


def rmse_nm_per_kg(pred, target, scaler_y, mass_kg: float) -> float:
    p = _to_nm_per_kg(pred, scaler_y, mass_kg)
    g = _to_nm_per_kg(target, scaler_y, mass_kg)
    return float(np.sqrt(np.mean((p - g) ** 2)))


def mae_nm_per_kg(pred, target, scaler_y, mass_kg: float) -> float:
    p = _to_nm_per_kg(pred, scaler_y, mass_kg)
    g = _to_nm_per_kg(target, scaler_y, mass_kg)
    return float(np.mean(np.abs(p - g)))


def normalized_mae(pred, target, scaler_y, mass_kg: float) -> float:
    """MAE divided by the peak-to-peak range of the target (per channel), averaged."""
    p = _to_nm_per_kg(pred, scaler_y, mass_kg)
    g = _to_nm_per_kg(target, scaler_y, mass_kg)
    mae = np.mean(np.abs(p - g), axis=0)
    p2p = np.clip(g.max(axis=0) - g.min(axis=0), 1e-6, None)
    return float(np.mean(mae / p2p))


def r2_score(pred: np.ndarray, target: np.ndarray) -> float:
    ss_res = np.sum((pred - target) ** 2)
    ss_tot = np.sum((target - target.mean()) ** 2) + 1e-8
    return float(1.0 - ss_res / ss_tot)


def batch_metrics(outputs: torch.Tensor, targets: torch.Tensor, trial_names,
                  scaler_y, subject_mass: dict[str, float], last_n: int = 0) -> dict[str, float]:
    """Mean RMSE / MAE / normalized-MAE (N·m/kg) over a batch.

    ``last_n`` > 0 restricts the metric to the final ``last_n`` timesteps, matching
    what deployment consumes.
    """
    if last_n and last_n > 0:
        outputs = outputs[..., -last_n:]
        targets = targets[..., -last_n:]

    pred = outputs.detach().cpu().numpy()          # (B, 1, T)
    true = targets.detach().cpu().numpy()
    mass = np.array([subject_mass[n.split("_")[0]] for n in trial_names])[:, None, None]

    scale = scaler_y.scale_[0]
    mean = scaler_y.mean_[0]
    pred_nm = (pred * scale + mean) / mass
    true_nm = (true * scale + mean) / mass
    err = pred_nm - true_nm

    rmse = np.sqrt((err ** 2).mean())
    mae = np.abs(err).mean()
    p2p = np.clip(true_nm.max(axis=2) - true_nm.min(axis=2), 1e-6, None)
    nmae = (np.abs(err).mean(axis=2) / p2p).mean()
    return {"rmse_nm_per_kg": float(rmse), "mae_nm_per_kg": float(mae),
            "normalized_mae": float(nmae)}
