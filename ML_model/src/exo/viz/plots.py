"""Plotting helpers for evaluation and data inspection."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_trial_reconstruction(
    name: str,
    pred: np.ndarray,
    truth: np.ndarray,
    valid: np.ndarray,
    metrics: dict[str, float],
    out_path: str | Path,
) -> None:
    """Full-length predicted vs. ground-truth ankle moment for one trial."""
    t = np.arange(len(truth))
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.plot(t, truth, color="#1f77b4", lw=1.2, label="ground truth")
    ax.plot(t, pred, color="#d62728", lw=1.0, ls="--", label="prediction")
    if (~valid).any():
        ax.fill_between(t, truth.min(), truth.max(), where=~valid,
                        color="grey", alpha=0.12, label="no label")
    ax.set_title(f"{name}   RMSE={metrics['rmse_nm_per_kg']:.3f}  "
                 f"MAE={metrics['mae_nm_per_kg']:.3f}  R²={metrics['r2']:.3f} (Nm/kg)")
    ax.set_xlabel("sample")
    ax.set_ylabel("ankle moment (Nm/kg)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_torque_window(
    pred: np.ndarray,
    truth: np.ndarray,
    title: str,
    out_path: str | Path,
) -> None:
    """Single window, prediction vs. ground truth."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(truth, color="#1f77b4", label="ground truth")
    ax.plot(pred, color="#d62728", ls="--", label="prediction")
    ax.set_title(title)
    ax.set_xlabel("timestep")
    ax.set_ylabel("torque (Nm/kg)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
