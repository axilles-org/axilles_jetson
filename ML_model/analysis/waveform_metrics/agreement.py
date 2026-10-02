from __future__ import annotations

import numpy as np


def cmc(exo_strides: np.ndarray, gatech_strides: np.ndarray) -> float:
    """Coefficient of Multiple Correlation between two (n_curves, n_samples) stacks.

    Kadaba et al. (1989) formulation: treats each curve (exo or gatech stride) as
    a repeated measure of the same underlying gait-cycle shape, and asks how much
    of the total variability is within-timepoint (noise) vs. across-the-cycle
    (signal). Values near 1 mean the curves trace the same shape; values near 0
    mean within-timepoint spread rivals the shape's own variation.
    """
    curves = np.concatenate([exo_strides, gatech_strides], axis=0)  # (n_curves, n_samples)
    n_curves, n_samples = curves.shape

    grand_mean = curves.mean()
    timepoint_mean = curves.mean(axis=0)  # mean curve across all strides, per sample
    curve_mean = curves.mean(axis=1)      # each curve's own mean level

    ss_timepoint = n_curves * np.sum((timepoint_mean - grand_mean) ** 2)
    ss_curve = n_samples * np.sum((curve_mean - grand_mean) ** 2)
    ss_total = np.sum((curves - grand_mean) ** 2)
    ss_error = ss_total - ss_timepoint - ss_curve

    df_error = (n_curves - 1) * (n_samples - 1)
    df_timepoint = n_samples - 1
    if df_error <= 0 or df_timepoint <= 0:
        return float("nan")

    ms_error = ss_error / df_error
    ms_timepoint = ss_timepoint / df_timepoint
    denom = ms_error + ms_timepoint
    if denom < 1e-9:
        return float("nan")
    return float(np.sqrt(max(0.0, 1.0 - ms_error / denom)))


def bland_altman(exo_values: np.ndarray, gatech_values: np.ndarray) -> tuple[float, tuple[float, float]]:
    """Bias and 95% limits of agreement between paired scalar measurements."""
    diffs = exo_values - gatech_values
    bias = float(diffs.mean())
    sd = diffs.std(ddof=1)
    loa = (bias - 1.96 * sd, bias + 1.96 * sd)
    return bias, (float(loa[0]), float(loa[1]))


def icc(exo_values: np.ndarray, gatech_values: np.ndarray) -> float:
    """ICC(2,1): two-way random-effects, single-measure intraclass correlation."""
    ratings = np.stack([exo_values, gatech_values], axis=1)  # (n_subjects, 2)
    n, k = ratings.shape

    subject_means = ratings.mean(axis=1)
    rater_means = ratings.mean(axis=0)
    grand_mean = ratings.mean()

    ss_subjects = k * np.sum((subject_means - grand_mean) ** 2)
    ss_raters = n * np.sum((rater_means - grand_mean) ** 2)
    ss_total = np.sum((ratings - grand_mean) ** 2)
    ss_error = ss_total - ss_subjects - ss_raters

    ms_subjects = ss_subjects / (n - 1)
    ms_raters = ss_raters / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))

    denom = ms_subjects + (k - 1) * ms_error + k * (ms_raters - ms_error) / n
    if abs(denom) < 1e-9:
        return float("nan")
    return float((ms_subjects - ms_error) / denom)
