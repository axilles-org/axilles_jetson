from __future__ import annotations

import numpy as np


def cmc(exo_strides: np.ndarray, gatech_strides: np.ndarray) -> float:
    """Coefficient of Multiple Correlation between two (n_curves, n_samples) stacks.

    Kadaba et al. (1989), as in Ferrari et al. (2010): within-timepoint variance
    (spread of the curves around their mean curve) over total variance. 1 = all
    curves identical; NaN when the spread exceeds the total variance (no common shape).
    """
    curves = np.concatenate([exo_strides, gatech_strides], axis=0)
    n_curves, n_samples = curves.shape
    if n_curves < 2:
        return float("nan")

    within = np.sum((curves - curves.mean(axis=0)) ** 2) / (n_samples * (n_curves - 1))
    total = np.sum((curves - curves.mean()) ** 2) / (n_samples * n_curves - 1)
    if total < 1e-12 or within > total:
        return float("nan")
    return float(np.sqrt(1.0 - within / total))


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
