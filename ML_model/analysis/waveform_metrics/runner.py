from __future__ import annotations

import numpy as np

from . import agreement, features, shape, timing
from .result import WaveformMetrics


def compute_all(
    exo: np.ndarray,
    gatech: np.ndarray,
    *,
    exo_strides: np.ndarray | None = None,
    gatech_strides: np.ndarray | None = None,
    include_dtw: bool = True,
) -> WaveformMetrics:
    """Compare two aligned, same-length waveforms (e.g. one gait cycle each).

    ``exo`` and ``gatech`` are 1-D arrays of equal length, already resampled to a
    common time base (e.g. 0-100% gait cycle). ``exo_strides``/``gatech_strides``
    are optional (n_strides, n_samples) stacks of per-stride peak values, used for
    the multi-trial agreement metrics (CMC, Bland-Altman, ICC); these are skipped
    when not provided.
    """
    if exo.shape != gatech.shape:
        raise ValueError(f"shape mismatch: exo {exo.shape} vs gatech {gatech.shape}")

    multi_trial = exo_strides is not None and gatech_strides is not None
    if multi_trial:
        exo_peaks = exo_strides.max(axis=1)
        gatech_peaks = gatech_strides.max(axis=1)
        ba_bias, ba_loa = agreement.bland_altman(exo_peaks, gatech_peaks)
        cmc_value = agreement.cmc(exo_strides, gatech_strides)
        icc_value = agreement.icc(exo_peaks, gatech_peaks)
    else:
        ba_bias = ba_loa = cmc_value = icc_value = None

    return WaveformMetrics(
        rmse=shape.rmse(exo, gatech),
        mae=shape.mae(exo, gatech),
        nrmse=shape.nrmse(exo, gatech),
        r2=shape.r2_score(exo, gatech),
        pearson_r=shape.pearson_r(exo, gatech),
        xcorr_lag=timing.xcorr_lag(exo, gatech),
        dtw_distance=timing.dtw_distance(exo, gatech) if include_dtw else None,
        peak_value_error=features.peak_value_error(exo, gatech),
        peak_value_pct_error=features.peak_value_pct_error(exo, gatech),
        peak_timing_error_pct=features.peak_timing_error_pct(exo, gatech),
        rom_error=features.rom_error(exo, gatech),
        cmc=cmc_value,
        bland_altman_bias=ba_bias,
        bland_altman_loa=ba_loa,
        icc=icc_value,
    )
