# waveform_metrics

Metrics for comparing an exo sensor waveform against a GaTech reference waveform
once both are time-aligned and resampled to a common axis (e.g. 0-100% gait cycle).

## Usage

```python
import numpy as np
from waveform_metrics import compute_all

# exo, gatech: 1-D arrays, same length, already transformed/aligned
metrics = compute_all(exo, gatech)
print(metrics.summary())
```

To also get cross-stride agreement metrics (CMC, Bland-Altman, ICC), pass stacks
of per-stride curves (`(n_strides, n_samples)`):

```python
metrics = compute_all(
    exo, gatech,
    exo_strides=exo_stride_stack,
    gatech_strides=gatech_stride_stack,
)
```

Without `exo_strides`/`gatech_strides`, `cmc`, `bland_altman_bias`,
`bland_altman_loa`, and `icc` are left as `None`.

Set `include_dtw=False` to skip the O(n·m) DTW distance on long signals.

## What each metric means

| Metric | File | Measures |
|---|---|---|
| `rmse`, `mae`, `nrmse` | `shape.py` | Overall magnitude of error between the two waveforms |
| `r2`, `pearson_r` | `shape.py` | How well the exo waveform's shape/trend tracks gatech's |
| `xcorr_lag` | `timing.py` | Sample offset that best aligns exo to gatech (detects residual timing mismatch) |
| `dtw_distance` | `timing.py` | Shape distance after allowing nonlinear time warping |
| `peak_value_error`, `peak_value_pct_error` | `features.py` | Difference in peak magnitude |
| `peak_timing_error_pct` | `features.py` | Difference in peak location, as % of gait cycle |
| `rom_error` | `features.py` | Difference in range of motion (max - min) |
| `cmc` | `agreement.py` | Shape agreement across repeated strides (Kadaba et al., 1989) |
| `bland_altman_bias`, `bland_altman_loa` | `agreement.py` | Systematic bias and 95% limits of agreement between paired stride peaks |
| `icc` | `agreement.py` | Reliability of exo vs. gatech as two "raters" of the same strides |

## Running the tests

```bash
cd ML_model/analysis
python3 -m pytest waveform_metrics/test_waveform_metrics.py -q
```
