# Transformation results: Georgia Tech vs Axilles exo

Georgia Tech foot/shank IMU data, transformed with **version 1** (average transform)
and **version 2** (each subject's own transform), compared with the exo's own walking
recordings:
- **Sept 11:** 2 × 60 s, 89 strides
- **Apr 4:** 2 km/h, 5 × 10 s, 28 strides

| Folder | Axes |
|---|---|
| `exo_frame/` | exo frame: x forward, y left, z up |
| `imu_raw_frame/` | your exo IMUs' raw sensor axes (Sept 11 mounting) |


## Files (in each folder)

| File | What it shows |
|---|---|
| `version1_average_transform.png` | All subjects (average transform) vs exo: stride-averaged curves, 0% = heel strike, Georgia Tech mean ± SD band |
| `version2_own_transform_overview.png` | The same with each subject's own transform |
| `version2_per_subject_plots/<AB>.png` | One subject (own transform, mean ± SD over strides) vs exo |
| `version1_match_heatmap.png` | Correlation per subject × channel, average transform |
| `version2_match_heatmap.png` | The same, own transform |
| `heatmap_v1_vs_v2.png` | Both heatmaps side by side, plus the difference (version 2 − version 1) |
| `match_metrics.csv` | r and RMSE per version, subject and channel (includes a rotation-only baseline) |
| `match_summary.json` | Median over subjects |
| `evaluation_<version>.csv` | Exo vs the Georgia Tech group, per channel: r, RMSE, NRMSE, gain, offset, % of cycle inside the group's ±2 SD band, leave-one-out ratio |
| `evaluation_<version>_rotation.json` | Rotation error in degrees: mid-stance gravity direction and dominant rotation axis |
| `mean_cycles.npz` | All stride-averaged curves used above |

## Reading the heatmaps

- **Cell:** Pearson correlation between one subject's stride-averaged curve and the
  exo's (Sept 11), for one channel. It measures **shape and timing**, not offset or
  amplitude.
- **Colours:** blue = same shape, white = unrelated, red = opposite.
- **Small channels:** for low-amplitude channels (e.g. gyro x/z in the exo frame)
  correlation is unreliable; a few degrees of misalignment changes their shape.

## Summary (exo frame, median over 21 subjects)

AB06 is excluded: no usable treadmill shank trials.

| | Version 1 (average) | Version 2 (own) | Rotation only (no lever arm) |
|---|---|---|---|
| Walking channels r: foot acc x/z, foot gyro y, shank acc z, shank gyro y | 0.91–0.98 | 0.91–0.98 | foot acc z 0.61, shank acc z 0.81 |
| Rotation-axis error | ~10° | ~5–6° | — |
| Mid-stance tilt (foot / shank) | 12° / 5° | 12° / 7° | — |

Leave-one-out ratio for the walking channels: 1.2–1.9. A value ≤ 1 would mean the exo
looks like "one more Georgia Tech subject".

What this shows:
- **Both versions match the walking channels equally well.** Version 2 only aligns
  the rotation axis better.
- **The lever-arm correction clearly helps** the vertical accelerations.
- **Exo-specific differences:**
  - shank acc x has a dip after toe-off that no Georgia Tech subject shows
  - lower push-off foot rotation (slower walking)
  - your IMUs update at only ~20–30 Hz

Rotation errors in degrees are identical in both frames. Correlations differ between
frames because the channels mix axes differently.
