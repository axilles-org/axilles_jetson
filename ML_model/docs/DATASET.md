# Dataset — schema and contribution guide

The canonical dataset is a **Hugging Face dataset repo**
([`aicognition/mrsd-exo-ankle`](https://huggingface.co/datasets/aicognition/mrsd-exo-ankle),
pinned by revision in `configs/*.yaml`). It holds raw sensor signals in Parquet;
all feature engineering and z-scoring happen locally and reproducibly via
`scripts/prepare.py`.

## Layout

```
metadata.parquet                              one row per (subject, trial)
subjects/<subject>/<trial>__imu.parquet
subjects/<subject>/<trial>__id.parquet
subjects/<subject>/<trial>__fp.parquet
subjects/<subject>/<trial>__gon.parquet
subjects/<subject>/<trial>__gcRight.parquet
```

Each sensor file: a `time_s` column plus that sensor's channels, **at native
sample rate** (no resampling, no filtering — the full-resolution signal).

## Sensors

| Sensor | Rate (Hz) | Channels | Notes |
|---|---|---|---|
| `imu` | 200 | `{foot,shank,thigh,trunk}_{Accel,Gyro}_{X,Y,Z}` | m/s², rad/s |
| `id` | 200 | inverse-dynamics joint moments (N·m) | target: `ankle_angle_r_moment` |
| `fp` | 1000 | `Treadmill_{R,L}_{vx,vy,vz,px,py,pz,moment_x,moment_y,moment_z}` | force plate; N, m, N·m |
| `gon` | 1000 | `{ankle,knee,hip}_{sagittal,frontal}` | goniometer, **degrees** |
| `gcRight` | 200 | `HeelStrike`, `ToeOff` | percent-of-cycle (0–100), wraps at the event |

## metadata.parquet columns

`subject`, `trial`, `terrain`, `n_imu_samples`, `age`, `gender`, `height_m`,
`weight_kg`, `speed_mean_mps`, `trialstarts_s`, `trialends_s`.

## Prediction target and conventions

- Target: `id.ankle_angle_r_moment`, right-ankle plantarflexion/dorsiflexion
  moment in **N·m**. **Plantarflexion (push-off) is negative.**
- Ground truth is produced by OpenSim inverse dynamics (GaTech data) or a 2-D
  Newton–Euler calculation from treadmill GRF + foot IMU (exo data).
- The training pipeline resamples to 100 Hz (`ingest.target_rate_hz`), anti-alias
  filtered, and derives a binary `stance` channel from `gcRight`.

## Provenance

| Subjects | Source | Collection |
|---|---|---|
| AB06–AB30 | Camargo et al. 2021 (GaTech EPIC lab), treadmill trials | no exoskeleton, able-bodied, CC BY 4.0 |
| (future) exo_* | MRSD ankle exoskeleton | instrumented treadmill, with assistance |

### Attribution (CC BY 4.0)

Derived from Camargo, Jonathan (2021), *"A comprehensive, open-source dataset of
lower limb biomechanics ... Part 1 of 3"*, Mendeley Data V2,
DOI [10.17632/fcgm3chfff.2](https://doi.org/10.17632/fcgm3chfff.2), under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode).

**Changes made:** kept only the `treadmill` condition; kept only the `imu`, `id`,
`fp`, `gon`, `gcRight` streams (dropped `emg`, `markers`, `ik`, `ik_offset`, `jp`,
`gcLeft`); reorganised the per-sensor CSV tree into per-(subject, trial, sensor)
Parquet with `Header` renamed to `time_s`; consolidated demographics and trial
conditions into `metadata.parquet`. No signal values, rates, or units were
altered. Not endorsed by Georgia Tech or the original authors. Third-party
content within the original dataset may require separate permission.

## Contributing new data

1. Convert recordings to the layout above (same column names, native rates,
   `time_s` in seconds). Add rows to `metadata.parquet`.
2. Keep the sign and unit conventions identical.
3. `python scripts/upload_dataset.py --src <dir> --repo aicognition/mrsd-exo-ankle`
   then tag a new revision (`v1.1`, …) and update `configs/*.yaml`.
4. Update the provenance table above.

## Citation

Camargo, J., Ramanathan, A., Flanagan, W., Young, A. (2021). *A comprehensive,
open-source dataset of lower limb biomechanics in multiple conditions of stairs,
ramps, and level-ground ambulation and transitions.* Journal of Biomechanics 119.
