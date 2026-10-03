# Data

## `exo/` — Axilles exo recordings (in Git)

| Folder | Contents |
|---|---|
| `raw_recordings/` | The original logs: `data_collection_*.csv` (Hugging Face `Amilyl/MRSD_Axilles_exo`) and `calibration/*.npz` (Sept 11 calibration session from `axilles_jetson/calibration scripts/calibration/`) |
| `exo_frame/<recording>.csv` | Walking recordings rotated into the exo frame (x forward, y left, z up) |
| `imu_raw_frame/<recording>.csv` | The same in the raw IMU axes of the Sept 11 mounting. April recordings are rotated into those axes; they were mounted 6–10° differently |

Processed CSV columns:
- `time_s`
- `foot_ax..foot_gz`, `shank_ax..shank_gz` (m/s², rad/s)
- `ankle_encoder_deg`, `toe_fsr_raw`, `heel_fsr_raw`

Processing: held (repeated) IMU samples removed and the rest linearly re-interpolated
to 200 Hz. No filtering. The exo IMUs only update at ~20–30 Hz.

Walking recordings:
- `calibration_20260911_175501_walking` and `walk_20260911_183138`: 60 s each
- the five `data_collection_20260404_*` clips: 2 km/h treadmill, 10 s each

The September 2026 `data_collection_*` logs are static (no walking).

## `georgia_tech_transformed/` — transformed Georgia Tech dataset (NOT in Git, ~4.2 GB)

Every trial of every activity for 22 subjects: 3,147 trials (157 treadmill, 680 level
ground, 1,394 ramp, 916 stair), in 4 variants:

```
georgia_tech_transformed/<frame>/<version>/<subject>/<mode>/<trial>.parquet
    frame   = exo_frame | imu_raw_frame
    version = v1_average | v2_own
```

Parquet columns:
- `time_s`
- `foot_ax..foot_gz`, `shank_ax..shank_gz` (transformed; m/s², rad/s; unfiltered)
- `heel_strike_pct`, `toe_off_pct`: Georgia Tech gait-cycle labels, 0–100%
- `foot_saturated`, `shank_saturated`: True where the raw sample is at the ±8 g /
  ±16 rad/s sensor range

`index.csv` (in Git) lists every trial with duration, saturation percentage and the
placement analysis's quality verdict per IMU. The verdict is one of:
- `ok`
- `rejected: <reason>`
- `not checked`: only 24 trials per subject were analysed

Check the `rejected` trials before training. Some have IMU data that matches the mocap
implausibly well and may not be raw recordings.

Regenerate with (needs the raw Camargo subject folders, see the top-level README):

```bash
python scripts/export_transformed_dataset.py --out 5_data --camargo-root data/camargo \
    --placement-dir 1_imu_locating_method/placement_results/per_subject \
    --transforms-exo 3_transformation_matrices/exo_frame --transforms-raw 3_transformation_matrices/imu_raw_frame \
    --exo-dir 5_data/exo/raw_recordings --calib-dir 5_data/exo/raw_recordings/calibration
```
