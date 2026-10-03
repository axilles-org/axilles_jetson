# Axilles IMU transform: Georgia Tech dataset → Axilles exo

We train on the Georgia Tech lower-limb dataset (Camargo et al., 2021), but its IMUs
sit in different places and orientations than the Axilles exo's. This repo:

1. **Locates** each Georgia Tech IMU (segment, position, orientation) from the
   dataset's raw IMU data and motion capture, using a MuJoCo model of each subject.
2. **Builds MuJoCo models** showing the Georgia Tech IMUs and the Axilles exo IMUs on
   each subject.
3. **Computes the transformation** (rotation + lever arm) that turns Georgia Tech
   foot/shank IMU data into what the Axilles exo IMUs would measure, in two versions:
   - **version 1:** one average transform for all subjects
   - **version 2:** one transform per subject
4. **Compares** the transformed Georgia Tech data with the exo's own walking
   recordings, in the exo frame and in the exo IMUs' raw axes.
5. **Exports** the transformed Georgia Tech dataset (every trial, all 4 variants) and
   the exo recordings.

## Contents

| Folder | What's inside |
|---|---|
| [`1_imu_locating_method/`](1_imu_locating_method/METHOD.md) | Method write-up, per-subject IMU placement tables, cross-subject comparison, synthetic validation |
| [`2_mujoco_models/`](2_mujoco_models/README.md) | Per-subject MuJoCo models (`.xml`) with Georgia Tech + exo IMUs, and renders (`.png`) |
| [`3_transformation_matrices/`](3_transformation_matrices/README.md) | R, lever arm d and 4×4 T: version 1 (average) and version 2 (per subject), each for `exo_frame/` and `imu_raw_frame/` |
| [`4_transformation_results/`](4_transformation_results/README.md) | Comparison plots, heatmaps (v1, v2, difference), numeric evaluation: for `exo_frame/` and `imu_raw_frame/` |
| [`5_data/`](5_data/README.md) | Exo recordings in both frames (in Git); transformed Georgia Tech dataset (generated, **not in Git**) |
| `imu_locator/` | Python package: OpenSim→MuJoCo conversion, kinematics, IMU locating, transform |
| `scripts/` | Command-line tools for every step (below) |

## Frames

| Frame | Definition |
|---|---|
| **Segment frame** (Georgia Tech model) | OpenSim body frames, Z-up convention (x forward, y left, z up). Foot origin = heel (`calcn_r`); shank origin = knee joint centre (`tibia_r`) |
| **Exo frame** | x forward, y left, z up when standing; z from standing gravity, y from the walking flexion axis. Defined the same way on Georgia Tech subjects (their static standing trial) and on the exo (its calibration session) |
| **Exo IMU raw frame** | What the BNO085s output (Sept 11 mounting) |

Raw axes in the exo frame: shank x = up, y = backward, z = right. Foot x = up tilted
~28° back, y = back and down, z = right.

## Key results

- **Georgia Tech IMU placement (22 subjects).**
  - Foot IMU: top of the midfoot, 72 ± 5% of the way from heel to toe joint.
  - Shank IMU: front of the shin, 61 ± 4% of the way down (≈ 17 cm above the ankle).
  - Trial-to-trial spread: typically ≤ 1–2 cm and a few degrees.
- **Match with the exo** (stride-averaged).
  - The walking channels match for every subject (correlation 0.84–0.99; median
    0.91–0.98): foot acc x/z, foot gyro y, shank acc z, shank gyro y.
  - The lever-arm correction raises foot acc z from 0.61 to 0.94 and shank acc z
    from 0.81 to 0.92.
  - Version 2 gives the same correlations as version 1, but halves the rotation-axis
    error (~10° → ~5–6°).
- **Dataset issues found.**
  - IMU saturation at ±8 g / ±16 rad/s in fast trials.
  - A damped shank accelerometer in most AB06 trials.
  - The thigh accelerometer is sign-inverted in most subjects (normal in AB21, AB25,
    AB27, AB28, AB30).
  - Some trials (AB10–AB12, AB15, AB16, AB25, AB30) have IMU data that matches the
    mocap implausibly well; it may not be raw recordings.
- **Exo issues found.**
  - The IMUs update at only ~20–30 Hz, although the logger runs at 100–200 Hz.
  - The IMU mounting changed by 6–10° between April and September.
  - The older `exo_frame_calibration.json` (in `axilles_jetson`) has its **shank
    rotation flipped by ~170°**. Use `3_transformation_matrices/` instead.

## Assumptions to check

- **Exo geometry:** right leg, lateral side.
  - Encoder 25.94 mm forward of the shoe's heel end, 107.08 mm lateral of the shoe
    centre, at ankle-joint height.
  - Foot IMU (+32.02, +39.14, −51.98) mm and shank IMU (0, +10.97, +168.34) mm from
    the encoder.
- **Encoder position:** this puts the encoder 2.5–3.9 cm behind the anatomical ankle
  axis (see the renders). `4_transformation_results/exo_frame_encoder_on_ankle_axis/`
  shows the alternative; it changes correlations by ≤ 0.05.
- **Heel reference:** the Georgia Tech shoe heel end is taken as the heel marker
  + 7 mm.

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

**Raw Camargo data** (needed only to re-run the analysis or regenerate `5_data/`):
download the subject folders (e.g. `AB06/` with `<date>/` and `osimxml/`) from the
dataset's site and put them in `data/camargo/`. They are git-ignored.

Camargo, J., Ramanathan, A., Flanagan, W., & Young, A. (2021). *A comprehensive,
open-source dataset of lower limb biomechanics in multiple conditions of stairs, ramps,
and level-ground ambulation and transitions.* Journal of Biomechanics, 119, 110320.

## Reproducing everything

Run from this folder. Working outputs go to `results/`, which is git-ignored; the
numbered folders hold a snapshot of the results.

```bash
P=.venv/bin/python; C=data/camargo; PL=results
# 1. locate the IMUs (per subject), tables, cross-subject comparison
for s in $(ls $C); do
  $P scripts/locate_camargo.py --subject-dir $C/$s --out $PL/$s
  $P scripts/write_placement_table.py --results $PL/$s --subject $s --camargo-root $C
done
$P scripts/compare_subjects.py --subjects $(ls $C) --results $PL --camargo-root $C
# 2-4. transforms + comparison, in both frames
EXO="--exo-dir 5_data/exo/raw_recordings --calib-dir 5_data/exo/raw_recordings/calibration --placement-dir $PL --camargo-root $C"
$P scripts/exo_compare.py --frame exo $EXO --out results/exo_transform
$P scripts/exo_compare.py --frame raw $EXO --out results/exo_transform_raw_frame
for d in results/exo_transform results/exo_transform_raw_frame; do
  $P scripts/plot_heatmaps.py --dir $d
  for v in v1 v2 v1_rot_only; do $P scripts/evaluate_match.py --dir $d --version $v; done
done
for s in $(ls $C); do $P scripts/build_placement_model.py --subject $s --camargo-root $C; done
# 5. transformed dataset
$P scripts/export_transformed_dataset.py --out 5_data --camargo-root $C --placement-dir $PL \
   --transforms-exo results/exo_transform --transforms-raw results/exo_transform_raw_frame \
   --exo-dir 5_data/exo/raw_recordings --calib-dir 5_data/exo/raw_recordings/calibration
```

Other tools:
- `scripts/demo_synthetic.py`: validation on simulated data with known IMU poses.
- `scripts/batch_camargo.py`: process downloaded subject zips automatically.
