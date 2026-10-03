# How the Georgia Tech IMUs were located

The Camargo et al. (2021) Georgia Tech lower-limb dataset gives raw IMU data (foot,
shank, thigh, trunk; right side) and motion capture, but not exactly where or how
each IMU was mounted. We recover each IMU's **segment, position and orientation**
from the data itself, per subject.

## Idea

A rigidly attached IMU at position **r** (segment frame) with orientation **Q**
(sensor → segment) measures:

| Sensor | Model | What it gives |
|---|---|---|
| Gyroscope | `gyro = Qᵀ ω + b_g` | orientation `Q` (independent of position); `‖ω‖` also gives the clock offset and which segment it's on |
| Accelerometer | `acc = Qᵀ (f + α × r + ω × (ω × r)) + b_a` | position `r` (linear in `r` once `Q` is known) |

- **ω, α:** the segment's angular velocity and angular acceleration.
- **f:** the specific force (acceleration minus gravity) at the segment origin.
- **b_g, b_a:** sensor biases.

The motion capture provides ω, α and f for every segment, through a MuJoCo model
driven by the dataset's inverse kinematics.

## Pipeline (per subject)

1. **MuJoCo model from the subject's own OpenSim model** (`imu_locator/osim.py`).
   - **What's kept:** the scaled `.osim`'s joint centres, its joint axes in the same
     order (oblique ankle and subtalar axes included), and the knee translation
     splines (as dependent slide joints).
   - **Validation:** driven by the IK angles, the model reproduces the measured
     markers to **0.9–1.5 cm RMS**, the normal IK residual.
2. **Segment kinematics.** Play the IK through the model and read each segment's ω,
   α and f. Both mocap and IMU are low-passed at 6 Hz (zero-phase).
3. **Clock offset.** Cross-correlate |gyro| with |ω|, refined to a fraction of a sample.
4. **Segment identification.** Fit every candidate segment and keep the best gyro fit.
   All 22 subjects' foot/shank IMUs were found on `calcn_r`/`tibia_r` independently.
5. **Orientation.** A Wahba/SVD fit using two sets of vector pairs at once:
   - gyro ↔ ω
   - low-passed accel ↔ f (the gravity direction)

   Gyro alone is not enough: walking is nearly a single-axis rotation, which leaves
   rotation *about* the flexion axis undetermined (20–50° errors in practice).
6. **Accelerometer sign.** Fit both signs and keep the one with the smaller residual,
   decided once per subject per IMU.
   - Foot and shank are normal in all subjects.
   - The thigh accelerometer is inverted relative to its gyro in most subjects, but
     normal for AB21, AB25, AB27, AB28 and AB30.
7. **Position.** Linear least squares for `r` and `b_a`. For the lever-arm terms,
   ω and α come from the IMU's own gyro, not the mocap; noisy mocap ω biases
   `ω × (ω × r)` by several cm otherwise.
8. **Joint refinement.** Nonlinear least squares over (Q, r, b_g, b_a).
9. **Saturation masking.** Frames within 0.15 s of the sensor range (±8 g, ±16 rad/s)
   are excluded.
10. **Quality gate.** A trial is rejected if:
    - the accelerometer is unexplained (residual RMS > 0.65 × signal RMS)
    - the gyro fit is much worse than typical (> 1.5 × the median)
    - the gyro is *implausibly* perfect (relative residual < 0.12). Normal strapped
      IMUs give 0.14–0.40. Such trials (in AB10–AB12, AB15, AB16, AB25, AB30) are
      likely not raw recordings.
    - the orientation is > 45° from the subject's other trials
11. **Aggregate.** Median position and mean rotation over the kept trials (24 trials
    per subject: 6 each of treadmill, level ground, ramp, stairs). The spread between
    trials is the uncertainty.

## Validation

- **Synthetic** (`validation_synthetic/`): simulated walking with IMUs at known poses,
  realistic noise, biases and a 137 ms clock offset.
  - Recovered every segment, the clock offset to < 0.3 ms, positions to ≤ 0.3 cm and
    orientations to ≤ 0.13°.
  - With 3× the mocap noise: ≤ 1 cm.
- **Real data:** fit residuals explain about 70–80% of the signal RMS. The rest is
  soft tissue, shoe bending, and motion the 1-DOF knee cannot represent. Trial-to-trial
  spread is typically 0.2–1.5 cm (IQR) and 1–4°.

## Results (22 subjects, `placement_results/`)

Positions are in segment frame, cm (x forward, y left, z up):

| IMU | Origin | Position, mean ± SD | Relative to segment | Location |
|---|---|---|---|---|
| Foot | heel (`calcn_r`) | (12.8 ± 1.0, 0.3 ± 0.8, 2.3 ± 0.7) | 72 ± 5% heel → toe joint | top of the midfoot (laces) |
| Shank | knee centre (`tibia_r`) | (2.6 ± 0.8, 0.5 ± 1.5, −26.3 ± 2.1) | 61 ± 4% down the shank (≈ 17 cm above the ankle) | front of the shin |

The trunk IMU cannot be located: the IK locks the lumbar joint and there are no torso markers.

## Files

| File | What it shows |
|---|---|
| `placement_results/subject_comparison.md` / `.csv` / `.png` | All subjects compared: positions, positions scaled by segment length, orientation angles, data quality |
| `placement_results/per_subject/<AB>/imu_placement_table.md` / `.csv` | One subject: segment, position, spread, place on the body, orientation, clock offset |
| `placement_results/per_subject/<AB>/summary.json` | Full-precision result, rejected trials and reasons |
| `placement_results/per_subject/<AB>/per_trial.json` | Every trial's estimate and fit errors, for both accelerometer signs |
| `placement_results/per_subject/<AB>/fit_*.png` | Measured vs model-predicted IMU signals for one trial |
| `validation_synthetic/` | Ground truth, estimates and an example fit from the synthetic test |

Code: `imu_locator/` and `scripts/locate_camargo.py` (see the top-level README).
