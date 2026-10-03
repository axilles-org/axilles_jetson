# Domain Randomization for the Deployment Gap

The model is trained on Georgia Tech treadmill data (unpowered walking, lab
sensors) and deployed on a powered exo with different hardware. Five training-time
perturbations narrow that gap by making the deployment nuisances part of the
training distribution. All are **train-split only**, applied on the z-scored
`(C, T)` window, and all default to **0 (off)** so the baseline is unchanged.

Config: `train.augment.*` in `configs/default.yaml`. Implemented in
`src/exo/data/augment.py` (`Augmenter`) and, for latency, in
`src/exo/data/dataset.py` (`WindowDataset.__getitem__`).

---

## DR-1  Latency  (`latency_samples`)

**Deploys as:** the sensor → window → inference → CAN chain lags real time by tens
of ms, so the model's input window ends slightly in the past relative to the
moment it is predicting.

**Perturbation:** shift the input window back a random `k ∈ [0, latency_samples]`
frames while keeping the target aligned to the true window end. The model learns
"predict the ankle moment `k` samples ahead of this history" for a range of `k`,
so a fixed deployment lag lands in-distribution. Combined with the last-`N`-step
loss, this directly trains the quantity the exo consumes.

Suggested: `5` (≈ 50 ms at 100 Hz).

## DR-2  Sensor-frame error  (`imu_rotation_deg`)

**Deploys as:** the `SensorAdapter` rotation is *fitted* from one walking trial,
not measured — residuals were ~1–2 inter-subject std on the accel channels, plus
a ~5–8 % phase lag, and it can drift if the exo is re-donned.

**Perturbation:** one random rotation of angle `≤ imu_rotation_deg` about a random
axis, applied per IMU accel/gyro triad (foot, shank), in physical units
(de-z-score → rotate → re-z-score — norm-preserving). The model learns not to
over-rely on a perfectly aligned frame.

Suggested: `8.0` degrees.

## DR-3  Encoder mounting  (`encoder_offset_rad`)

**Deploys as:** the ankle encoder's zero pose is calibrated approximately; a few
degrees of bias is realistic.

**Perturbation:** a random additive offset `∈ ±encoder_offset_rad` on the
`gon_ankle_sagittal` channel (zero-mean over draws).

Suggested: `0.05` rad (≈ 3°).

## DR-4  Stance-edge timing  (`stance_jitter_samples`)

**Deploys as:** the `stance` bit comes from FSR thresholds at deploy vs
gait-phase events in training; the on/off edges can be a few samples early or
late.

**Perturbation:** circularly shift the `stance` 0/1 channel by `±k`,
`k ∈ [0, stance_jitter_samples]`, then snap back to its two z-scored levels.

Suggested: `3` samples.

## DR-5  "Assistance felt"  (`assist_perturb`)

**Deploys as:** the model is trained on *unpowered* gait, but at deploy the exo
applies plantarflexion assistance, which changes the wearer's own kinematics —
they push off less, the ankle-angle trajectory shifts, the foot IMU signals
change. If the model keeps predicting the unassisted moment while the controller
assists a fraction of it, the loop can over- or under-assist.

**Perturbation:** draw a random assist level `a ∈ [0, assist_perturb]`; in the
late-stance (push-off) region of the window, reduce the foot gyro peak by up to
`a` and bias the ankle angle toward dorsiflexion proportionally. The model learns
to give a consistent prediction whether or not assistance is active.

Suggested: `0.3` (fraction).

**The principled alternative — residual learning.** Instead of (or in addition
to) DR-5, train the model to predict `τ_true − τ_base`, where `τ_base` is the
TBE feedforward profile. Then `command = τ_base + ML_residual` and the feedback
is well-behaved because `τ_base` is fixed. The `model.target: absolute | residual`
config hook is designed for this; it needs `src/exo/control/tbe.py` built from the
TBE profile params (`TAU_PHASE_ARRAY`, `TAU_VAL_ARRAY`, `PEAK_TORQUE` in the
`axilles_jetson` repo). Not yet implemented.

---

## Recommended "DR profile"

```yaml
train:
  augment:
    enabled: true
    noise_std: 0.01
    imu_gain_jitter: 0.05
    latency_samples: 5
    imu_rotation_deg: 8.0
    encoder_offset_rad: 0.05
    stance_jitter_samples: 3
    assist_perturb: 0.3
```

## How to evaluate whether DR helped

DR trades a little clean-data accuracy for robustness, so compare on held-out
subjects **and** under perturbation:

1. Train two models — baseline (DR off) and DR profile.
2. `scripts/evaluate.py --run <dir> --split test` for both — expect the DR model's
   clean RMSE to be within ~10 % of baseline (a larger drop means DR is too
   aggressive).
3. Re-evaluate each on a **perturbed** test set (rotate the test IMUs 8°, add a
   50 ms lag, jitter stance) — the DR model should degrade far less.
4. Run both through `scripts/plot_mock_run.py` on the exo recording — the DR model
   should track the GaTech band more tightly and show less phase lag.

The DR model is the one to deploy; keep the baseline as the accuracy reference.
