# AB16 IMU placement

Estimated from mocap IK + raw IMU data with the MuJoCo model converted from `AB16.osim`. Median over trials (treadmill, levelground, ramp, stair).

Position is in each segment's own frame, in cm: **x forward, y left, z up**, measured from the segment origin. IQR is the spread between trials.

## Position

| IMU | Segment (origin) | Trials used | Position (x, y, z) cm | Trial spread IQR (x, y, z) cm | Place on the body |
|---|---|---|---|---|---|
| Foot | `calcn_r` (heel) | 23/24 | (12.2, -1.5, 2.8) | (0.7, 2.4, 0.6) | Top of the foot, 69% of the way from heel to toe joint (midfoot/laces), lateral |
| Shank | `tibia_r` (knee centre) | 16/24 | (1.4, 1.2, -24.9) | (0.8, 2.1, 2.1) | Front of the shin, 60% of the way down, medial |
| Thigh | `femur_r` (hip centre) | 16/24 | (6.2, 7.3, -24.9) | (2.6, 2.2, 2.2) | Front of the thigh, 67% of the way down |
| Trunk | `torso` (lumbar joint) | 21/24 | not reliable | — | Not reliable (lumbar locked in IK, no torso markers) |

## Orientation (sensor axes in the segment frame)

| IMU | Sensor axes | Quaternion w, x, y, z (sensor → segment) | Orientation spread | Clock offset* |
|---|---|---|---|---|
| Foot | x forward, ~45° down the instep; y left (medial); z out of the top of the foot | 0.8907, 0.2340, 0.3879, -0.0371 | 2.0° | -6.4 ms |
| Shank | x down the shank; y left (medial); z out of the shin, turned 10° laterally | 0.7042, 0.1040, 0.7022, -0.0150 | 3.3° | -13.8 ms |
| Thigh | x up the thigh; y backward (into the thigh); z right (lateral) | 0.5820, 0.4804, -0.5555, 0.3491 | 7.9° | -8.8 ms |
| Trunk | not reliable | — | — | -23.7 ms |

*Clock offset: t_mocap = t_imu + offset, so negative means the IMU samples lag the mocap.

## Foot and shank by activity

| IMU | Activity | Trials | Position (x, y, z) cm | Trial-to-trial SD (x, y, z) cm |
|---|---|---|---|---|
| Foot | treadmill | 6 | (12.3, -2.9, 2.9) | (2.0, 1.8, 5.7) |
| Foot | levelground | 6 | (12.3, -2.9, 2.1) | (0.2, 0.5, 0.4) |
| Foot | ramp | 6 | (12.0, -1.4, 2.9) | (0.1, 0.2, 0.3) |
| Foot | stair | 6 | (11.1, 0.0, 3.6) | (0.2, 0.2, 0.2) |
| Shank | treadmill | 6 | (1.0, 3.4, -23.4) | (2.4, 0.5, 5.1) |
| Shank | levelground | 6 | (-3.4, 3.2, -10.6) | (0.9, 1.1, 0.4) |
| Shank | ramp | 6 | (2.2, 0.9, -24.8) | (0.5, 0.5, 1.7) |
| Shank | stair | 6 | (1.4, 1.1, -25.7) | (0.2, 0.7, 0.2) |

## Notes

- **Mounting orientation is constant**: it is how the sensor sits on its own segment. The shank IMU is above the ankle, so ankle motion does not affect it; the foot IMU turns with the foot.
- **Foot pitch drifts with ankle angle** (checked on AB09): up to ~±10° at the extremes (push-off, deep dorsiflexion), likely shoe/midfoot bending that the rigid model foot does not capture.
- **Rejected trials**: trials whose accelerometer the model cannot explain (residual RMS > 0.65 × signal RMS: saturation at ±8 g / ±16 rad/s, damped or corrupted recordings) are left out; see `summary.json` → `rejected_trials`.
- **Thigh accelerometer is sign-inverted** relative to its gyro in this dataset; multiply it by −1.
- Frames: (x, y, z)_MuJoCo = (x, −z, y)_OpenSim.
