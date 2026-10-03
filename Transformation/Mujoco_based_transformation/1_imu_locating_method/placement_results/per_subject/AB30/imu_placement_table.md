# AB30 IMU placement

Estimated from mocap IK + raw IMU data with the MuJoCo model converted from `AB30.osim`. Median over trials (treadmill, levelground, ramp, stair).

Position is in each segment's own frame, in cm: **x forward, y left, z up**, measured from the segment origin. IQR is the spread between trials.

## Position

| IMU | Segment (origin) | Trials used | Position (x, y, z) cm | Trial spread IQR (x, y, z) cm | Place on the body |
|---|---|---|---|---|---|
| Foot | `calcn_r` (heel) | 16/24 | (13.1, 0.6, 3.0) | (0.2, 0.5, 0.3) | Top of the foot, 70% of the way from heel to toe joint (midfoot/laces), centred |
| Shank | `tibia_r` (knee centre) | 17/24 | (3.4, -1.9, -28.0) | (0.4, 0.6, 1.1) | Front of the shin, 62% of the way down, lateral |
| Thigh | `femur_r` (hip centre) | 24/24 | (4.8, 1.2, -5.5) | (2.2, 3.9, 6.2) | Front of the thigh, 14% of the way down |
| Trunk | `torso` (lumbar joint) | 24/24 | not reliable | — | Not reliable (lumbar locked in IK, no torso markers) |

## Orientation (sensor axes in the segment frame)

| IMU | Sensor axes | Quaternion w, x, y, z (sensor → segment) | Orientation spread | Clock offset* |
|---|---|---|---|---|
| Foot | x forward, ~36° down the instep; y left (medial); z out of the top of the foot | 0.9319, 0.1807, 0.2988, -0.0984 | 1.4° | -8.2 ms |
| Shank | x down the shank; y left (medial); z out of the shin, turned 9° laterally | 0.7052, 0.0518, 0.7045, -0.0615 | 1.8° | -14.6 ms |
| Thigh | x up the thigh; y backward (into the thigh); z right (lateral) | 0.7111, 0.0501, 0.6957, -0.0886 | 40.2° | -31.5 ms |
| Trunk | not reliable | — | — | -73.2 ms |

*Clock offset: t_mocap = t_imu + offset, so negative means the IMU samples lag the mocap.

## Foot and shank by activity

| IMU | Activity | Trials | Position (x, y, z) cm | Trial-to-trial SD (x, y, z) cm |
|---|---|---|---|---|
| Foot | treadmill | 6 | (12.9, 0.4, 2.9) | (0.1, 0.3, 0.2) |
| Foot | levelground | 6 | (13.7, 0.1, 3.4) | (1.4, 1.6, 0.5) |
| Foot | ramp | 6 | (13.2, 0.6, 3.1) | (0.2, 0.3, 0.1) |
| Foot | stair | 6 | (12.2, -0.4, 2.8) | (2.4, 0.8, 0.7) |
| Shank | treadmill | 6 | (3.7, -2.2, -27.6) | (0.1, 0.3, 0.1) |
| Shank | levelground | 6 | (3.1, -1.2, -27.8) | (2.2, 2.9, 2.2) |
| Shank | ramp | 6 | (3.2, -1.7, -28.7) | (0.1, 0.3, 0.4) |
| Shank | stair | 6 | (2.6, -2.3, -28.3) | (1.5, 3.2, 3.2) |

## Notes

- **Mounting orientation is constant**: it is how the sensor sits on its own segment. The shank IMU is above the ankle, so ankle motion does not affect it; the foot IMU turns with the foot.
- **Foot pitch drifts with ankle angle** (checked on AB09): up to ~±10° at the extremes (push-off, deep dorsiflexion), likely shoe/midfoot bending that the rigid model foot does not capture.
- **Rejected trials**: trials whose accelerometer the model cannot explain (residual RMS > 0.65 × signal RMS: saturation at ±8 g / ±16 rad/s, damped or corrupted recordings) are left out; see `summary.json` → `rejected_trials`.
- **Thigh accelerometer is sign-inverted** relative to its gyro in this dataset; multiply it by −1.
- Frames: (x, y, z)_MuJoCo = (x, −z, y)_OpenSim.
