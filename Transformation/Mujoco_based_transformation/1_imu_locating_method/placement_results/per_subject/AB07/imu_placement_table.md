# AB07 IMU placement

Estimated from mocap IK + raw IMU data with the MuJoCo model converted from `AB07.osim`. Median over trials (treadmill, levelground, ramp, stair).

Position is in each segment's own frame, in cm: **x forward, y left, z up**, measured from the segment origin. IQR is the spread between trials.

## Position

| IMU | Segment (origin) | Trials used | Position (x, y, z) cm | Trial spread IQR (x, y, z) cm | Place on the body |
|---|---|---|---|---|---|
| Foot | `calcn_r` (heel) | 20/24 | (14.1, 0.0, 1.7) | (0.6, 1.5, 0.2) | Top of the foot, 79% of the way from heel to toe joint (midfoot/laces), centred |
| Shank | `tibia_r` (knee centre) | 21/24 | (2.9, 0.9, -25.8) | (0.3, 1.4, 2.1) | Front of the shin, 62% of the way down, centred |
| Thigh | `femur_r` (hip centre) | 23/24 | (5.6, 4.3, -23.1) | (1.8, 1.8, 6.1) | Front of the thigh, 60% of the way down |
| Trunk | `torso` (lumbar joint) | 3/24 | not reliable | — | Not reliable (lumbar locked in IK, no torso markers) |

## Orientation (sensor axes in the segment frame)

| IMU | Sensor axes | Quaternion w, x, y, z (sensor → segment) | Orientation spread | Clock offset* |
|---|---|---|---|---|
| Foot | x forward, ~42° down the instep; y left (medial); z out of the top of the foot | 0.9187, 0.1661, 0.3577, -0.0237 | 3.8° | -9.0 ms |
| Shank | x down the shank; y left (medial); z out of the shin, turned 11° laterally | 0.7020, 0.1376, 0.6987, 0.0046 | 2.7° | -14.2 ms |
| Thigh | x up the thigh; y backward (into the thigh); z right (lateral) | 0.4599, 0.5954, -0.4185, 0.5088 | 4.5° | -12.7 ms |
| Trunk | not reliable | — | — | -22.6 ms |

*Clock offset: t_mocap = t_imu + offset, so negative means the IMU samples lag the mocap.

## Foot and shank by activity

| IMU | Activity | Trials | Position (x, y, z) cm | Trial-to-trial SD (x, y, z) cm |
|---|---|---|---|---|
| Foot | treadmill | 6 | (14.7, -1.2, 1.8) | (0.1, 0.2, 0.1) |
| Foot | levelground | 6 | (13.6, 0.9, 1.7) | (2.7, 2.0, 2.7) |
| Foot | ramp | 6 | (14.0, 0.4, 1.8) | (0.2, 0.2, 0.1) |
| Foot | stair | 6 | (13.9, -0.1, 1.8) | (0.8, 0.4, 0.2) |
| Shank | treadmill | 6 | (2.8, 2.5, -25.4) | (0.1, 0.6, 0.1) |
| Shank | levelground | 6 | (3.2, 1.5, -24.1) | (1.2, 3.2, 11.3) |
| Shank | ramp | 6 | (3.0, 0.6, -26.5) | (0.3, 1.0, 0.7) |
| Shank | stair | 6 | (2.9, 0.2, -28.0) | (0.2, 0.3, 0.4) |

## Notes

- **Mounting orientation is constant**: it is how the sensor sits on its own segment. The shank IMU is above the ankle, so ankle motion does not affect it; the foot IMU turns with the foot.
- **Foot pitch drifts with ankle angle** (checked on AB09): up to ~±10° at the extremes (push-off, deep dorsiflexion), likely shoe/midfoot bending that the rigid model foot does not capture.
- **Rejected trials**: trials whose accelerometer the model cannot explain (residual RMS > 0.65 × signal RMS: saturation at ±8 g / ±16 rad/s, damped or corrupted recordings) are left out; see `summary.json` → `rejected_trials`.
- **Thigh accelerometer is sign-inverted** relative to its gyro in this dataset; multiply it by −1.
- Frames: (x, y, z)_MuJoCo = (x, −z, y)_OpenSim.
