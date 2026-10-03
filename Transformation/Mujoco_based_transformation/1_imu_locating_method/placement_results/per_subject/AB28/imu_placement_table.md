# AB28 IMU placement

Estimated from mocap IK + raw IMU data with the MuJoCo model converted from `AB28.osim`. Median over trials (treadmill, levelground, ramp, stair).

Position is in each segment's own frame, in cm: **x forward, y left, z up**, measured from the segment origin. IQR is the spread between trials.

## Position

| IMU | Segment (origin) | Trials used | Position (x, y, z) cm | Trial spread IQR (x, y, z) cm | Place on the body |
|---|---|---|---|---|---|
| Foot | `calcn_r` (heel) | 21/24 | (12.6, 0.4, 1.6) | (0.2, 2.5, 1.2) | Top of the foot, 73% of the way from heel to toe joint (midfoot/laces), centred |
| Shank | `tibia_r` (knee centre) | 20/24 | (2.5, 0.9, -24.9) | (0.6, 0.8, 1.8) | Front of the shin, 59% of the way down, centred |
| Thigh | `femur_r` (hip centre) | 4/24 | (5.2, 5.1, -29.3) | (0.9, 2.0, 2.4) | Front of the thigh, 82% of the way down |
| Trunk | `torso` (lumbar joint) | 14/24 | not reliable | — | Not reliable (lumbar locked in IK, no torso markers) |

## Orientation (sensor axes in the segment frame)

| IMU | Sensor axes | Quaternion w, x, y, z (sensor → segment) | Orientation spread | Clock offset* |
|---|---|---|---|---|
| Foot | x forward, ~50° down the instep; y left (medial); z out of the top of the foot | 0.8845, 0.1835, 0.4282, -0.0245 | 4.7° | -6.5 ms |
| Shank | x down the shank; y left (medial); z out of the shin, turned 10° laterally | 0.7253, 0.1269, 0.6765, 0.0110 | 2.7° | -17.0 ms |
| Thigh | x up the thigh; y backward (into the thigh); z right (lateral) | 0.6170, 0.0221, 0.7866, -0.0085 | 2.0° | -20.8 ms |
| Trunk | not reliable | — | — | -30.2 ms |

*Clock offset: t_mocap = t_imu + offset, so negative means the IMU samples lag the mocap.

## Foot and shank by activity

| IMU | Activity | Trials | Position (x, y, z) cm | Trial-to-trial SD (x, y, z) cm |
|---|---|---|---|---|
| Foot | treadmill | 6 | (12.5, -2.0, 1.0) | (0.1, 0.2, 0.2) |
| Foot | levelground | 6 | (11.8, -0.8, 0.9) | (0.5, 0.4, 0.9) |
| Foot | ramp | 6 | (12.6, 0.7, 1.7) | (0.2, 0.3, 0.3) |
| Foot | stair | 6 | (12.6, 0.8, 2.4) | (0.2, 0.3, 0.2) |
| Shank | treadmill | 6 | (2.9, 0.6, -23.7) | (0.1, 0.3, 0.2) |
| Shank | levelground | 6 | (2.4, 1.1, -24.0) | (1.2, 0.9, 1.2) |
| Shank | ramp | 6 | (2.4, 1.1, -25.3) | (0.2, 0.4, 0.5) |
| Shank | stair | 6 | (2.1, 1.4, -25.9) | (0.2, 0.7, 0.2) |

## Notes

- **Mounting orientation is constant**: it is how the sensor sits on its own segment. The shank IMU is above the ankle, so ankle motion does not affect it; the foot IMU turns with the foot.
- **Foot pitch drifts with ankle angle** (checked on AB09): up to ~±10° at the extremes (push-off, deep dorsiflexion), likely shoe/midfoot bending that the rigid model foot does not capture.
- **Rejected trials**: trials whose accelerometer the model cannot explain (residual RMS > 0.65 × signal RMS: saturation at ±8 g / ±16 rad/s, damped or corrupted recordings) are left out; see `summary.json` → `rejected_trials`.
- **Thigh accelerometer is sign-inverted** relative to its gyro in this dataset; multiply it by −1.
- Frames: (x, y, z)_MuJoCo = (x, −z, y)_OpenSim.
