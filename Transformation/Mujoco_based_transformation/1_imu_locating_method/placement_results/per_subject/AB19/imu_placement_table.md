# AB19 IMU placement

Estimated from mocap IK + raw IMU data with the MuJoCo model converted from `AB19.osim`. Median over trials (treadmill, levelground, ramp, stair).

Position is in each segment's own frame, in cm: **x forward, y left, z up**, measured from the segment origin. IQR is the spread between trials.

## Position

| IMU | Segment (origin) | Trials used | Position (x, y, z) cm | Trial spread IQR (x, y, z) cm | Place on the body |
|---|---|---|---|---|---|
| Foot | `calcn_r` (heel) | 18/24 | (12.8, 0.3, 2.0) | (0.5, 0.7, 0.2) | Top of the foot, 68% of the way from heel to toe joint (midfoot/laces), centred |
| Shank | `tibia_r` (knee centre) | 16/24 | (3.3, 0.3, -22.6) | (0.6, 1.0, 0.8) | Front of the shin, 52% of the way down, centred |
| Thigh | `femur_r` (hip centre) | 8/24 | (6.5, 6.9, -20.3) | (0.9, 0.9, 1.6) | Front of the thigh, 58% of the way down |
| Trunk | `torso` (lumbar joint) | 8/24 | not reliable | — | Not reliable (lumbar locked in IK, no torso markers) |

## Orientation (sensor axes in the segment frame)

| IMU | Sensor axes | Quaternion w, x, y, z (sensor → segment) | Orientation spread | Clock offset* |
|---|---|---|---|---|
| Foot | x forward, ~37° down the instep; y left (medial); z out of the top of the foot | 0.9346, 0.1520, 0.3062, -0.0982 | 3.6° | -10.8 ms |
| Shank | x down the shank; y left (medial); z out of the shin, turned 9° laterally | 0.6793, 0.0988, 0.7270, -0.0164 | 1.5° | -17.9 ms |
| Thigh | x up the thigh; y backward (into the thigh); z right (lateral) | 0.5263, 0.5387, -0.5115, 0.4137 | 6.1° | -19.6 ms |
| Trunk | not reliable | — | — | -20.8 ms |

*Clock offset: t_mocap = t_imu + offset, so negative means the IMU samples lag the mocap.

## Foot and shank by activity

| IMU | Activity | Trials | Position (x, y, z) cm | Trial-to-trial SD (x, y, z) cm |
|---|---|---|---|---|
| Foot | treadmill | 6 | (12.2, -1.4, 1.7) | (0.4, 0.5, 0.3) |
| Foot | levelground | 6 | (12.3, -0.3, 1.5) | (1.8, 0.9, 4.2) |
| Foot | ramp | 6 | (12.9, 0.3, 1.9) | (0.3, 0.9, 1.0) |
| Foot | stair | 6 | (12.8, 0.5, 2.0) | (0.2, 0.3, 0.1) |
| Shank | treadmill | 6 | (4.6, 0.8, -22.0) | (0.4, 0.3, 0.5) |
| Shank | levelground | 6 | (5.6, 0.2, -21.7) | (2.2, 7.2, 1.0) |
| Shank | ramp | 6 | (3.5, 0.5, -22.9) | (0.2, 0.5, 2.4) |
| Shank | stair | 6 | (2.9, -0.2, -22.7) | (0.2, 0.5, 0.3) |

## Notes

- **Mounting orientation is constant**: it is how the sensor sits on its own segment. The shank IMU is above the ankle, so ankle motion does not affect it; the foot IMU turns with the foot.
- **Foot pitch drifts with ankle angle** (checked on AB09): up to ~±10° at the extremes (push-off, deep dorsiflexion), likely shoe/midfoot bending that the rigid model foot does not capture.
- **Rejected trials**: trials whose accelerometer the model cannot explain (residual RMS > 0.65 × signal RMS: saturation at ±8 g / ±16 rad/s, damped or corrupted recordings) are left out; see `summary.json` → `rejected_trials`.
- **Thigh accelerometer is sign-inverted** relative to its gyro in this dataset; multiply it by −1.
- Frames: (x, y, z)_MuJoCo = (x, −z, y)_OpenSim.
