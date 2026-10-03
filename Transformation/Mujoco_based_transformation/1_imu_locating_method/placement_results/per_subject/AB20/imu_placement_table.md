# AB20 IMU placement

Estimated from mocap IK + raw IMU data with the MuJoCo model converted from `AB20.osim`. Median over trials (treadmill, levelground, ramp, stair).

Position is in each segment's own frame, in cm: **x forward, y left, z up**, measured from the segment origin. IQR is the spread between trials.

## Position

| IMU | Segment (origin) | Trials used | Position (x, y, z) cm | Trial spread IQR (x, y, z) cm | Place on the body |
|---|---|---|---|---|---|
| Foot | `calcn_r` (heel) | 23/24 | (13.1, 0.7, 2.1) | (0.7, 1.5, 0.4) | Top of the foot, 73% of the way from heel to toe joint (midfoot/laces), centred |
| Shank | `tibia_r` (knee centre) | 24/24 | (3.1, 0.8, -25.4) | (0.4, 0.8, 2.6) | Front of the shin, 54% of the way down, centred |
| Thigh | `femur_r` (hip centre) | 8/24 | (10.5, 7.3, -16.9) | (1.5, 0.6, 1.9) | Front of the thigh, 45% of the way down |
| Trunk | `torso` (lumbar joint) | 9/24 | not reliable | — | Not reliable (lumbar locked in IK, no torso markers) |

## Orientation (sensor axes in the segment frame)

| IMU | Sensor axes | Quaternion w, x, y, z (sensor → segment) | Orientation spread | Clock offset* |
|---|---|---|---|---|
| Foot | x forward, ~29° down the instep; y left (medial); z out of the top of the foot | 0.9414, 0.2039, 0.2652, 0.0437 | 4.7° | -8.7 ms |
| Shank | x down the shank; y left (medial); z out of the shin, turned 6° medially | 0.6778, 0.0438, 0.7258, 0.1094 | 2.5° | -14.6 ms |
| Thigh | x up the thigh; y backward (into the thigh); z right (lateral) | 0.4159, 0.3848, -0.5500, 0.6136 | 5.3° | -1.6 ms |
| Trunk | not reliable | — | — | -37.0 ms |

*Clock offset: t_mocap = t_imu + offset, so negative means the IMU samples lag the mocap.

## Foot and shank by activity

| IMU | Activity | Trials | Position (x, y, z) cm | Trial-to-trial SD (x, y, z) cm |
|---|---|---|---|---|
| Foot | treadmill | 6 | (13.7, -0.3, 2.0) | (0.3, 0.2, 0.2) |
| Foot | levelground | 6 | (13.4, 0.1, 1.9) | (0.4, 0.7, 0.4) |
| Foot | ramp | 6 | (12.6, 1.1, 2.9) | (0.5, 0.3, 0.4) |
| Foot | stair | 6 | (13.0, 1.4, 2.4) | (0.3, 0.4, 0.2) |
| Shank | treadmill | 6 | (3.1, 0.4, -24.9) | (0.1, 0.3, 0.1) |
| Shank | levelground | 6 | (2.8, 1.3, -24.1) | (0.4, 0.7, 0.4) |
| Shank | ramp | 6 | (3.1, 0.4, -26.5) | (0.2, 0.5, 0.6) |
| Shank | stair | 6 | (3.3, 1.2, -28.0) | (0.4, 0.5, 0.3) |

## Notes

- **Mounting orientation is constant**: it is how the sensor sits on its own segment. The shank IMU is above the ankle, so ankle motion does not affect it; the foot IMU turns with the foot.
- **Foot pitch drifts with ankle angle** (checked on AB09): up to ~±10° at the extremes (push-off, deep dorsiflexion), likely shoe/midfoot bending that the rigid model foot does not capture.
- **Rejected trials**: trials whose accelerometer the model cannot explain (residual RMS > 0.65 × signal RMS: saturation at ±8 g / ±16 rad/s, damped or corrupted recordings) are left out; see `summary.json` → `rejected_trials`.
- **Thigh accelerometer is sign-inverted** relative to its gyro in this dataset; multiply it by −1.
- Frames: (x, y, z)_MuJoCo = (x, −z, y)_OpenSim.
