# AB06 IMU placement

Estimated from mocap IK + raw IMU data with the MuJoCo model converted from `AB06.osim`. Median over trials (treadmill, levelground, ramp, stair).

Position is in each segment's own frame, in cm: **x forward, y left, z up**, measured from the segment origin. IQR is the spread between trials.

## Position

| IMU | Segment (origin) | Trials used | Position (x, y, z) cm | Trial spread IQR (x, y, z) cm | Place on the body |
|---|---|---|---|---|---|
| Foot | `calcn_r` (heel) | 23/24 | (14.1, 0.4, 2.7) | (0.4, 1.4, 0.3) | Top of the foot, 75% of the way from heel to toe joint (midfoot/laces), centred |
| Shank | `tibia_r` (knee centre) | 7/24 | (3.5, 1.4, -28.5) | (0.3, 0.3, 0.4) | Front of the shin, 61% of the way down, medial |
| Thigh | `femur_r` (hip centre) | 22/24 | (5.6, 5.9, -22.8) | (0.9, 3.6, 2.7) | Front of the thigh, 56% of the way down |
| Trunk | `torso` (lumbar joint) | 8/24 | not reliable | — | Not reliable (lumbar locked in IK, no torso markers) |

## Orientation (sensor axes in the segment frame)

| IMU | Sensor axes | Quaternion w, x, y, z (sensor → segment) | Orientation spread | Clock offset* |
|---|---|---|---|---|
| Foot | x forward, ~38° down the instep; y left (medial); z out of the top of the foot | 0.9349, 0.1318, 0.3283, -0.0278 | 3.6° | -7.9 ms |
| Shank | x down the shank; y left (medial); z out of the shin, turned 14° medially | 0.7086, -0.0243, 0.6902, 0.1444 | 1.2° | -13.0 ms |
| Thigh | x up the thigh; y backward (into the thigh); z right (lateral) | 0.5121, 0.5538, -0.5391, 0.3748 | 5.7° | -17.6 ms |
| Trunk | not reliable | — | — | -37.7 ms |

*Clock offset: t_mocap = t_imu + offset, so negative means the IMU samples lag the mocap.

## Foot and shank by activity

| IMU | Activity | Trials | Position (x, y, z) cm | Trial-to-trial SD (x, y, z) cm |
|---|---|---|---|---|
| Foot | treadmill | 6 | (13.9, -1.2, 2.6) | (2.9, 6.2, 3.6) |
| Foot | levelground | 6 | (14.2, 0.1, 2.5) | (0.2, 0.6, 0.2) |
| Foot | ramp | 6 | (14.3, 0.5, 2.8) | (0.2, 0.3, 0.2) |
| Foot | stair | 6 | (14.1, 0.8, 2.8) | (0.5, 0.2, 0.3) |
| Shank | treadmill | 6 | (8.4, 5.7, -9.9) | (0.1, 0.2, 0.4) |
| Shank | levelground | 6 | (3.9, 1.3, -25.7) | (1.3, 1.8, 10.2) |
| Shank | ramp | 6 | (3.4, 1.2, -28.5) | (0.2, 0.8, 7.4) |
| Shank | stair | 6 | (8.6, -2.5, -9.0) | (1.8, 1.9, 0.8) |

## Notes

- **Mounting orientation is constant**: it is how the sensor sits on its own segment. The shank IMU is above the ankle, so ankle motion does not affect it; the foot IMU turns with the foot.
- **Foot pitch drifts with ankle angle** (checked on AB09): up to ~±10° at the extremes (push-off, deep dorsiflexion), likely shoe/midfoot bending that the rigid model foot does not capture.
- **Rejected trials**: trials whose accelerometer the model cannot explain (residual RMS > 0.65 × signal RMS: saturation at ±8 g / ±16 rad/s, damped or corrupted recordings) are left out; see `summary.json` → `rejected_trials`.
- **Thigh accelerometer is sign-inverted** relative to its gyro in this dataset; multiply it by −1.
- Frames: (x, y, z)_MuJoCo = (x, −z, y)_OpenSim.
