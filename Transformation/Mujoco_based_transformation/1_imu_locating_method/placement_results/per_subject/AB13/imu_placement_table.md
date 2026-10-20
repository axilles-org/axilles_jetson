# AB13 IMU placement

Estimated from mocap IK + raw IMU data with the MuJoCo model converted from `AB13.osim`. Median over trials (treadmill, levelground, ramp, stair).

Position is in each segment's own frame, in cm: **x forward, y left, z up**, measured from the segment origin. IQR is the spread between trials.

## Position

| IMU | Segment (origin) | Trials used | Position (x, y, z) cm | Trial spread IQR (x, y, z) cm | Place on the body |
|---|---|---|---|---|---|
| Foot | `calcn_r` (heel) | 23/24 | (13.6, 0.6, 2.5) | (0.5, 0.4, 0.4) | Top of the foot, 75% of the way from heel to toe joint (midfoot/laces), centred |
| Shank | `tibia_r` (knee centre) | 23/24 | (2.0, 0.5, -29.1) | (0.7, 1.9, 1.3) | Front of the shin, 65% of the way down, centred |
| Thigh | `femur_r` (hip centre) | 21/24 | (6.7, 2.8, -23.2) | (1.6, 2.1, 2.2) | Front of the thigh, 58% of the way down |
| Trunk | `torso` (lumbar joint) | 7/24 | not reliable | — | Not reliable (lumbar locked in IK, no torso markers) |

## Orientation (sensor axes in the segment frame)

| IMU | Sensor axes | Quaternion w, x, y, z (sensor → segment) | Orientation spread | Clock offset* |
|---|---|---|---|---|
| Foot | x forward, ~40° down the instep; y left (medial); z out of the top of the foot | 0.9275, 0.1480, 0.3413, -0.0362 | 3.4° | -9.4 ms |
| Shank | x down the shank; y left (medial); z out of the shin, turned 16° laterally | 0.6954, 0.1613, 0.6991, -0.0399 | 1.8° | -15.5 ms |
| Thigh | x up the thigh; y backward (into the thigh); z right (lateral) | 0.5128, 0.5069, -0.5016, 0.4781 | 7.0° | -16.6 ms |
| Trunk | not reliable | — | — | -17.7 ms |

*Clock offset: t_mocap = t_imu + offset, so negative means the IMU samples lag the mocap.

## Foot and shank by activity

| IMU | Activity | Trials | Position (x, y, z) cm | Trial-to-trial SD (x, y, z) cm |
|---|---|---|---|---|
| Foot | treadmill | 6 | (13.3, 0.7, 2.8) | (4.1, 4.1, 4.2) |
| Foot | levelground | 6 | (13.7, 0.2, 2.3) | (0.4, 0.2, 0.2) |
| Foot | ramp | 6 | (14.0, 0.6, 2.6) | (0.2, 0.4, 0.1) |
| Foot | stair | 6 | (13.2, 0.7, 2.1) | (0.4, 0.3, 0.2) |
| Shank | treadmill | 6 | (2.0, 1.3, -28.0) | (0.2, 2.0, 5.7) |
| Shank | levelground | 6 | (2.1, 1.3, -28.5) | (0.4, 0.3, 0.5) |
| Shank | ramp | 6 | (2.7, -0.4, -30.3) | (0.2, 0.4, 0.5) |
| Shank | stair | 6 | (1.5, -0.6, -29.2) | (0.3, 0.8, 0.3) |

## Notes

- **Mounting orientation is constant**: it is how the sensor sits on its own segment. The shank IMU is above the ankle, so ankle motion does not affect it; the foot IMU turns with the foot.
- **Foot pitch drifts with ankle angle** (checked on AB09): up to ~±10° at the extremes (push-off, deep dorsiflexion), likely shoe/midfoot bending that the rigid model foot does not capture.
- **Rejected trials**: trials whose accelerometer the model cannot explain (residual RMS > 0.65 × signal RMS: saturation at ±8 g / ±16 rad/s, damped or corrupted recordings) are left out; see `summary.json` → `rejected_trials`.
- **Thigh accelerometer is sign-inverted** relative to its gyro in this dataset; multiply it by −1.
- Frames: (x, y, z)_MuJoCo = (x, −z, y)_OpenSim.
