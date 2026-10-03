# AB27 IMU placement

Estimated from mocap IK + raw IMU data with the MuJoCo model converted from `AB27.osim`. Median over trials (treadmill, levelground, ramp, stair).

Position is in each segment's own frame, in cm: **x forward, y left, z up**, measured from the segment origin. IQR is the spread between trials.

## Position

| IMU | Segment (origin) | Trials used | Position (x, y, z) cm | Trial spread IQR (x, y, z) cm | Place on the body |
|---|---|---|---|---|---|
| Foot | `calcn_r` (heel) | 22/24 | (14.4, 1.0, 2.3) | (0.3, 0.7, 0.5) | Top of the foot, 77% of the way from heel to toe joint (midfoot/laces), centred |
| Shank | `tibia_r` (knee centre) | 22/24 | (2.7, -0.8, -29.6) | (0.3, 1.2, 1.1) | Front of the shin, 65% of the way down, centred |
| Thigh | `femur_r` (hip centre) | 15/24 | (5.1, 5.5, -33.8) | (0.7, 1.4, 2.9) | Front of the thigh, 88% of the way down |
| Trunk | `torso` (lumbar joint) | 16/24 | not reliable | — | Not reliable (lumbar locked in IK, no torso markers) |

## Orientation (sensor axes in the segment frame)

| IMU | Sensor axes | Quaternion w, x, y, z (sensor → segment) | Orientation spread | Clock offset* |
|---|---|---|---|---|
| Foot | x forward, ~40° down the instep; y left (medial); z out of the top of the foot | 0.9212, 0.1692, 0.3502, 0.0084 | 2.3° | -7.8 ms |
| Shank | x down the shank; y left (medial); z out of the shin, turned 12° laterally | 0.6817, 0.1235, 0.7204, -0.0330 | 1.8° | -13.9 ms |
| Thigh | x up the thigh; y backward (into the thigh); z right (lateral) | 0.6467, 0.0332, 0.7621, 0.0017 | 3.9° | -16.2 ms |
| Trunk | not reliable | — | — | -24.2 ms |

*Clock offset: t_mocap = t_imu + offset, so negative means the IMU samples lag the mocap.

## Foot and shank by activity

| IMU | Activity | Trials | Position (x, y, z) cm | Trial-to-trial SD (x, y, z) cm |
|---|---|---|---|---|
| Foot | treadmill | 6 | (14.9, -0.7, 2.1) | (0.2, 0.8, 0.3) |
| Foot | levelground | 6 | (13.9, 1.0, 2.0) | (0.5, 1.1, 1.0) |
| Foot | ramp | 6 | (14.3, 1.3, 2.5) | (0.1, 0.2, 0.2) |
| Foot | stair | 6 | (14.4, 1.0, 2.5) | (0.4, 0.3, 0.2) |
| Shank | treadmill | 6 | (2.7, -0.4, -29.3) | (0.1, 0.8, 0.2) |
| Shank | levelground | 6 | (2.7, -0.5, -29.4) | (5.4, 2.6, 2.5) |
| Shank | ramp | 6 | (2.8, -1.6, -29.9) | (0.1, 0.9, 0.5) |
| Shank | stair | 6 | (2.4, -0.7, -30.8) | (0.2, 0.6, 0.3) |

## Notes

- **Mounting orientation is constant**: it is how the sensor sits on its own segment. The shank IMU is above the ankle, so ankle motion does not affect it; the foot IMU turns with the foot.
- **Foot pitch drifts with ankle angle** (checked on AB09): up to ~±10° at the extremes (push-off, deep dorsiflexion), likely shoe/midfoot bending that the rigid model foot does not capture.
- **Rejected trials**: trials whose accelerometer the model cannot explain (residual RMS > 0.65 × signal RMS: saturation at ±8 g / ±16 rad/s, damped or corrupted recordings) are left out; see `summary.json` → `rejected_trials`.
- **Thigh accelerometer is sign-inverted** relative to its gyro in this dataset; multiply it by −1.
- Frames: (x, y, z)_MuJoCo = (x, −z, y)_OpenSim.
