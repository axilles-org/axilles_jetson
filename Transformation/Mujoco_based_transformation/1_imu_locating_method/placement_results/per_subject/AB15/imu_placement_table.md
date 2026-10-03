# AB15 IMU placement

Estimated from mocap IK + raw IMU data with the MuJoCo model converted from `AB15.osim`. Median over trials (treadmill, levelground, ramp, stair).

Position is in each segment's own frame, in cm: **x forward, y left, z up**, measured from the segment origin. IQR is the spread between trials.

## Position

| IMU | Segment (origin) | Trials used | Position (x, y, z) cm | Trial spread IQR (x, y, z) cm | Place on the body |
|---|---|---|---|---|---|
| Foot | `calcn_r` (heel) | 16/24 | (13.2, 0.2, 3.1) | (0.4, 0.5, 0.3) | Top of the foot, 69% of the way from heel to toe joint (midfoot/laces), centred |
| Shank | `tibia_r` (knee centre) | 19/24 | (4.2, -2.7, -27.6) | (1.5, 2.1, 0.6) | Front of the shin, 63% of the way down, lateral |
| Thigh | `femur_r` (hip centre) | 4/24 | (7.9, 3.4, -16.4) | (0.6, 1.1, 1.2) | Front of the thigh, 45% of the way down |
| Trunk | `torso` (lumbar joint) | 1/24 | not reliable | — | Not reliable (lumbar locked in IK, no torso markers) |

## Orientation (sensor axes in the segment frame)

| IMU | Sensor axes | Quaternion w, x, y, z (sensor → segment) | Orientation spread | Clock offset* |
|---|---|---|---|---|
| Foot | x forward, ~36° down the instep; y left (medial); z out of the top of the foot | 0.9412, 0.1141, 0.2978, -0.1119 | 6.0° | -11.6 ms |
| Shank | x down the shank; y left (medial); z out of the shin, turned 9° laterally | 0.6794, 0.0500, 0.7299, -0.0562 | 3.6° | -12.7 ms |
| Thigh | x up the thigh; y backward (into the thigh); z right (lateral) | 0.5492, 0.5311, -0.5402, 0.3529 | 3.1° | -17.0 ms |
| Trunk | not reliable | — | — | -85.1 ms |

*Clock offset: t_mocap = t_imu + offset, so negative means the IMU samples lag the mocap.

## Foot and shank by activity

| IMU | Activity | Trials | Position (x, y, z) cm | Trial-to-trial SD (x, y, z) cm |
|---|---|---|---|---|
| Foot | treadmill | 6 | (13.0, 0.0, 3.2) | (0.1, 0.2, 0.1) |
| Foot | levelground | 6 | (13.3, 0.2, 3.3) | (0.2, 1.6, 0.8) |
| Foot | ramp | 6 | (15.6, 0.2, 3.7) | (2.0, 0.6, 0.9) |
| Foot | stair | 6 | (13.2, 0.5, 2.5) | (0.3, 0.3, 0.4) |
| Shank | treadmill | 6 | (5.0, -3.5, -27.8) | (0.1, 0.3, 0.2) |
| Shank | levelground | 6 | (4.4, -2.8, -27.2) | (1.0, 1.2, 1.5) |
| Shank | ramp | 6 | (4.1, -3.0, -28.4) | (0.4, 0.7, 0.4) |
| Shank | stair | 6 | (2.9, -0.3, -27.0) | (0.5, 0.9, 0.9) |

## Notes

- **Mounting orientation is constant**: it is how the sensor sits on its own segment. The shank IMU is above the ankle, so ankle motion does not affect it; the foot IMU turns with the foot.
- **Foot pitch drifts with ankle angle** (checked on AB09): up to ~±10° at the extremes (push-off, deep dorsiflexion), likely shoe/midfoot bending that the rigid model foot does not capture.
- **Rejected trials**: trials whose accelerometer the model cannot explain (residual RMS > 0.65 × signal RMS: saturation at ±8 g / ±16 rad/s, damped or corrupted recordings) are left out; see `summary.json` → `rejected_trials`.
- **Thigh accelerometer is sign-inverted** relative to its gyro in this dataset; multiply it by −1.
- Frames: (x, y, z)_MuJoCo = (x, −z, y)_OpenSim.
