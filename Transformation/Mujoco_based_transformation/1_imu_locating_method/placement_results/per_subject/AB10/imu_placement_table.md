# AB10 IMU placement

Estimated from mocap IK + raw IMU data with the MuJoCo model converted from `AB10.osim`. Median over trials (treadmill, levelground, ramp, stair).

Position is in each segment's own frame, in cm: **x forward, y left, z up**, measured from the segment origin. IQR is the spread between trials.

## Position

| IMU | Segment (origin) | Trials used | Position (x, y, z) cm | Trial spread IQR (x, y, z) cm | Place on the body |
|---|---|---|---|---|---|
| Foot | `calcn_r` (heel) | 16/24 | (13.3, -0.1, 1.0) | (0.4, 0.7, 0.3) | Top of the foot, 73% of the way from heel to toe joint (midfoot/laces), centred |
| Shank | `tibia_r` (knee centre) | 15/24 | (3.4, 0.1, -28.0) | (0.6, 1.2, 1.5) | Front of the shin, 66% of the way down, centred |
| Thigh | `femur_r` (hip centre) | 12/24 | (8.3, 4.7, -15.8) | (0.6, 2.6, 2.2) | Front of the thigh, 40% of the way down |
| Trunk | `torso` (lumbar joint) | 8/24 | not reliable | — | Not reliable (lumbar locked in IK, no torso markers) |

## Orientation (sensor axes in the segment frame)

| IMU | Sensor axes | Quaternion w, x, y, z (sensor → segment) | Orientation spread | Clock offset* |
|---|---|---|---|---|
| Foot | x forward, ~36° down the instep; y left (medial); z out of the top of the foot | 0.9311, 0.1785, 0.2974, -0.1129 | 4.4° | -10.2 ms |
| Shank | x down the shank; y left (medial); z out of the shin, turned 17° laterally | 0.7005, 0.1639, 0.6929, -0.0482 | 2.4° | -15.9 ms |
| Thigh | x up the thigh; y backward (into the thigh); z right (lateral) | -0.4349, -0.5594, 0.5855, -0.3939 | 4.4° | -9.9 ms |
| Trunk | not reliable | — | — | -22.5 ms |

*Clock offset: t_mocap = t_imu + offset, so negative means the IMU samples lag the mocap.

## Foot and shank by activity

| IMU | Activity | Trials | Position (x, y, z) cm | Trial-to-trial SD (x, y, z) cm |
|---|---|---|---|---|
| Foot | treadmill | 6 | (13.2, -0.0, 1.0) | (1.5, 7.8, 2.1) |
| Foot | levelground | 6 | (13.2, -0.2, 0.9) | (0.7, 1.0, 1.9) |
| Foot | ramp | 6 | (10.7, -0.4, 1.0) | (8.3, 6.9, 14.1) |
| Foot | stair | 6 | (13.4, -0.7, 1.2) | (3.1, 2.9, 2.2) |
| Shank | treadmill | 6 | (3.5, 0.7, -27.0) | (2.9, 1.5, 5.9) |
| Shank | levelground | 6 | (3.2, 0.1, -27.5) | (2.4, 0.7, 2.6) |
| Shank | ramp | 6 | (3.9, -1.1, -28.8) | (1.4, 3.9, 11.4) |
| Shank | stair | 6 | (3.0, -0.4, -28.3) | (0.7, 0.6, 4.0) |

## Notes

- **Mounting orientation is constant**: it is how the sensor sits on its own segment. The shank IMU is above the ankle, so ankle motion does not affect it; the foot IMU turns with the foot.
- **Foot pitch drifts with ankle angle** (checked on AB09): up to ~±10° at the extremes (push-off, deep dorsiflexion), likely shoe/midfoot bending that the rigid model foot does not capture.
- **Rejected trials**: trials whose accelerometer the model cannot explain (residual RMS > 0.65 × signal RMS: saturation at ±8 g / ±16 rad/s, damped or corrupted recordings) are left out; see `summary.json` → `rejected_trials`.
- **Thigh accelerometer is sign-inverted** relative to its gyro in this dataset; multiply it by −1.
- Frames: (x, y, z)_MuJoCo = (x, −z, y)_OpenSim.
