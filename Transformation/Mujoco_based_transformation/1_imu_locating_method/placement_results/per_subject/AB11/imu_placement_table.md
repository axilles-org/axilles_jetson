# AB11 IMU placement

Estimated from mocap IK + raw IMU data with the MuJoCo model converted from `AB11.osim`. Median over trials (treadmill, levelground, ramp, stair).

Position is in each segment's own frame, in cm: **x forward, y left, z up**, measured from the segment origin. IQR is the spread between trials.

## Position

| IMU | Segment (origin) | Trials used | Position (x, y, z) cm | Trial spread IQR (x, y, z) cm | Place on the body |
|---|---|---|---|---|---|
| Foot | `calcn_r` (heel) | 18/24 | (14.3, -0.2, 1.6) | (0.5, 0.7, 0.7) | Top of the foot, 78% of the way from heel to toe joint (midfoot/laces), centred |
| Shank | `tibia_r` (knee centre) | 16/24 | (3.5, -0.3, -27.5) | (0.4, 1.5, 1.0) | Front of the shin, 63% of the way down, centred |
| Thigh | `femur_r` (hip centre) | 13/24 | (6.2, 7.6, -24.4) | (0.8, 2.4, 3.3) | Front of the thigh, 62% of the way down |
| Trunk | `torso` (lumbar joint) | 4/24 | not reliable | — | Not reliable (lumbar locked in IK, no torso markers) |

## Orientation (sensor axes in the segment frame)

| IMU | Sensor axes | Quaternion w, x, y, z (sensor → segment) | Orientation spread | Clock offset* |
|---|---|---|---|---|
| Foot | x forward, ~36° down the instep; y left (medial); z out of the top of the foot | 0.9429, 0.1045, 0.2973, -0.1075 | 2.3° | -10.0 ms |
| Shank | x down the shank; y left (medial); z out of the shin, straight forward | 0.7086, 0.0500, 0.7022, 0.0481 | 2.7° | -12.3 ms |
| Thigh | x up the thigh; y backward (into the thigh); z right (lateral) | 0.4455, 0.6291, -0.4664, 0.4339 | 4.0° | -3.4 ms |
| Trunk | not reliable | — | — | -26.6 ms |

*Clock offset: t_mocap = t_imu + offset, so negative means the IMU samples lag the mocap.

## Foot and shank by activity

| IMU | Activity | Trials | Position (x, y, z) cm | Trial-to-trial SD (x, y, z) cm |
|---|---|---|---|---|
| Foot | treadmill | 6 | (14.2, -0.5, 1.3) | (1.1, 2.0, 0.7) |
| Foot | levelground | 6 | (15.5, -4.6, -0.2) | (0.1, 0.2, 0.2) |
| Foot | ramp | 6 | (14.3, -0.4, 1.6) | (0.2, 0.5, 0.3) |
| Foot | stair | 6 | (14.2, 0.2, 2.0) | (0.4, 0.3, 0.2) |
| Shank | treadmill | 6 | (3.4, -1.3, -25.7) | (0.5, 6.3, 3.8) |
| Shank | levelground | 6 | (2.6, -7.2, -32.1) | (0.1, 0.6, 0.0) |
| Shank | ramp | 6 | (3.9, -0.2, -27.2) | (0.2, 0.8, 0.3) |
| Shank | stair | 6 | (3.6, -0.6, -28.0) | (0.2, 0.4, 0.3) |

## Notes

- **Mounting orientation is constant**: it is how the sensor sits on its own segment. The shank IMU is above the ankle, so ankle motion does not affect it; the foot IMU turns with the foot.
- **Foot pitch drifts with ankle angle** (checked on AB09): up to ~±10° at the extremes (push-off, deep dorsiflexion), likely shoe/midfoot bending that the rigid model foot does not capture.
- **Rejected trials**: trials whose accelerometer the model cannot explain (residual RMS > 0.65 × signal RMS: saturation at ±8 g / ±16 rad/s, damped or corrupted recordings) are left out; see `summary.json` → `rejected_trials`.
- **Thigh accelerometer is sign-inverted** relative to its gyro in this dataset; multiply it by −1.
- Frames: (x, y, z)_MuJoCo = (x, −z, y)_OpenSim.
