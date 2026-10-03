# AB24 IMU placement

Estimated from mocap IK + raw IMU data with the MuJoCo model converted from `AB24.osim`. Median over trials (treadmill, levelground, ramp, stair).

Position is in each segment's own frame, in cm: **x forward, y left, z up**, measured from the segment origin. IQR is the spread between trials.

## Position

| IMU | Segment (origin) | Trials used | Position (x, y, z) cm | Trial spread IQR (x, y, z) cm | Place on the body |
|---|---|---|---|---|---|
| Foot | `calcn_r` (heel) | 22/24 | (12.9, 0.9, 2.5) | (0.4, 0.9, 0.2) | Top of the foot, 73% of the way from heel to toe joint (midfoot/laces), centred |
| Shank | `tibia_r` (knee centre) | 22/24 | (1.8, 1.4, -27.9) | (0.6, 1.5, 1.4) | Front of the shin, 62% of the way down, medial |
| Thigh | `femur_r` (hip centre) | 10/24 | (7.6, 3.9, -22.6) | (1.1, 1.0, 2.7) | Front of the thigh, 57% of the way down |
| Trunk | `torso` (lumbar joint) | 11/24 | not reliable | — | Not reliable (lumbar locked in IK, no torso markers) |

## Orientation (sensor axes in the segment frame)

| IMU | Sensor axes | Quaternion w, x, y, z (sensor → segment) | Orientation spread | Clock offset* |
|---|---|---|---|---|
| Foot | x forward, ~42° down the instep; y left (medial); z out of the top of the foot | 0.9102, 0.2139, 0.3477, -0.0698 | 5.5° | -7.6 ms |
| Shank | x down the shank; y left (medial); z out of the shin, straight forward | 0.6985, 0.0356, 0.7108, 0.0748 | 2.1° | -12.5 ms |
| Thigh | x up the thigh; y backward (into the thigh); z right (lateral) | 0.5188, 0.5641, -0.4941, 0.4106 | 2.2° | -10.3 ms |
| Trunk | not reliable | — | — | -15.1 ms |

*Clock offset: t_mocap = t_imu + offset, so negative means the IMU samples lag the mocap.

## Foot and shank by activity

| IMU | Activity | Trials | Position (x, y, z) cm | Trial-to-trial SD (x, y, z) cm |
|---|---|---|---|---|
| Foot | treadmill | 6 | (12.6, 0.3, 2.4) | (0.1, 0.3, 0.1) |
| Foot | levelground | 6 | (12.8, 0.6, 2.3) | (0.7, 3.0, 0.4) |
| Foot | ramp | 6 | (13.2, 1.7, 2.6) | (0.2, 0.3, 0.3) |
| Foot | stair | 6 | (13.1, 1.1, 2.3) | (0.2, 0.2, 0.2) |
| Shank | treadmill | 6 | (1.5, 2.4, -27.8) | (0.0, 0.2, 0.1) |
| Shank | levelground | 6 | (1.5, 2.1, -27.9) | (0.6, 1.1, 1.1) |
| Shank | ramp | 6 | (2.1, 0.9, -27.8) | (0.3, 0.3, 0.6) |
| Shank | stair | 6 | (2.1, 0.8, -29.5) | (0.2, 0.3, 0.3) |

## Notes

- **Mounting orientation is constant**: it is how the sensor sits on its own segment. The shank IMU is above the ankle, so ankle motion does not affect it; the foot IMU turns with the foot.
- **Foot pitch drifts with ankle angle** (checked on AB09): up to ~±10° at the extremes (push-off, deep dorsiflexion), likely shoe/midfoot bending that the rigid model foot does not capture.
- **Rejected trials**: trials whose accelerometer the model cannot explain (residual RMS > 0.65 × signal RMS: saturation at ±8 g / ±16 rad/s, damped or corrupted recordings) are left out; see `summary.json` → `rejected_trials`.
- **Thigh accelerometer is sign-inverted** relative to its gyro in this dataset; multiply it by −1.
- Frames: (x, y, z)_MuJoCo = (x, −z, y)_OpenSim.
