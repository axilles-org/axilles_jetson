# AB25 IMU placement

Estimated from mocap IK + raw IMU data with the MuJoCo model converted from `AB25.osim`. Median over trials (treadmill, levelground, ramp, stair).

Position is in each segment's own frame, in cm: **x forward, y left, z up**, measured from the segment origin. IQR is the spread between trials.

## Position

| IMU | Segment (origin) | Trials used | Position (x, y, z) cm | Trial spread IQR (x, y, z) cm | Place on the body |
|---|---|---|---|---|---|
| Foot | `calcn_r` (heel) | 12/24 | (12.4, -0.1, 1.5) | (0.3, 0.4, 0.3) | Top of the foot, 76% of the way from heel to toe joint (midfoot/laces), centred |
| Shank | `tibia_r` (knee centre) | 12/24 | (2.0, -1.1, -25.4) | (1.2, 1.2, 1.1) | Front of the shin, 61% of the way down, lateral |
| Thigh | `femur_r` (hip centre) | 3/24 | (9.5, 14.4, -24.9) | (0.2, 0.3, 0.2) | Front of the thigh, 70% of the way down |
| Trunk | `torso` (lumbar joint) | 10/24 | not reliable | — | Not reliable (lumbar locked in IK, no torso markers) |

## Orientation (sensor axes in the segment frame)

| IMU | Sensor axes | Quaternion w, x, y, z (sensor → segment) | Orientation spread | Clock offset* |
|---|---|---|---|---|
| Foot | x forward, ~37° down the instep; y left (medial); z out of the top of the foot | 0.9314, 0.1789, 0.3070, -0.0783 | 2.8° | -8.0 ms |
| Shank | x down the shank; y left (medial); z out of the shin, turned 13° laterally | 0.7007, 0.1137, 0.7030, -0.0436 | 2.2° | -14.0 ms |
| Thigh | x up the thigh; y backward (into the thigh); z right (lateral) | 0.6954, 0.0817, 0.7139, 0.0029 | 0.9° | -22.0 ms |
| Trunk | not reliable | — | — | -22.6 ms |

*Clock offset: t_mocap = t_imu + offset, so negative means the IMU samples lag the mocap.

## Foot and shank by activity

| IMU | Activity | Trials | Position (x, y, z) cm | Trial-to-trial SD (x, y, z) cm |
|---|---|---|---|---|
| Foot | treadmill | 6 | (13.7, 1.7, 2.9) | (2.1, 1.8, 1.8) |
| Foot | levelground | 6 | (11.5, -0.9, 2.1) | (2.6, 8.3, 2.4) |
| Foot | ramp | 6 | (16.0, -1.8, 2.3) | (2.1, 1.1, 0.4) |
| Foot | stair | 6 | (12.4, -0.2, 1.3) | (0.2, 0.4, 0.2) |
| Shank | treadmill | 6 | (2.9, 0.2, -25.1) | (0.1, 3.4, 1.3) |
| Shank | levelground | 6 | (2.3, -0.7, -25.3) | (6.9, 8.3, 1.5) |
| Shank | ramp | 6 | (2.3, -1.5, -27.1) | (0.3, 0.4, 1.0) |
| Shank | stair | 6 | (1.5, -0.7, -25.8) | (0.2, 0.3, 0.3) |

## Notes

- **Mounting orientation is constant**: it is how the sensor sits on its own segment. The shank IMU is above the ankle, so ankle motion does not affect it; the foot IMU turns with the foot.
- **Foot pitch drifts with ankle angle** (checked on AB09): up to ~±10° at the extremes (push-off, deep dorsiflexion), likely shoe/midfoot bending that the rigid model foot does not capture.
- **Rejected trials**: trials whose accelerometer the model cannot explain (residual RMS > 0.65 × signal RMS: saturation at ±8 g / ±16 rad/s, damped or corrupted recordings) are left out; see `summary.json` → `rejected_trials`.
- **Thigh accelerometer is sign-inverted** relative to its gyro in this dataset; multiply it by −1.
- Frames: (x, y, z)_MuJoCo = (x, −z, y)_OpenSim.
