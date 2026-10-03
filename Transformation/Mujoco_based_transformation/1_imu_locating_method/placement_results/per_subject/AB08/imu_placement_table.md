# AB08 IMU placement

Estimated from mocap IK + raw IMU data with the MuJoCo model converted from `AB08.osim`. Median over trials (treadmill, levelground, ramp, stair).

Position is in each segment's own frame, in cm: **x forward, y left, z up**, measured from the segment origin. IQR is the spread between trials.

## Position

| IMU | Segment (origin) | Trials used | Position (x, y, z) cm | Trial spread IQR (x, y, z) cm | Place on the body |
|---|---|---|---|---|---|
| Foot | `calcn_r` (heel) | 23/24 | (11.7, -0.1, 3.2) | (0.2, 0.9, 0.3) | Top of the foot, 63% of the way from heel to toe joint (midfoot/laces), centred |
| Shank | `tibia_r` (knee centre) | 21/24 | (2.4, 0.9, -26.1) | (0.6, 1.0, 0.9) | Front of the shin, 58% of the way down, centred |
| Thigh | `femur_r` (hip centre) | 20/24 | (7.0, 1.9, -21.2) | (1.1, 3.0, 3.3) | Front of the thigh, 52% of the way down |
| Trunk | `torso` (lumbar joint) | 12/24 | not reliable | — | Not reliable (lumbar locked in IK, no torso markers) |

## Orientation (sensor axes in the segment frame)

| IMU | Sensor axes | Quaternion w, x, y, z (sensor → segment) | Orientation spread | Clock offset* |
|---|---|---|---|---|
| Foot | x forward, ~43° down the instep; y left (medial); z out of the top of the foot | 0.9145, 0.1777, 0.3510, -0.0948 | 2.6° | -11.2 ms |
| Shank | x down the shank; y left (medial); z out of the shin, turned 17° laterally | 0.6990, 0.1369, 0.6976, -0.0782 | 2.2° | -16.6 ms |
| Thigh | x up the thigh; y backward (into the thigh); z right (lateral) | 0.4862, 0.5918, -0.4472, 0.4618 | 6.5° | -18.4 ms |
| Trunk | not reliable | — | — | -36.0 ms |

*Clock offset: t_mocap = t_imu + offset, so negative means the IMU samples lag the mocap.

## Foot and shank by activity

| IMU | Activity | Trials | Position (x, y, z) cm | Trial-to-trial SD (x, y, z) cm |
|---|---|---|---|---|
| Foot | treadmill | 6 | (11.7, 0.8, 3.4) | (0.1, 0.2, 0.1) |
| Foot | levelground | 6 | (12.3, -0.2, 3.5) | (1.8, 23.8, 5.0) |
| Foot | ramp | 6 | (11.8, -0.1, 3.2) | (0.1, 0.5, 0.3) |
| Foot | stair | 6 | (11.6, -0.5, 3.0) | (0.2, 0.4, 0.2) |
| Shank | treadmill | 6 | (2.9, 0.4, -25.9) | (0.1, 0.5, 0.0) |
| Shank | levelground | 6 | (2.4, -0.6, -25.7) | (1.7, 9.9, 1.9) |
| Shank | ramp | 6 | (2.3, 1.1, -26.8) | (0.2, 0.4, 0.3) |
| Shank | stair | 6 | (1.3, 2.0, -26.7) | (0.3, 1.2, 0.4) |

## Notes

- **Mounting orientation is constant**: it is how the sensor sits on its own segment. The shank IMU is above the ankle, so ankle motion does not affect it; the foot IMU turns with the foot.
- **Foot pitch drifts with ankle angle** (checked on AB09): up to ~±10° at the extremes (push-off, deep dorsiflexion), likely shoe/midfoot bending that the rigid model foot does not capture.
- **Rejected trials**: trials whose accelerometer the model cannot explain (residual RMS > 0.65 × signal RMS: saturation at ±8 g / ±16 rad/s, damped or corrupted recordings) are left out; see `summary.json` → `rejected_trials`.
- **Thigh accelerometer is sign-inverted** relative to its gyro in this dataset; multiply it by −1.
- Frames: (x, y, z)_MuJoCo = (x, −z, y)_OpenSim.
