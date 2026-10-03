# AB17 IMU placement

Estimated from mocap IK + raw IMU data with the MuJoCo model converted from `AB17.osim`. Median over trials (treadmill, levelground, ramp, stair).

Position is in each segment's own frame, in cm: **x forward, y left, z up**, measured from the segment origin. IQR is the spread between trials.

## Position

| IMU | Segment (origin) | Trials used | Position (x, y, z) cm | Trial spread IQR (x, y, z) cm | Place on the body |
|---|---|---|---|---|---|
| Foot | `calcn_r` (heel) | 23/24 | (11.2, 0.2, 3.0) | (0.4, 0.8, 0.2) | Top of the foot, 64% of the way from heel to toe joint (midfoot/laces), centred |
| Shank | `tibia_r` (knee centre) | 23/24 | (2.5, 0.8, -26.1) | (0.3, 0.9, 1.8) | Front of the shin, 60% of the way down, centred |
| Thigh | `femur_r` (hip centre) | 22/24 | (5.0, 1.2, -25.1) | (0.6, 1.2, 1.5) | Front of the thigh, 65% of the way down |
| Trunk | `torso` (lumbar joint) | 14/24 | not reliable | — | Not reliable (lumbar locked in IK, no torso markers) |

## Orientation (sensor axes in the segment frame)

| IMU | Sensor axes | Quaternion w, x, y, z (sensor → segment) | Orientation spread | Clock offset* |
|---|---|---|---|---|
| Foot | x forward, ~38° down the instep; y left (medial); z out of the top of the foot | 0.9340, 0.1465, 0.3245, -0.0286 | 1.7° | -8.7 ms |
| Shank | x down the shank; y left (medial); z out of the shin, straight forward | 0.7003, 0.0323, 0.7102, 0.0638 | 2.1° | -13.5 ms |
| Thigh | x up the thigh; y backward (into the thigh); z right (lateral) | 0.5481, 0.5187, -0.4693, 0.4587 | 7.0° | -13.9 ms |
| Trunk | not reliable | — | — | -37.1 ms |

*Clock offset: t_mocap = t_imu + offset, so negative means the IMU samples lag the mocap.

## Foot and shank by activity

| IMU | Activity | Trials | Position (x, y, z) cm | Trial-to-trial SD (x, y, z) cm |
|---|---|---|---|---|
| Foot | treadmill | 6 | (11.1, -0.3, 3.1) | (0.1, 0.3, 0.1) |
| Foot | levelground | 6 | (11.2, -0.4, 3.1) | (0.4, 0.8, 0.9) |
| Foot | ramp | 6 | (11.5, 0.4, 3.1) | (0.2, 0.4, 0.1) |
| Foot | stair | 6 | (11.2, 0.5, 2.9) | (0.3, 0.3, 0.1) |
| Shank | treadmill | 6 | (2.8, 0.8, -25.5) | (0.1, 0.2, 0.1) |
| Shank | levelground | 6 | (2.6, 0.8, -25.3) | (0.6, 0.4, 0.4) |
| Shank | ramp | 6 | (2.4, -0.8, -26.7) | (0.2, 0.3, 0.6) |
| Shank | stair | 6 | (2.3, 1.4, -27.6) | (0.2, 0.9, 0.3) |

## Notes

- **Mounting orientation is constant**: it is how the sensor sits on its own segment. The shank IMU is above the ankle, so ankle motion does not affect it; the foot IMU turns with the foot.
- **Foot pitch drifts with ankle angle** (checked on AB09): up to ~±10° at the extremes (push-off, deep dorsiflexion), likely shoe/midfoot bending that the rigid model foot does not capture.
- **Rejected trials**: trials whose accelerometer the model cannot explain (residual RMS > 0.65 × signal RMS: saturation at ±8 g / ±16 rad/s, damped or corrupted recordings) are left out; see `summary.json` → `rejected_trials`.
- **Thigh accelerometer is sign-inverted** relative to its gyro in this dataset; multiply it by −1.
- Frames: (x, y, z)_MuJoCo = (x, −z, y)_OpenSim.
