# AB23 IMU placement

Estimated from mocap IK + raw IMU data with the MuJoCo model converted from `AB23.osim`. Median over trials (treadmill, levelground, ramp, stair).

Position is in each segment's own frame, in cm: **x forward, y left, z up**, measured from the segment origin. IQR is the spread between trials.

## Position

| IMU | Segment (origin) | Trials used | Position (x, y, z) cm | Trial spread IQR (x, y, z) cm | Place on the body |
|---|---|---|---|---|---|
| Foot | `calcn_r` (heel) | 23/24 | (12.7, 1.2, 2.5) | (0.6, 0.7, 0.3) | Top of the foot, 67% of the way from heel to toe joint (midfoot/laces), medial |
| Shank | `tibia_r` (knee centre) | 24/24 | (2.6, 1.3, -28.2) | (0.6, 1.2, 1.9) | Front of the shin, 63% of the way down, medial |
| Thigh | `femur_r` (hip centre) | 23/24 | (4.9, 4.5, -25.0) | (4.4, 2.5, 3.9) | Front of the thigh, 62% of the way down |
| Trunk | `torso` (lumbar joint) | 5/24 | not reliable | — | Not reliable (lumbar locked in IK, no torso markers) |

## Orientation (sensor axes in the segment frame)

| IMU | Sensor axes | Quaternion w, x, y, z (sensor → segment) | Orientation spread | Clock offset* |
|---|---|---|---|---|
| Foot | x forward, ~40° down the instep; y left (medial); z out of the top of the foot | 0.9161, 0.2055, 0.3423, -0.0377 | 4.3° | -5.5 ms |
| Shank | x down the shank; y left (medial); z out of the shin, turned 13° laterally | 0.7047, 0.1431, 0.6948, -0.0113 | 2.2° | -14.3 ms |
| Thigh | x up the thigh; y backward (into the thigh); z right (lateral) | 0.5142, 0.5777, -0.4682, 0.4273 | 4.6° | -17.4 ms |
| Trunk | not reliable | — | — | -34.7 ms |

*Clock offset: t_mocap = t_imu + offset, so negative means the IMU samples lag the mocap.

## Foot and shank by activity

| IMU | Activity | Trials | Position (x, y, z) cm | Trial-to-trial SD (x, y, z) cm |
|---|---|---|---|---|
| Foot | treadmill | 6 | (12.5, 0.9, 2.5) | (0.1, 0.4, 0.0) |
| Foot | levelground | 6 | (12.5, 1.5, 2.7) | (0.5, 0.7, 0.4) |
| Foot | ramp | 6 | (13.2, 1.7, 3.2) | (0.1, 0.2, 0.3) |
| Foot | stair | 6 | (12.7, 1.0, 2.5) | (0.3, 0.2, 0.4) |
| Shank | treadmill | 6 | (2.8, 1.5, -27.1) | (0.2, 0.5, 0.1) |
| Shank | levelground | 6 | (2.6, 1.0, -27.8) | (0.2, 0.8, 0.2) |
| Shank | ramp | 6 | (3.1, 0.1, -29.1) | (0.2, 0.6, 0.5) |
| Shank | stair | 6 | (1.9, 1.8, -29.5) | (0.5, 0.5, 0.3) |

## Notes

- **Mounting orientation is constant**: it is how the sensor sits on its own segment. The shank IMU is above the ankle, so ankle motion does not affect it; the foot IMU turns with the foot.
- **Foot pitch drifts with ankle angle** (checked on AB09): up to ~±10° at the extremes (push-off, deep dorsiflexion), likely shoe/midfoot bending that the rigid model foot does not capture.
- **Rejected trials**: trials whose accelerometer the model cannot explain (residual RMS > 0.65 × signal RMS: saturation at ±8 g / ±16 rad/s, damped or corrupted recordings) are left out; see `summary.json` → `rejected_trials`.
- **Thigh accelerometer is sign-inverted** relative to its gyro in this dataset; multiply it by −1.
- Frames: (x, y, z)_MuJoCo = (x, −z, y)_OpenSim.
