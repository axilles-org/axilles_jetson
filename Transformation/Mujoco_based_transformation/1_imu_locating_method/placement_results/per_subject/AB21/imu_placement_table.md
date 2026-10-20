# AB21 IMU placement

Estimated from mocap IK + raw IMU data with the MuJoCo model converted from `AB21.osim`. Median over trials (treadmill, levelground, ramp, stair).

Position is in each segment's own frame, in cm: **x forward, y left, z up**, measured from the segment origin. IQR is the spread between trials.

## Position

| IMU | Segment (origin) | Trials used | Position (x, y, z) cm | Trial spread IQR (x, y, z) cm | Place on the body |
|---|---|---|---|---|---|
| Foot | `calcn_r` (heel) | 23/24 | (10.9, 2.3, 3.2) | (0.4, 0.6, 0.4) | Top of the foot, 65% of the way from heel to toe joint (midfoot/laces), medial |
| Shank | `tibia_r` (knee centre) | 22/24 | (1.8, 1.7, -23.2) | (0.8, 1.6, 4.2) | Front of the shin, 56% of the way down, medial |
| Thigh | `femur_r` (hip centre) | 24/24 | (7.0, 6.7, -20.3) | (1.0, 5.7, 8.2) | Front of the thigh, 57% of the way down |
| Trunk | `torso` (lumbar joint) | 7/24 | not reliable | — | Not reliable (lumbar locked in IK, no torso markers) |

## Orientation (sensor axes in the segment frame)

| IMU | Sensor axes | Quaternion w, x, y, z (sensor → segment) | Orientation spread | Clock offset* |
|---|---|---|---|---|
| Foot | x forward, ~36° down the instep; y left (medial); z out of the top of the foot | 0.9443, -0.0826, 0.3036, 0.0966 | 5.0° | -7.7 ms |
| Shank | x down the shank; y left (medial); z out of the shin, straight forward | 0.6905, 0.0678, 0.7189, 0.0415 | 4.1° | -13.0 ms |
| Thigh | x up the thigh; y backward (into the thigh); z right (lateral) | 0.6820, 0.0241, 0.7310, 0.0072 | 11.5° | -21.1 ms |
| Trunk | not reliable | — | — | -21.9 ms |

*Clock offset: t_mocap = t_imu + offset, so negative means the IMU samples lag the mocap.

## Foot and shank by activity

| IMU | Activity | Trials | Position (x, y, z) cm | Trial-to-trial SD (x, y, z) cm |
|---|---|---|---|---|
| Foot | treadmill | 6 | (10.9, 2.6, 3.1) | (0.5, 1.1, 0.7) |
| Foot | levelground | 6 | (10.7, 2.1, 3.2) | (0.4, 0.9, 0.7) |
| Foot | ramp | 6 | (11.2, 2.4, 3.3) | (0.1, 0.3, 0.1) |
| Foot | stair | 6 | (10.9, 2.0, 2.9) | (0.3, 0.3, 0.2) |
| Shank | treadmill | 6 | (1.1, 3.2, -21.5) | (0.5, 1.2, 2.0) |
| Shank | levelground | 6 | (1.4, 2.2, -21.1) | (1.4, 0.9, 0.9) |
| Shank | ramp | 6 | (1.9, 1.5, -23.5) | (0.1, 0.2, 0.6) |
| Shank | stair | 6 | (2.5, 1.3, -25.8) | (0.3, 0.4, 0.2) |

## Notes

- **Mounting orientation is constant**: it is how the sensor sits on its own segment. The shank IMU is above the ankle, so ankle motion does not affect it; the foot IMU turns with the foot.
- **Foot pitch drifts with ankle angle** (checked on AB09): up to ~±10° at the extremes (push-off, deep dorsiflexion), likely shoe/midfoot bending that the rigid model foot does not capture.
- **Rejected trials**: trials whose accelerometer the model cannot explain (residual RMS > 0.65 × signal RMS: saturation at ±8 g / ±16 rad/s, damped or corrupted recordings) are left out; see `summary.json` → `rejected_trials`.
- **Thigh accelerometer is sign-inverted** relative to its gyro in this dataset; multiply it by −1.
- Frames: (x, y, z)_MuJoCo = (x, −z, y)_OpenSim.
