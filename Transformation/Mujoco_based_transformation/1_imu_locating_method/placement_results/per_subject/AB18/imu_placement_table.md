# AB18 IMU placement

Estimated from mocap IK + raw IMU data with the MuJoCo model converted from `AB18.osim`. Median over trials (treadmill, levelground, ramp, stair).

Position is in each segment's own frame, in cm: **x forward, y left, z up**, measured from the segment origin. IQR is the spread between trials.

## Position

| IMU | Segment (origin) | Trials used | Position (x, y, z) cm | Trial spread IQR (x, y, z) cm | Place on the body |
|---|---|---|---|---|---|
| Foot | `calcn_r` (heel) | 20/24 | (12.2, 0.1, 1.2) | (1.9, 1.3, 0.4) | Top of the foot, 71% of the way from heel to toe joint (midfoot/laces), centred |
| Shank | `tibia_r` (knee centre) | 20/24 | (0.7, 4.9, -27.1) | (0.8, 2.8, 1.6) | Front of the shin, 59% of the way down, medial |
| Thigh | `femur_r` (hip centre) | 13/24 | (8.8, 3.9, -20.0) | (4.0, 1.6, 6.8) | Front of the thigh, 49% of the way down |
| Trunk | `torso` (lumbar joint) | 13/24 | not reliable | — | Not reliable (lumbar locked in IK, no torso markers) |

## Orientation (sensor axes in the segment frame)

| IMU | Sensor axes | Quaternion w, x, y, z (sensor → segment) | Orientation spread | Clock offset* |
|---|---|---|---|---|
| Foot | x forward, ~43° down the instep; y left (medial); z out of the top of the foot | 0.9022, 0.2169, 0.3720, -0.0262 | 5.5° | -9.0 ms |
| Shank | x down the shank; y left (medial); z out of the shin, turned 6° laterally | 0.6834, 0.1231, 0.7179, 0.0495 | 4.6° | -13.6 ms |
| Thigh | x up the thigh; y backward (into the thigh); z right (lateral) | 0.5139, 0.4473, -0.5155, 0.5198 | 10.2° | -6.8 ms |
| Trunk | not reliable | — | — | -35.5 ms |

*Clock offset: t_mocap = t_imu + offset, so negative means the IMU samples lag the mocap.

## Foot and shank by activity

| IMU | Activity | Trials | Position (x, y, z) cm | Trial-to-trial SD (x, y, z) cm |
|---|---|---|---|---|
| Foot | treadmill | 6 | (11.5, -1.0, 1.5) | (0.1, 0.1, 0.2) |
| Foot | levelground | 6 | (9.0, -2.2, 0.3) | (6.0, 4.3, 4.0) |
| Foot | ramp | 6 | (12.4, 0.2, 1.2) | (0.9, 0.5, 0.7) |
| Foot | stair | 6 | (13.5, 0.4, 1.1) | (0.4, 0.5, 0.2) |
| Shank | treadmill | 6 | (0.3, 6.8, -25.8) | (0.3, 0.3, 0.3) |
| Shank | levelground | 6 | (0.8, 6.3, -25.5) | (4.4, 4.4, 8.8) |
| Shank | ramp | 6 | (0.7, 4.3, -27.5) | (1.5, 1.6, 1.9) |
| Shank | stair | 6 | (1.3, 3.2, -27.5) | (0.2, 0.7, 0.6) |

## Notes

- **Mounting orientation is constant**: it is how the sensor sits on its own segment. The shank IMU is above the ankle, so ankle motion does not affect it; the foot IMU turns with the foot.
- **Foot pitch drifts with ankle angle** (checked on AB09): up to ~±10° at the extremes (push-off, deep dorsiflexion), likely shoe/midfoot bending that the rigid model foot does not capture.
- **Rejected trials**: trials whose accelerometer the model cannot explain (residual RMS > 0.65 × signal RMS: saturation at ±8 g / ±16 rad/s, damped or corrupted recordings) are left out; see `summary.json` → `rejected_trials`.
- **Thigh accelerometer is sign-inverted** relative to its gyro in this dataset; multiply it by −1.
- Frames: (x, y, z)_MuJoCo = (x, −z, y)_OpenSim.
