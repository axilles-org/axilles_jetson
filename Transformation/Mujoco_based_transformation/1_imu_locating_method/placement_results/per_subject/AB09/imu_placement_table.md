# AB09 IMU placement

Estimated from mocap IK + raw IMU data with the MuJoCo model converted from `AB09.osim`. Median over trials (treadmill, levelground, ramp, stair).

Position is in each segment's own frame, in cm: **x forward, y left, z up**, measured from the segment origin. IQR is the spread between trials.

## Position

| IMU | Segment (origin) | Trials used | Position (x, y, z) cm | Trial spread IQR (x, y, z) cm | Place on the body |
|---|---|---|---|---|---|
| Foot | `calcn_r` (heel) | 22/24 | (11.7, 0.1, 3.1) | (0.4, 0.6, 0.9) | Top of the foot, 71% of the way from heel to toe joint (midfoot/laces), centred |
| Shank | `tibia_r` (knee centre) | 21/24 | (2.0, -0.8, -24.4) | (0.7, 4.0, 1.6) | Front of the shin, 63% of the way down, centred |
| Thigh | `femur_r` (hip centre) | 18/24 | (6.4, 1.4, -17.0) | (0.9, 1.2, 3.3) | Front of the thigh, 45% of the way down |
| Trunk | `torso` (lumbar joint) | 17/24 | not reliable | — | Not reliable (lumbar locked in IK, no torso markers) |

## Orientation (sensor axes in the segment frame)

| IMU | Sensor axes | Quaternion w, x, y, z (sensor → segment) | Orientation spread | Clock offset* |
|---|---|---|---|---|
| Foot | x forward, ~32° down the instep; y left (medial); z out of the top of the foot | 0.9578, 0.0867, 0.2717, -0.0348 | 10.4° | -7.7 ms |
| Shank | x down the shank; y left (medial); z out of the shin, turned 12° laterally | 0.7219, 0.1197, 0.6812, -0.0208 | 13.5° | -12.0 ms |
| Thigh | x up the thigh; y backward (into the thigh); z right (lateral) | 0.5334, 0.5556, -0.4640, 0.4376 | 3.7° | -12.3 ms |
| Trunk | not reliable | — | — | -24.6 ms |

*Clock offset: t_mocap = t_imu + offset, so negative means the IMU samples lag the mocap.

## Foot and shank by activity

| IMU | Activity | Trials | Position (x, y, z) cm | Trial-to-trial SD (x, y, z) cm |
|---|---|---|---|---|
| Foot | treadmill | 6 | (11.5, 0.1, 3.6) | (0.1, 0.3, 0.0) |
| Foot | levelground | 6 | (11.9, 0.3, 3.3) | (0.3, 1.5, 1.0) |
| Foot | ramp | 6 | (11.8, 0.2, 2.9) | (0.2, 0.7, 0.2) |
| Foot | stair | 6 | (11.5, -0.2, 2.5) | (0.5, 0.1, 0.4) |
| Shank | treadmill | 6 | (1.3, 3.4, -25.3) | (0.3, 0.7, 0.9) |
| Shank | levelground | 6 | (1.7, 3.8, -26.8) | (1.0, 1.0, 1.0) |
| Shank | ramp | 6 | (2.3, -1.6, -23.6) | (0.2, 0.6, 0.3) |
| Shank | stair | 6 | (2.2, -0.9, -24.4) | (0.2, 0.2, 0.3) |

## Notes

- **Mounting orientation is constant**: it is how the sensor sits on its own segment. The shank IMU is above the ankle, so ankle motion does not affect it; the foot IMU turns with the foot.
- **Foot pitch drifts with ankle angle** (checked on AB09): up to ~±10° at the extremes (push-off, deep dorsiflexion), likely shoe/midfoot bending that the rigid model foot does not capture.
- **Rejected trials**: trials whose accelerometer the model cannot explain (residual RMS > 0.65 × signal RMS: saturation at ±8 g / ±16 rad/s, damped or corrupted recordings) are left out; see `summary.json` → `rejected_trials`.
- **Thigh accelerometer is sign-inverted** relative to its gyro in this dataset; multiply it by −1.
- Frames: (x, y, z)_MuJoCo = (x, −z, y)_OpenSim.
