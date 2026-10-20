# AB12 IMU placement

Estimated from mocap IK + raw IMU data with the MuJoCo model converted from `AB12.osim`. Median over trials (treadmill, levelground, ramp, stair).

Position is in each segment's own frame, in cm: **x forward, y left, z up**, measured from the segment origin. IQR is the spread between trials.

## Position

| IMU | Segment (origin) | Trials used | Position (x, y, z) cm | Trial spread IQR (x, y, z) cm | Place on the body |
|---|---|---|---|---|---|
| Foot | `calcn_r` (heel) | 20/24 | (13.3, -0.9, 1.7) | (0.6, 1.5, 0.3) | Top of the foot, 76% of the way from heel to toe joint (midfoot/laces), centred |
| Shank | `tibia_r` (knee centre) | 20/24 | (2.3, 0.4, -26.5) | (1.1, 4.5, 3.4) | Front of the shin, 64% of the way down, centred |
| Thigh | `femur_r` (hip centre) | 24/24 | (5.2, 0.6, -19.7) | (1.9, 3.2, 1.8) | Front of the thigh, 52% of the way down |
| Trunk | `torso` (lumbar joint) | 6/24 | not reliable | — | Not reliable (lumbar locked in IK, no torso markers) |

## Orientation (sensor axes in the segment frame)

| IMU | Sensor axes | Quaternion w, x, y, z (sensor → segment) | Orientation spread | Clock offset* |
|---|---|---|---|---|
| Foot | x forward, ~43° down the instep; y left (medial); z out of the top of the foot | 0.9203, 0.1301, 0.3586, -0.0870 | 3.5° | -10.1 ms |
| Shank | x down the shank; y left (medial); z out of the shin, straight forward | 0.6856, 0.0555, 0.7251, 0.0343 | 4.2° | -12.9 ms |
| Thigh | x up the thigh; y backward (into the thigh); z right (lateral) | 0.5622, 0.5448, -0.4710, 0.4066 | 50.1° | -13.4 ms |
| Trunk | not reliable | — | — | -32.1 ms |

*Clock offset: t_mocap = t_imu + offset, so negative means the IMU samples lag the mocap.

## Foot and shank by activity

| IMU | Activity | Trials | Position (x, y, z) cm | Trial-to-trial SD (x, y, z) cm |
|---|---|---|---|---|
| Foot | treadmill | 6 | (13.8, -2.0, 1.8) | (0.2, 0.4, 0.1) |
| Foot | levelground | 6 | (13.4, -1.1, 1.7) | (1.5, 4.5, 1.3) |
| Foot | ramp | 6 | (13.2, -0.5, 1.5) | (0.4, 0.4, 0.3) |
| Foot | stair | 6 | (13.2, -0.3, 1.8) | (6.5, 4.2, 0.6) |
| Shank | treadmill | 6 | (1.4, 4.4, -23.4) | (0.2, 0.8, 0.5) |
| Shank | levelground | 6 | (2.1, 2.3, -25.4) | (2.2, 5.6, 1.5) |
| Shank | ramp | 6 | (2.6, -0.8, -27.0) | (0.4, 0.8, 1.1) |
| Shank | stair | 6 | (3.0, -0.1, -27.6) | (3.4, 1.6, 12.4) |

## Notes

- **Mounting orientation is constant**: it is how the sensor sits on its own segment. The shank IMU is above the ankle, so ankle motion does not affect it; the foot IMU turns with the foot.
- **Foot pitch drifts with ankle angle** (checked on AB09): up to ~±10° at the extremes (push-off, deep dorsiflexion), likely shoe/midfoot bending that the rigid model foot does not capture.
- **Rejected trials**: trials whose accelerometer the model cannot explain (residual RMS > 0.65 × signal RMS: saturation at ±8 g / ±16 rad/s, damped or corrupted recordings) are left out; see `summary.json` → `rejected_trials`.
- **Thigh accelerometer is sign-inverted** relative to its gyro in this dataset; multiply it by −1.
- Frames: (x, y, z)_MuJoCo = (x, −z, y)_OpenSim.
