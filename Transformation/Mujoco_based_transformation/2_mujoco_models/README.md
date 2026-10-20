# MuJoCo models of the IMU placement

One model per Georgia Tech subject, converted from the subject's own scaled OpenSim
model.

| File | What it shows |
|---|---|
| `<AB>_imu_placement.xml` | Subject model with **both** IMU sets (described below) |
| `<AB>_imu_placement.png` | Renders of that model in the subject's static standing pose: lower leg from the side and front, foot close-ups |
| `georgia_tech_imus_only/<AB>_with_imus.xml` | Subject model with the four estimated Georgia Tech IMUs (foot, shank, thigh, trunk), each with MuJoCo `accelerometer` and `gyro` sensors attached |
| `georgia_tech_imus_only/AB09_render.png` | Render of AB09 with its Georgia Tech IMUs |

What's drawn in `<AB>_imu_placement.xml`:
- **Blue boxes:** Georgia Tech foot/shank IMUs at the estimated poses.
- **Orange boxes:** Axilles exo foot/shank IMUs, at the positions from the exo
  geometry, with the orientation from the exo IMU calibration (Sept 11 mounting).
- **Grey:** encoder (on the exo rotation axis) and simple struts.
- **Dark box:** shoe outline, from the subject's shoe markers.
- **Thin arrows on every IMU:** its axes, x = red, y = green, z = blue.

## Viewing

```bash
python -m mujoco.viewer --mjcf=2_mujoco_models/AB09_imu_placement.xml        # static model
python scripts/build_placement_model.py --subject AB09 --view                 # standing pose
python scripts/build_placement_model.py --subject AB09 --view --trial treadmill_01_01   # walking
```

The last two need the raw Camargo subject folder (see the top-level README).

## Exo geometry used

- **Exo:** right leg, exo on the lateral side.
- **Encoder:** 25.94 mm forward of the shoe's heel end and 107.08 mm lateral of the
  shoe centre line, at ankle-joint height.
- **Foot IMU:** (+32.02, +39.14, −51.98) mm from the encoder (exo frame: x forward,
  y left, z up).
- **Shank IMU:** (0, +10.97, +168.34) mm from the encoder.

On the Georgia Tech subjects, this places the encoder **2.5–3.9 cm behind the
anatomical ankle joint centre**. That is either real exo-axis misalignment or a
different heel reference; the renders are the easiest way to judge it.
