"""Parametric MuJoCo lower-body + trunk walker.

Joint names and sign conventions follow the OpenSim gait2392 / Rajagopal
models so an OpenSim IK .mot file can be played back directly:

    pelvis_tx/ty/tz, pelvis_tilt/list/rotation,
    lumbar_extension/bending/rotation,
    hip_flexion_*, hip_adduction_*, hip_rotation_*,
    knee_angle_*   (positive = flexion, Rajagopal convention),
    ankle_angle_*  (positive = dorsiflexion),
    subtalar_angle_* (positive = inversion)

Frames: MuJoCo world is Z-up (x forward, y left, z up). OpenSim is Y-up
(x forward, y up, z right).  v_mj = (x_os, -z_os, y_os).  At the zero pose every
body frame is aligned with the world, like the OpenSim segment frames.

Body frame origins (where IMU positions are expressed):
    pelvis  : OpenSim pelvis origin (between the ASIS, approx.)
    torso   : lumbar joint
    femur_* : hip joint centre
    tibia_* : knee joint centre
    foot_*  : ankle joint centre
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .subject import Subject

# OpenSim (Y-up) -> MuJoCo (Z-up) rotation, v_mj = C_OS2MJ @ v_os
C_OS2MJ = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
C_MJ2OS = C_OS2MJ.T

SEGMENTS = ["pelvis", "torso", "femur_r", "tibia_r", "foot_r",
            "femur_l", "tibia_l", "foot_l"]


@dataclass
class ImuSite:
    name: str
    body: str
    pos: tuple = (0.0, 0.0, 0.0)          # in body frame [m]
    quat: tuple = (1.0, 0.0, 0.0, 0.0)    # sensor->body, wxyz
    rgba: str = "1 0.5 0 1"


def _v(x):
    return " ".join(f"{float(a):.6g}" for a in x)


def _imu_xml(imus, body):
    out = []
    for s in imus:
        if s.body == body:
            out.append(
                f'<site name="{s.name}" type="box" size="0.02 0.015 0.006" '
                f'pos="{_v(s.pos)}" quat="{_v(s.quat)}" rgba="{s.rgba}" group="1"/>')
    return "\n".join(out)


def _leg_xml(side, d, imus):
    sgn = -1.0 if side == "r" else 1.0      # right leg on -y
    m = 1.0 if side == "r" else -1.0        # mirrored axes for adduction/rotation/subtalar
    Lt, Ls = d["thigh_length"], d["shank_length"]
    fl, ah = d["foot_length"], d["ankle_height"]
    rt, rs = d["thigh_radius"], d["shank_radius"]
    hip = (d["hip_offset_forward"], sgn * d["hip_width"] / 2, -d["hip_offset_down"])
    heel, toe = -0.25 * fl, 0.75 * fl
    fw = 0.045 * d["height"] / 1.8
    return f"""
      <body name="femur_{side}" pos="{_v(hip)}">
        <joint name="hip_flexion_{side}" axis="0 -1 0"/>
        <joint name="hip_adduction_{side}" axis="{m:g} 0 0"/>
        <joint name="hip_rotation_{side}" axis="0 0 {m:g}"/>
        <geom name="thigh_{side}" type="capsule" fromto="0 0 0 0 0 {-Lt:.6g}" size="{rt:.4g}" material="skin"/>
        {_imu_xml(imus, f"femur_{side}")}
        <body name="tibia_{side}" pos="0 0 {-Lt:.6g}">
          <joint name="knee_angle_{side}" axis="0 1 0"/>
          <geom name="shank_{side}" type="capsule" fromto="0 0 0 0 0 {-Ls:.6g}" size="{rs:.4g}" material="skin"/>
          {_imu_xml(imus, f"tibia_{side}")}
          <body name="foot_{side}" pos="0 0 {-Ls:.6g}">
            <joint name="ankle_angle_{side}" axis="0 -1 0"/>
            <joint name="subtalar_angle_{side}" axis="{m:g} 0 0"/>
            <geom name="foot_{side}" type="box" pos="{(heel + toe) / 2:.6g} 0 {-ah / 2:.6g}"
                  size="{(toe - heel) / 2:.6g} {fw:.4g} {ah / 2:.6g}" material="shoe"/>
            {_imu_xml(imus, f"foot_{side}")}
          </body>
        </body>
      </body>"""


def build_mjcf(subject: Subject | None = None, imus: list[ImuSite] = (),
               with_sensors: bool = True) -> str:
    subject = subject or Subject()
    d = subject.resolved()
    H = d["height"]
    s = H / 1.8
    lumbar = (d["lumbar_offset_forward"], 0.0, d["lumbar_offset_up"])
    Ltor = d["torso_length"]
    hw = d["hip_width"]

    sensors = ""
    if with_sensors and imus:
        rows = []
        for i in imus:
            rows.append(f'<accelerometer name="{i.name}_acc" site="{i.name}"/>')
            rows.append(f'<gyro name="{i.name}_gyr" site="{i.name}"/>')
        sensors = "<sensor>\n" + "\n".join(rows) + "\n</sensor>"

    return f"""<mujoco model="imu_locator_walker">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.005" gravity="0 0 -9.81">
    <flag contact="disable"/>
  </option>
  <visual>
    <global offwidth="1280" offheight="720"/>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.6 0.6 0.6"/>
  </visual>
  <default>
    <joint type="hinge" limited="false" damping="0"/>
    <geom contype="0" conaffinity="0" density="1000"/>
  </default>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.2 0.3 0.4" rgb2="0.3 0.4 0.5" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="2 2" texuniform="true"/>
    <material name="skin" rgba="0.8 0.65 0.55 0.6"/>
    <material name="shoe" rgba="0.2 0.2 0.25 0.8"/>
  </asset>
  <worldbody>
    <light pos="0 0 4" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="grid"/>
    <body name="pelvis" pos="0 0 0">
      <joint name="pelvis_tx" type="slide" axis="1 0 0"/>
      <joint name="pelvis_tz" type="slide" axis="0 -1 0"/>
      <joint name="pelvis_ty" type="slide" axis="0 0 1"/>
      <joint name="pelvis_tilt" axis="0 -1 0"/>
      <joint name="pelvis_list" axis="1 0 0"/>
      <joint name="pelvis_rotation" axis="0 0 1"/>
      <geom name="pelvis" type="capsule" fromto="{-0.07 * s:.4g} {hw / 2 + 0.03 * s:.4g} {-0.03 * s:.4g} {-0.07 * s:.4g} {-hw / 2 - 0.03 * s:.4g} {-0.03 * s:.4g}" size="{0.07 * s:.4g}" material="skin"/>
      {_imu_xml(imus, "pelvis")}
      <body name="torso" pos="{_v(lumbar)}">
        <joint name="lumbar_extension" axis="0 -1 0"/>
        <joint name="lumbar_bending" axis="1 0 0"/>
        <joint name="lumbar_rotation" axis="0 0 1"/>
        <geom name="torso" type="capsule" fromto="0 0 {0.05 * s:.4g} 0 0 {Ltor - 0.05 * s:.4g}" size="{0.12 * s:.4g}" material="skin"/>
        <geom name="head" type="sphere" pos="0 0 {Ltor + 0.12 * s:.4g}" size="{0.1 * s:.4g}" material="skin"/>
        {_imu_xml(imus, "torso")}
      </body>
      {_leg_xml("r", d, imus)}
      {_leg_xml("l", d, imus)}
    </body>
  </worldbody>
  {sensors}
</mujoco>
"""
