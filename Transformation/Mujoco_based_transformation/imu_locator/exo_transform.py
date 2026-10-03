"""Georgia Tech (Camargo) IMU data -> Axilles exo IMU frame.

Target frame: the exo frame (x forward, y left, z up when standing), located at
the exo's foot / shank IMU positions. Georgia Tech segment frames use the same
axis convention in the neutral pose, so the exo frame is taken as the segment
frame.

For a GT sensor with orientation R (GT sensor -> segment) at position r_gt and an
exo IMU at r_exo (both in the segment frame), with d = r_exo - r_gt:

    gyro_exo = R gyro_gt
    acc_exo  = R acc_gt + alpha x d + omega x (omega x d)      (omega = gyro_exo)

The gyro needs only the rotation; the accelerometer also needs the lever arm d,
because a different point on the same rigid segment accelerates differently.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from .kinematics import Walker
from .osim import osim_to_mjcf
from .signal_utils import lowpass

G = 9.80665


@dataclass
class ExoGeometry:
    """Axilles exo geometry (metres, exo frame: +x forward, +y left, +z up).
    The exo is worn on the right leg with the encoder on the lateral side."""
    encoder_from_heel_x: float = 0.02594       # encoder ahead of the shoe's heel end
    encoder_from_centre_y: float = -0.10708    # encoder lateral (-y) of the shoe centre line
    foot_imu_from_encoder: tuple = (0.03202, 0.03914, -0.05198)
    shank_imu_from_encoder: tuple = (0.0, 0.01097, 0.16834)
    heel_marker_offset: float = 0.007          # GT heel-marker centre sits ~7 mm behind the shoe
    encoder_height: str = "ankle_joint"        # encoder on the rotation axis, at ankle-joint height
    encoder_x_anchor: str = "heel"             # "heel": heel end + encoder_from_heel_x; "ankle": on the ankle joint centre


def standing_up(osim_path, static_ik_path) -> dict:
    """World 'up' expressed in the calcn and tibia frames during the subject's
    static standing trial: the GT counterpart of the exo's standing-gravity
    calibration (people do not stand at the model's zero pose)."""
    from .camargo import _table
    from .kinematics import TRANSLATIONS
    o = osim_to_mjcf(osim_path)
    w = Walker(o.xml)
    df = _table(static_ik_path)
    coords = {c: (df[c].to_numpy(float) if c in TRANSLATIONS else np.radians(df[c].to_numpy(float)))
              for c in df.columns[1:]}
    q = w.qpos_from_coords(o.coords_with_dependents(coords), len(df))
    ups = {b: [] for b in ("calcn_r", "tibia_r")}
    for k in range(0, len(q), max(1, len(q) // 50)):
        w.data.qpos[:] = q[k]
        mujoco.mj_kinematics(w.model, w.data)
        for b in ups:
            R = w.data.xmat[mujoco.mj_name2id(w.model, mujoco.mjtObj.mjOBJ_BODY, b)].reshape(3, 3)
            ups[b].append(R.T @ np.array([0.0, 0.0, 1.0]))
    return {b: np.mean(v, 0) / np.linalg.norm(np.mean(v, 0)) for b, v in ups.items()}


def exo_imu_positions(osim_path, geo: ExoGeometry = ExoGeometry(), R_s2e=None) -> dict:
    """R_s2e: optional {'foot','shank'} segment -> exo-frame rotations; the IMU
    offsets from the encoder are measured along exo axes, so they are rotated
    back into the segment frame with it."""
    """Exo IMU positions in a GT subject's segment frames (zero pose).

    The shoe is located on the subject from their own shoe markers: the heel end
    from R_Heel, the centre line from R_Heel and the toe markers. Encoder height
    is the subject's ankle joint centre (the exo axis is the ankle axis)."""
    o = osim_to_mjcf(osim_path)
    w = Walker(o.xml)
    m, d = w.model, w.data
    coords = o.coords_with_dependents({c: np.zeros(1) for c in o.coordinates})
    d.qpos[:] = w.qpos_from_coords(coords, 1)[0]
    mujoco.mj_kinematics(m, d)
    bid = lambda b: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)
    site = lambda n: d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "m_" + n)].copy()
    Rc, pc = d.xmat[bid("calcn_r")].reshape(3, 3), d.xpos[bid("calcn_r")].copy()
    Rt, pt = d.xmat[bid("tibia_r")].reshape(3, 3), d.xpos[bid("tibia_r")].copy()
    in_calcn = lambda p: Rc.T @ (p - pc)
    calcn_to_tibia = lambda p: Rt.T @ (Rc @ p + pc - pt)

    heel = in_calcn(site("R_Heel"))
    toe_mid = 0.5 * (in_calcn(site("R_Toe_Lat")) + in_calcn(site("R_Toe_Med")))
    ankle = in_calcn(d.xpos[bid("talus_r")])
    heel_end_x = heel[0] + geo.heel_marker_offset
    centre_y = 0.5 * (heel[1] + toe_mid[1])
    enc_x = heel_end_x + geo.encoder_from_heel_x if geo.encoder_x_anchor == "heel" else ankle[0]
    enc = np.array([enc_x, centre_y + geo.encoder_from_centre_y, ankle[2]])
    R_s2e = R_s2e or {"foot": np.eye(3), "shank": np.eye(3)}
    foot = enc + R_s2e["foot"].T @ np.array(geo.foot_imu_from_encoder)
    shank = calcn_to_tibia(enc) + R_s2e["shank"].T @ np.array(geo.shank_imu_from_encoder)
    return dict(encoder_calcn=enc, encoder_tibia=calcn_to_tibia(enc), foot=foot, shank=shank,
                ankle_calcn=ankle, encoder_minus_ankle=enc - ankle,
                heel_end_x=heel_end_x, shoe_centre_y=centre_y)


@dataclass
class Transform:
    """GT sensor -> exo IMU frame for one segment."""
    R: np.ndarray            # GT sensor axes -> exo frame (3x3), = R_s2e @ R_sensor_to_segment
    r_gt: np.ndarray         # GT sensor position, segment frame [m]
    r_exo: np.ndarray        # exo IMU position, segment frame [m]
    acc_sign: int = 1        # GT accelerometer sign convention (+1 normal)
    R_s2e: np.ndarray = None  # segment frame -> exo frame (standing-based); identity if None
    meta: dict = field(default_factory=dict)
    d_override: np.ndarray = None   # lever arm given directly in the exo frame (averages)

    @property
    def d(self) -> np.ndarray:
        """Lever arm: exo IMU position minus GT sensor position, in the exo frame [m]."""
        if self.d_override is not None:
            return np.asarray(self.d_override, float)
        Rse = np.eye(3) if self.R_s2e is None else self.R_s2e
        return Rse @ (self.r_exo - self.r_gt)

    @property
    def T(self) -> np.ndarray:
        """4x4 homogeneous pose of the GT sensor expressed in the exo IMU frame:
        p_exo = R p_gt - d."""
        T = np.eye(4)
        T[:3, :3] = self.R
        T[:3, 3] = -self.d
        return T

    def apply(self, acc, gyro, fs, acc_in_g=True, fc_lever=10.0):
        """Map GT accel [g or m/s^2] and gyro [rad/s] to the exo IMU frame [m/s^2, rad/s]."""
        a = np.asarray(acc, float) * (G if acc_in_g else 1.0) * self.acc_sign
        w = np.asarray(gyro, float) @ self.R.T
        a = a @ self.R.T
        wf = lowpass(w, fs, fc_lever)                       # lever arm terms from smoothed gyro
        dw = np.gradient(wf, 1.0 / fs, axis=0)
        d = self.d
        a = a + np.cross(dw, d) + np.cross(wf, np.cross(wf, d))
        return a, w

    def to_dict(self):
        r = lambda v: np.round(np.asarray(v, float), 6).tolist()
        out = dict(R_gt_sensor_to_exo=r(self.R), lever_arm_d_m=r(self.d), T_gt_in_exo_4x4=r(self.T),
                   r_gt_segment_m=r(self.r_gt), r_exo_segment_m=r(self.r_exo), acc_sign=self.acc_sign)
        if self.R_s2e is not None:
            out["R_segment_to_exo"] = r(self.R_s2e)
        return {**out, **self.meta}


def rotation_mean(Rs):
    from scipy.spatial.transform import Rotation as Rot
    return Rot.from_matrix(np.asarray(Rs)).mean().as_matrix()


# ----------------------------------------------------------------- exo IMU calibration
def principal_axis(gyro):
    g = gyro - gyro.mean(0)
    w, v = np.linalg.eigh(g.T @ g)
    return v[:, -1], float(w[-1] / w.sum())


def exo_frame_rotation(gravity_dir, flex_axis, sign):
    """Exo-frame rotation of one exo IMU (exo frame <- sensor) from
    z = measured 'up' (standing gravity reading) and y = flexion axis (sign chosen
    by the caller), orthogonalised; x = y cross z."""
    z = np.asarray(gravity_dir, float)
    z = z / np.linalg.norm(z)
    y = sign * (flex_axis - (flex_axis @ z) * z)
    y = y / np.linalg.norm(y)
    x = np.cross(y, z)
    return np.vstack([x, y, z])
