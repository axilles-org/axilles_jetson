"""Kinematic playback of joint trajectories through the MuJoCo walker.

For each frame we set (qpos, qvel, qacc), run inverse dynamics (which also
computes body accelerations) and read out, per segment and in the segment's
own frame:

    omega : angular velocity                      [rad/s]
    alpha : angular acceleration                  [rad/s^2]
    f     : specific force at the frame origin    [m/s^2]  (= R^T (a - g))

An accelerometer rigidly attached at r (segment frame) with orientation Q
(sensor -> segment) then reads   Q^T (f + alpha x r + omega x (omega x r))
and a gyro reads                 Q^T omega.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .model import SEGMENTS
from .signal_utils import differentiate

TRANSLATIONS = {"pelvis_tx", "pelvis_ty", "pelvis_tz"}


@dataclass
class SegmentKin:
    t: np.ndarray
    omega: np.ndarray
    alpha: np.ndarray
    f: np.ndarray
    R: np.ndarray   # (T,3,3) segment -> world
    p: np.ndarray   # (T,3) origin in world


class Walker:
    def __init__(self, xml: str):
        self.xml = xml
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.joint_names = [m.joint(i).name for i in range(m.njnt)]
        self.qadr = {n: int(m.jnt_qposadr[i]) for i, n in enumerate(self.joint_names)}

    def qpos_from_coords(self, coords: dict[str, np.ndarray], T: int) -> np.ndarray:
        q = np.zeros((T, self.model.nq))
        for name, series in coords.items():
            if name in self.qadr:
                q[:, self.qadr[name]] = series
        return q

    def playback(self, t: np.ndarray, qpos: np.ndarray, fc: float | None = 6.0,
                 bodies=SEGMENTS, derivs=None):
        """Filter + differentiate qpos, then compute segment kinematics and
        any IMU sensor outputs defined in the model.

        derivs: optional (qpos, qvel, qacc) to skip numerical differentiation.
        Returns (dict body -> SegmentKin, dict sensor_name -> (T,3) array).
        """
        m, d = self.model, self.data
        if derivs is None:
            qf, qd, qdd = differentiate(qpos, t, fc)
        else:
            qf, qd, qdd = derivs
        T = len(t)
        ids = {b: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in bodies}
        out = {b: dict(omega=np.zeros((T, 3)), alpha=np.zeros((T, 3)), f=np.zeros((T, 3)),
                       R=np.zeros((T, 3, 3)), p=np.zeros((T, 3))) for b in bodies}
        sens = np.zeros((T, m.nsensordata))
        res = np.zeros(6)
        for k in range(T):
            d.qpos[:] = qf[k]
            d.qvel[:] = qd[k]
            d.qacc[:] = qdd[k]
            mujoco.mj_inverse(m, d)
            mujoco.mj_rnePostConstraint(m, d)   # fills cacc (body accelerations)
            for b, bid in ids.items():
                o = out[b]
                mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_XBODY, bid, res, 1)
                o["omega"][k] = res[:3]
                mujoco.mj_objectAcceleration(m, d, mujoco.mjtObj.mjOBJ_XBODY, bid, res, 1)
                o["alpha"][k] = res[:3]
                o["f"][k] = res[3:]          # includes -gravity (specific force)
                o["R"][k] = d.xmat[bid].reshape(3, 3)
                o["p"][k] = d.xpos[bid]
            sens[k] = d.sensordata
        segs = {b: SegmentKin(t=t, **v) for b, v in out.items()}
        sensors = {}
        for i in range(m.nsensor):
            s = m.sensor(i)
            a = int(m.sensor_adr[i])
            sensors[s.name] = sens[:, a:a + 3].copy()
        return segs, sensors


def predict_imu(seg: SegmentKin, r: np.ndarray, Q: np.ndarray):
    """Ideal IMU output at position r with orientation Q (sensor -> segment)."""
    a_pt = seg.f + np.cross(seg.alpha, r) + np.cross(seg.omega, np.cross(seg.omega, r))
    return a_pt @ Q, seg.omega @ Q
