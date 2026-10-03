"""OpenSim .osim (gait2392-style CustomJoints) -> MuJoCo MJCF.

Keeps the subject's exact segment frames, so IK coordinates play back 1:1:
  * joint centres from the parent offset frames (already scaled in the .osim)
  * rotational TransformAxes -> hinges in the same (body-fixed) order and axes
  * coordinate-dependent translations (knee SimmSplines) -> slide joints whose
    values are computed from the driving coordinate ("dependent joints")
  * markers -> sites (for validating the conversion against measured markers)

OpenSim coordinate names and sign conventions are kept unchanged (e.g. knee
flexion is negative, as in the .osim). Frames: v_mj = (x_os, -z_os, y_os).
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from .model import C_OS2MJ


class SimmSpline:
    """Forsythe-Malcolm-Moler cubic spline, as used by OpenSim's SimmSpline."""

    def __init__(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        n = len(x)
        b, c, d = np.zeros(n), np.zeros(n), np.zeros(n)
        if n < 3:
            b[:] = (y[-1] - y[0]) / (x[-1] - x[0])
        else:
            nm1 = n - 1
            d[0] = x[1] - x[0]
            c[1] = (y[1] - y[0]) / d[0]
            for i in range(1, nm1):
                d[i] = x[i + 1] - x[i]
                b[i] = 2.0 * (d[i - 1] + d[i])
                c[i + 1] = (y[i + 1] - y[i]) / d[i]
                c[i] = c[i + 1] - c[i]
            b[0], b[nm1] = -d[0], -d[n - 2]
            c[0] = c[nm1] = 0.0
            if n > 3:
                c[0] = c[2] / (x[3] - x[1]) - c[1] / (x[2] - x[0])
                c[nm1] = c[n - 2] / (x[nm1] - x[n - 3]) - c[n - 3] / (x[n - 2] - x[n - 4])
                c[0] = c[0] * d[0] ** 2 / (x[3] - x[0])
                c[nm1] = -c[nm1] * d[n - 2] ** 2 / (x[nm1] - x[n - 4])
            for i in range(1, n):
                t = d[i - 1] / b[i - 1]
                b[i] -= t * d[i - 1]
                c[i] -= t * c[i - 1]
            c[nm1] /= b[nm1]
            for i in range(nm1 - 1, -1, -1):
                c[i] = (c[i] - d[i] * c[i + 1]) / b[i]
            b[nm1] = (y[nm1] - y[n - 2]) / d[n - 2] + d[n - 2] * (c[n - 2] + 2.0 * c[nm1])
            for i in range(nm1):
                b[i] = (y[i + 1] - y[i]) / d[i] - d[i] * (c[i + 1] + 2.0 * c[i])
                d[i] = (c[i + 1] - c[i]) / d[i]
                c[i] = 3.0 * c[i]
            c[nm1] = 3.0 * c[nm1]
            d[nm1] = d[n - 2]
        self.x, self.y, self.b, self.c, self.d = x, y, b, c, d

    def __call__(self, q):
        q = np.asarray(q, float)
        i = np.clip(np.searchsorted(self.x, q, side="right") - 1, 0, len(self.x) - 1)
        dx = q - self.x[i]
        out = self.y[i] + dx * (self.b[i] + dx * (self.c[i] + dx * self.d[i]))
        # linear extrapolation outside the knots
        lo, hi = q < self.x[0], q > self.x[-1]
        out[lo] = self.y[0] + self.b[0] * (q[lo] - self.x[0])
        out[hi] = self.y[-1] + self.b[-1] * (q[hi] - self.x[-1])
        return out


def _vec(s):
    return np.array([float(v) for v in s.split()])


def _function(el):
    """TransformAxis function -> python callable of the coordinate value."""
    fn = el.find("*[@name='function']")
    if fn is None:
        return lambda q: np.zeros_like(q)
    scale = 1.0
    if fn.tag == "MultiplierFunction":
        scale = float(fn.findtext("scale", "1"))
        fn = fn.find("function/*")
    if fn.tag == "Constant":
        v = float(fn.findtext("value")) * scale
        return lambda q, v=v: np.full_like(np.asarray(q, float), v)
    if fn.tag == "LinearFunction":
        a, b0 = _vec(fn.findtext("coefficients"))
        return lambda q, a=a * scale, b0=b0 * scale: a * np.asarray(q, float) + b0
    if fn.tag in ("SimmSpline", "NaturalCubicSpline", "GCVSpline"):
        sp = SimmSpline(_vec(fn.findtext("x")), _vec(fn.findtext("y")))
        return lambda q, sp=sp, s=scale: s * sp(np.asarray(q, float))
    raise NotImplementedError(f"OpenSim function {fn.tag}")


@dataclass
class OsimModel:
    xml: str
    coordinates: list                    # independent OpenSim coordinates
    dependent: dict = field(default_factory=dict)  # mj joint -> (coord, fn)
    markers: list = field(default_factory=list)
    bodies: list = field(default_factory=list)

    def coords_with_dependents(self, coords: dict) -> dict:
        T = len(next(iter(coords.values())))
        out = dict(coords)
        for jname, (cname, fn) in self.dependent.items():
            out[jname] = fn(coords.get(cname, np.zeros(T)))
        return out


def _q(R):
    return " ".join(f"{v:.8g}" for v in Rot.from_matrix(R).as_quat(scalar_first=True))


def _p(v):
    return " ".join(f"{x:.8g}" for x in v)


def osim_to_mjcf(path, imus=(), marker_sites=True) -> OsimModel:
    root = ET.parse(path).getroot()
    model = root.find("Model")
    bodies = {b.get("name"): b for b in model.find("BodySet/objects")}
    joints = list(model.find("JointSet/objects"))
    C = C_OS2MJ

    children, parent_of, joint_of = {}, {}, {}
    for j in joints:
        frames = {f.get("name"): f for f in j.find("frames")}
        pf, cf = frames[j.findtext("socket_parent_frame")], frames[j.findtext("socket_child_frame")]
        pbody = pf.findtext("socket_parent").split("/")[-1]
        cbody = cf.findtext("socket_parent").split("/")[-1]
        if np.any(_vec(cf.findtext("translation"))) or np.any(_vec(cf.findtext("orientation"))):
            raise NotImplementedError(f"{j.get('name')}: non-zero child offset frame")
        children.setdefault(pbody, []).append(cbody)
        parent_of[cbody] = pbody
        joint_of[cbody] = (j, pf)

    coordinates, dependent = [], {}
    markers = {}
    for m in model.iter("Marker"):
        b = m.findtext("socket_parent_frame").split("/")[-1]
        markers.setdefault(b, []).append((m.get("name"), C @ _vec(m.findtext("location"))))

    imu_by_body = {}
    for s in imus:
        imu_by_body.setdefault(s.body, []).append(s)

    def body_xml(name, depth):
        j, pf = joint_of[name]
        jn = j.get("name")
        pos = C @ _vec(pf.findtext("translation"))
        Rp = Rot.from_euler("XYZ", _vec(pf.findtext("orientation"))).as_matrix()  # OpenSim body-fixed XYZ
        Rp = C @ Rp @ C.T
        ind = "  " * depth
        lines = [f'{ind}<body name="{name}" pos="{_p(pos)}" quat="{_q(Rp)}">']
        st = j.find("SpatialTransform")
        axes = {ta.get("name"): ta for ta in st}
        # translations first (parent-frame axes), then body-fixed rotations
        for tn in ("translation1", "translation2", "translation3"):
            ta = axes[tn]
            coord = (ta.findtext("coordinates") or "").strip()
            ax = C @ _vec(ta.findtext("axis"))
            fn = _function(ta)
            if not coord:
                v = float(fn(np.zeros(1))[0])
                if abs(v) > 0:       # constant offset along a parent-frame axis
                    pos = pos + v * ax
                    lines[0] = f'{ind}<body name="{name}" pos="{_p(pos)}" quat="{_q(Rp)}">'
                continue
            elm = ta.find("*[@name='function']")
            if elm.tag == "LinearFunction" and np.allclose(_vec(elm.findtext("coefficients")), [1, 0]):
                lines.append(f'{ind}  <joint name="{coord}" type="slide" axis="{_p(ax)}"/>')
                coordinates.append(coord)
            else:
                jname = f"{jn}_{tn}"
                lines.append(f'{ind}  <joint name="{jname}" type="slide" axis="{_p(ax)}"/>')
                dependent[jname] = (coord, fn)
        for rn in ("rotation1", "rotation2", "rotation3"):
            ta = axes[rn]
            coord = (ta.findtext("coordinates") or "").strip()
            ax = C @ _vec(ta.findtext("axis"))
            fn = _function(ta)
            if not coord:
                if abs(float(fn(np.zeros(1))[0])) > 1e-12:
                    raise NotImplementedError(f"{jn}.{rn}: constant rotation")
                continue
            elm = ta.find("*[@name='function']")
            if not (elm.tag == "LinearFunction" and np.allclose(_vec(elm.findtext("coefficients")), [1, 0])):
                raise NotImplementedError(f"{jn}.{rn}: non-identity rotation function")
            lines.append(f'{ind}  <joint name="{coord}" type="hinge" axis="{_p(ax / np.linalg.norm(ax))}"/>')
            coordinates.append(coord)
        b = bodies[name]
        mass = max(float(b.findtext("mass", "0.1")), 0.05)
        com = C @ _vec(b.findtext("mass_center", "0 0 0"))
        lines.append(f'{ind}  <inertial pos="{_p(com)}" mass="{mass:.6g}" diaginertia="{_p([mass * 0.01] * 3)}"/>')
        if np.linalg.norm(com) > 1e-3:
            lines.append(f'{ind}  <geom type="capsule" fromto="0 0 0 {_p(2 * com)}" size="{0.035 if mass > 2 else 0.02}" material="skin" mass="0"/>')
        else:
            lines.append(f'{ind}  <geom type="sphere" size="0.025" material="skin" mass="0"/>')
        if marker_sites:
            for mn, loc in markers.get(name, []):
                lines.append(f'{ind}  <site name="m_{mn}" pos="{_p(loc)}" size="0.008" rgba="0.2 0.4 1 1" group="2"/>')
        for s in imu_by_body.get(name, []):
            lines.append(f'{ind}  <site name="{s.name}" type="box" size="0.02 0.015 0.006" pos="{_p(s.pos)}" '
                         f'quat="{_p(s.quat)}" rgba="{s.rgba}" group="1"/>')
        for c in children.get(name, []):
            lines.append(body_xml(c, depth + 1))
        lines.append(f"{ind}</body>")
        return "\n".join(lines)

    roots = [c for c in children.get("ground", [])]
    world = "\n".join(body_xml(r, 2) for r in roots)
    sensors = ""
    if imus:
        sensors = "<sensor>\n" + "\n".join(
            f'<accelerometer name="{s.name}_acc" site="{s.name}"/>\n<gyro name="{s.name}_gyr" site="{s.name}"/>'
            for s in imus) + "\n</sensor>"
    xml = f"""<mujoco model="{model.get('name')}">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.005" gravity="0 0 -9.81"><flag contact="disable"/></option>
  <visual><global offwidth="1280" offheight="720"/></visual>
  <default><joint limited="false" damping="0"/><geom contype="0" conaffinity="0"/></default>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.2 0.3 0.4" rgb2="0.3 0.4 0.5" width="512" height="512"/>
    <material name="grid" texture="grid" texrepeat="2 2" texuniform="true"/>
    <material name="skin" rgba="0.8 0.65 0.55 0.7"/>
  </asset>
  <worldbody>
    <light pos="0 0 4" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="grid"/>
{world}
  </worldbody>
  {sensors}
</mujoco>
"""
    return OsimModel(xml=xml, coordinates=coordinates, dependent=dependent,
                     markers=[m for ms in markers.values() for m, _ in ms],
                     bodies=list(bodies))


def segment_lengths(osim_path) -> dict:
    """Right-side segment lengths [m] at the zero pose: femur (hip->knee centre),
    tibia (knee->ankle centre), foot (heel/calcn origin -> MTP joint)."""
    import mujoco
    from .kinematics import Walker
    o = osim_to_mjcf(osim_path, marker_sites=False)
    w = Walker(o.xml)
    coords = o.coords_with_dependents({c: np.zeros(1) for c in o.coordinates})
    w.data.qpos[:] = w.qpos_from_coords(coords, 1)[0]
    mujoco.mj_kinematics(w.model, w.data)
    x = lambda b: w.data.xpos[mujoco.mj_name2id(w.model, mujoco.mjtObj.mjOBJ_BODY, b)].copy()
    return dict(femur=float(np.linalg.norm(x("tibia_r") - x("femur_r"))),
                tibia=float(np.linalg.norm(x("talus_r") - x("tibia_r"))),
                foot=float(np.linalg.norm(x("toes_r") - x("calcn_r"))))
