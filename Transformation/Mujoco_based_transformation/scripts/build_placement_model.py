"""MuJoCo model of the IMU placement on one Georgia Tech subject, for a visual check.

Shows on the subject's own model (posed from their static standing trial):
  * Georgia Tech foot/shank IMUs at the estimated poses       (blue boxes)
  * Axilles exo foot/shank IMUs from the exo geometry, with the
    orientation from the exo calibration (Sept 11 session)     (orange boxes)
  * the encoder (rotation axis) and simple exo struts/foot plate (grey)
  * a shoe outline from the subject's shoe markers               (dark)
Every IMU site shows its x/y/z axes (red/green/blue) in the renders; in the
viewer turn on Rendering -> Frame -> Site.

  python scripts/build_placement_model.py --subject AB09              # xml + renders
  python scripts/build_placement_model.py --subject AB09 --view       # interactive viewer
  python scripts/build_placement_model.py --subject AB09 --view --trial treadmill_01_01
"""
import argparse
import json
import sys
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from imu_locator.camargo import _table, load_ik  # noqa: E402
from imu_locator.kinematics import TRANSLATIONS, Walker  # noqa: E402
from imu_locator.model import ImuSite  # noqa: E402
from imu_locator.osim import osim_to_mjcf  # noqa: E402

BODY = dict(foot="calcn_r", shank="tibia_r")
SESSION = "Sept 11 (2×60 s)"


def quat(R):
    return tuple(Rot.from_matrix(R).as_quat(scalar_first=True))


def fmt(v):
    return " ".join(f"{x:.5f}" for x in v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="AB09")
    ap.add_argument("--results", default="results/exo_transform")
    ap.add_argument("--out", default="results/exo_transform/placement_model")
    ap.add_argument("--camargo-root", default=".", help="folder with the raw Camargo subject folders")
    ap.add_argument("--view", action="store_true", help="open the interactive MuJoCo viewer")
    ap.add_argument("--trial", help="with --view: play this treadmill trial instead of standing")
    a = ap.parse_args()
    s_ = a.subject
    sdir = Path(a.camargo_root) / s_
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    tr = json.loads((Path(a.results) / "transform_version2_per_subject.json").read_text())[s_]
    cal = json.loads((Path(a.results) / "exo_imu_calibration.json").read_text())
    R_exo_from_sensor = {k: np.array(v) for k, v in cal["R_exo_frame_from_exo_sensor"][SESSION].items()}

    sites, extra = [], {b: [] for b in BODY.values()}
    for s, body in BODY.items():
        t = tr[s]
        R_s2e = np.array(t["R_segment_to_exo"])
        R_gt = R_s2e.T @ np.array(t["R_gt_sensor_to_exo"])          # GT sensor -> segment
        R_ex = R_s2e.T @ R_exo_from_sensor[s]                         # exo sensor -> segment
        sites.append(ImuSite(f"gt_{s}_imu", body, tuple(t["r_gt_segment_m"]), quat(R_gt), "0.16 0.47 0.84 1"))
        sites.append(ImuSite(f"exo_{s}_imu", body, tuple(t["r_exo_segment_m"]), quat(R_ex), "0.92 0.41 0.20 1"))
        # explicit axis arrows (x red, y green, z blue), visible without viewer settings
        for tag, p, R in (("gt", np.array(t["r_gt_segment_m"]), R_gt), ("exo", np.array(t["r_exo_segment_m"]), R_ex)):
            for k, col in enumerate(("1 0.1 0.1 1", "0.1 0.8 0.1 1", "0.1 0.3 1 1")):
                extra[body].append(f'<geom name="{tag}_{s}_axis_{"xyz"[k]}" type="capsule" size="0.0012" '
                                   f'fromto="{fmt(p)} {fmt(p + 0.035 * R[:, k])}" rgba="{col}" '
                                   f'contype="0" conaffinity="0" mass="0"/>')
        enc = np.array(t["encoder_segment_m"])
        y_exo = R_s2e.T @ np.array([0, 1.0, 0])                       # encoder axis = exo y
        grey = 'rgba="0.55 0.55 0.58 1" contype="0" conaffinity="0" mass="0"'
        extra[body].append(f'<geom name="encoder_{s}" type="cylinder" size="0.03 0.012" '
                           f'fromto="{fmt(enc - 0.012 * y_exo)} {fmt(enc + 0.012 * y_exo)}" {grey}/>')
        imu = np.array(t["r_exo_segment_m"])
        top = imu + (imu - enc) * 0.25 if s == "shank" else imu
        extra[body].append(f'<geom name="strut_{s}" type="capsule" size="0.008" '
                           f'fromto="{fmt(enc)} {fmt(top)}" {grey}/>')

    # shoe outline on the foot from the subject's own shoe markers
    o0 = osim_to_mjcf(next((sdir / "osimxml").glob("*.osim")))
    w0 = Walker(o0.xml)
    w0.data.qpos[:] = w0.qpos_from_coords(o0.coords_with_dependents({c: np.zeros(1) for c in o0.coordinates}), 1)[0]
    mujoco.mj_kinematics(w0.model, w0.data)
    bid = lambda b: mujoco.mj_name2id(w0.model, mujoco.mjtObj.mjOBJ_BODY, b)
    Rc, pc = w0.data.xmat[bid("calcn_r")].reshape(3, 3), w0.data.xpos[bid("calcn_r")]
    mk = lambda n: Rc.T @ (w0.data.site_xpos[mujoco.mj_name2id(w0.model, mujoco.mjtObj.mjOBJ_SITE, "m_" + n)] - pc)
    heel, tip, tl, tm = mk("R_Heel"), mk("R_Toe_Tip"), mk("R_Toe_Lat"), mk("R_Toe_Med")
    x0, x1 = heel[0] + 0.007, tip[0] + 0.01
    yc, half_w = 0.5 * (heel[1] + 0.5 * (tl[1] + tm[1])), 0.5 * abs(tm[1] - tl[1]) + 0.01
    z0 = min(tl[2], tm[2]) - 0.02
    extra["calcn_r"].append(f'<geom name="shoe" type="box" pos="{fmt([(x0 + x1) / 2, yc, z0 + 0.03])}" '
                            f'size="{fmt([(x1 - x0) / 2, half_w, 0.03])}" rgba="0.15 0.15 0.18 0.35" '
                            f'contype="0" conaffinity="0" mass="0"/>')

    o = osim_to_mjcf(next((sdir / "osimxml").glob("*.osim")), imus=sites)
    xml = o.xml
    for body, geoms in extra.items():
        i = xml.index(f'<body name="{body}"')
        j = xml.index(">", i) + 1
        xml = xml[:j] + "\n" + "\n".join(geoms) + xml[j:]
    xml = xml.replace('size="0.02 0.015 0.006"', 'size="0.016 0.012 0.005"')
    xml = xml.replace('<global offwidth="1280" offheight="720"/>',
                      '<global offwidth="1600" offheight="1200"/>\n'
                      '    <headlight ambient="0.55 0.55 0.55" diffuse="0.6 0.6 0.6"/>')
    xml = xml.replace('rgba="0.8 0.65 0.55 0.7"', 'rgba="0.85 0.72 0.62 0.28"')
    xml = xml.replace('rgb1="0.2 0.3 0.4" rgb2="0.3 0.4 0.5"', 'rgb1="0.75 0.77 0.8" rgb2="0.85 0.87 0.9"')
    path = out / f"{s_}_imu_placement.xml"
    path.write_text(xml)

    w = Walker(xml)
    m, d = w.model, w.data
    dd = next(p for p in sdir.iterdir() if p.is_dir() and p.name[0].isdigit())
    st = _table(next((dd / "static" / "ik").glob("*.mat")))
    coords = {c: (st[c].to_numpy(float) if c in TRANSLATIONS else np.radians(st[c].to_numpy(float)))
              for c in st.columns[1:]}
    q_stand = w.qpos_from_coords(o.coords_with_dependents(coords), len(st))[len(st) // 2]

    if a.view:
        import time
        if a.trial:
            t, c = load_ik(dd, "treadmill", a.trial)
            Q = w.qpos_from_coords(o.coords_with_dependents(c), len(t))
        else:
            t, Q = np.array([0.0, 1.0]), np.vstack([q_stand, q_stand])
        with mujoco.viewer.launch_passive(m, d) as v:
            pass  # IMU axes are drawn as geoms
            v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            v.cam.trackbodyid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "tibia_r")
            v.cam.distance = 1.0
            while v.is_running():
                t0 = time.time()
                for k in range(len(t)):
                    if not v.is_running():
                        break
                    d.qpos[:] = Q[k]
                    mujoco.mj_forward(m, d)
                    v.sync()
                    lag = (t[k] - t[0]) - (time.time() - t0)
                    if lag > 0:
                        time.sleep(lag)
        return

    d.qpos[:] = q_stand
    mujoco.mj_forward(m, d)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tib = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "talus_r")]
    pel = d.xmat[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")].reshape(3, 3)[:, 0]
    yaw = np.degrees(np.arctan2(pel[1], pel[0]))
    opt = mujoco.MjvOption()
    opt.sitegroup[:] = 0
    opt.sitegroup[1] = 1
    foot_c = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "calcn_r")] + np.array([0.05, 0, 0.02])
    views = [("lower leg, right side (lateral)", yaw + 90, -5, tib + np.array([0, 0, 0.12]), 0.8),
             ("lower leg, front", yaw + 180, -5, tib + np.array([0, 0, 0.12]), 0.8),
             ("foot close-up, right side", yaw + 90, -10, foot_c, 0.38),
             ("foot close-up, from above / right-front", yaw + 135, -50, foot_c, 0.42)]
    imgs = []
    with mujoco.Renderer(m, 900, 800) as r:
        for _, az, el, look, dist in views:
            cam = mujoco.MjvCamera()
            cam.lookat[:] = look
            cam.distance, cam.azimuth, cam.elevation = dist, az, el
            r.update_scene(d, cam, opt)
            imgs.append(r.render())
    fig, axs = plt.subplots(2, 2, figsize=(13, 11.5))
    for ax, im, (lab, *_) in zip(axs.ravel(), imgs, views):
        ax.imshow(im)
        ax.set_title(lab, fontsize=11)
        ax.axis("off")
    fig.suptitle(f"{s_} standing: blue = Georgia Tech IMUs (estimated), orange = Axilles exo IMUs, grey = encoder/struts\n"
                 "IMU axes: red = x, green = y, blue = z", fontsize=12)
    fig.tight_layout()
    png = out / f"{s_}_imu_placement.png"
    fig.savefig(png, dpi=90)
    print(f"wrote {path} and {png}")


if __name__ == "__main__":
    main()
