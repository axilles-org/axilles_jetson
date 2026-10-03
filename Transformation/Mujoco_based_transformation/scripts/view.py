"""Play mocap through the walker and show the estimated IMU sites.

  python scripts/view.py --model results/trial01/walker_with_imus.xml --mocap trial01_ik.mot
  python scripts/view.py ... --snapshot frame.png --time 3.0     # headless (MUJOCO_GL=egl)
"""
import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from imu_locator.io import load_coordinates  # noqa: E402
from imu_locator.kinematics import Walker  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mocap", required=True)
    ap.add_argument("--coord-map")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--snapshot", help="render one frame to PNG instead of opening the viewer")
    ap.add_argument("--time", type=float, default=None, help="mocap time for --snapshot")
    a = ap.parse_args()

    w = Walker(Path(a.model).read_text())
    t, coords = load_coordinates(a.mocap, coord_map=a.coord_map)
    q = w.qpos_from_coords(coords, len(t))
    m, d = w.model, w.data

    if a.snapshot:
        k = int(np.searchsorted(t, a.time if a.time is not None else t[len(t) // 2]))
        d.qpos[:] = q[min(k, len(t) - 1)]
        mujoco.mj_forward(m, d)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")] - [0, 0, 0.3]
        cam.distance, cam.azimuth, cam.elevation = 2.6, 120, -12
        with mujoco.Renderer(m, 720, 1280) as r:
            opt = mujoco.MjvOption()
            opt.sitegroup[:] = 1
            r.update_scene(d, cam, opt)
            import matplotlib.pyplot as plt
            plt.imsave(a.snapshot, r.render())
        print(f"wrote {a.snapshot}")
        return


    pelvis = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    with mujoco.viewer.launch_passive(m, d) as v:
        v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        v.cam.trackbodyid = pelvis
        v.cam.distance = 3.0
        while v.is_running():
            t0 = time.time()
            for k in range(len(t)):
                if not v.is_running():
                    break
                d.qpos[:] = q[k]
                mujoco.mj_forward(m, d)
                v.sync()
                lag = (t[k] - t[0]) / a.speed - (time.time() - t0)
                if lag > 0:
                    time.sleep(lag)


if __name__ == "__main__":
    main()
