"""Validate the whole pipeline on synthetic walking where the truth is known.

1. Synthetic gait -> MuJoCo walker with IMUs at secret poses -> MuJoCo
   accelerometer/gyro readings (1 kHz), then down-sampled to 148 Hz with noise,
   biases and a clock offset.
2. Joint angles down-sampled to 100 Hz with noise and written as an OpenSim .mot.
3. The estimator sees only the .mot, the IMU CSVs and the subject file, and has
   to find segment, time offset, orientation and position for each IMU.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from imu_locator import ImuSite, Subject, Walker, build_mjcf  # noqa: E402
from imu_locator.gait import walking  # noqa: E402
from imu_locator.kinematics import TRANSLATIONS  # noqa: E402
from imu_locator.pipeline import run  # noqa: E402
from imu_locator.signal_utils import interp_rows  # noqa: E402


def q(euler_deg):
    return tuple(Rot.from_euler("xyz", euler_deg, degrees=True).as_quat(scalar_first=True))


TRUE_IMUS = [
    ImuSite("imu_thigh", "femur_r", (0.02, -0.075, -0.17), q([90, 0, -90])),  # lateral thigh
    ImuSite("imu_shank", "tibia_r", (0.05, -0.01, -0.14), q([0, -90, 15])),   # anterior shank
    ImuSite("imu_foot", "foot_r", (0.07, 0.0, -0.01), q([180, 5, 10])),       # dorsum
    ImuSite("imu_pelvis", "pelvis", (-0.17, 0.0, 0.0), q([0, 90, 0])),        # sacrum
    ImuSite("imu_trunk", "torso", (-0.12, 0.0, 0.32), q([-90, 0, 90])),       # upper back
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--duration", type=float, default=40.0)
    ap.add_argument("--fc", type=float, default=6.0)
    ap.add_argument("--mocap-noise", type=float, default=1.0,
                    help="scale of mocap noise (1 = 0.3 deg / 0.5 mm)")
    ap.add_argument("--length-error", type=float, default=0.0,
                    help="error [m] added to thigh & shank length given to the estimator")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)

    subject = Subject(height=1.72, mass=68, thigh_length=0.415, shank_length=0.42)
    subject.to_json(out / "subject.json")

    # ---- ground truth simulation at 1 kHz
    t, coords = walking(duration=args.duration, fs=1000.0)
    walker = Walker(build_mjcf(subject, TRUE_IMUS))
    qp = walker.qpos_from_coords(coords, len(t))
    qv = np.gradient(qp, t, axis=0)
    qa = np.gradient(qv, t, axis=0)
    _, sensors = walker.playback(t, qp, derivs=(qp, qv, qa), bodies=[])
    (out / "walker_true_imus.xml").write_text(walker.xml)

    # ---- mocap: 100 Hz, noisy, OpenSim .mot (degrees, gait2392 knee sign)
    tm = np.arange(0, args.duration, 0.01)
    cols = list(coords)
    M = interp_rows(t, np.column_stack([coords[c] for c in cols]), tm)
    lines = ["synthetic_ik", "version=1", f"nRows={len(tm)}", f"nColumns={len(cols) + 1}",
             "inDegrees=yes", "endheader", "\t".join(["time"] + cols)]
    for k in range(len(tm)):
        row = []
        for j, c in enumerate(cols):
            if c in TRANSLATIONS:
                row.append(M[k, j] + rng.normal(0, 0.0005 * args.mocap_noise))
            else:
                v = np.degrees(M[k, j]) + rng.normal(0, 0.3 * args.mocap_noise)
                row.append(-v if c.startswith("knee") else v)
        lines.append("\t".join(f"{x:.6f}" for x in [tm[k]] + row))
    (out / "ik.mot").write_text("\n".join(lines) + "\n")

    # ---- IMUs: 148 Hz, own clock (offset), noise, bias; acc in g, gyro in deg/s
    offset_true = 0.137
    ti = np.arange(0.2, args.duration - 0.2, 1 / 148.0)      # IMU clock
    truth = {}
    for s in TRUE_IMUS:
        tq = ti + offset_true                                   # corresponding mocap time
        acc = interp_rows(t, sensors[f"{s.name}_acc"], tq)
        gyr = interp_rows(t, sensors[f"{s.name}_gyr"], tq)
        ba, bg = rng.normal(0, 0.15, 3), rng.normal(0, 0.02, 3)
        acc = acc + ba + rng.normal(0, 0.08, acc.shape)
        gyr = gyr + bg + rng.normal(0, 0.015, gyr.shape)
        data = np.column_stack([ti, acc / 9.80665, np.degrees(gyr)])
        np.savetxt(out / f"{s.name}.csv", data, delimiter=",", fmt="%.6f",
                   header="time,acc_x,acc_y,acc_z,gyr_x,gyr_y,gyr_z", comments="")
        truth[s.name] = dict(segment=s.body, pos=list(s.pos), quat=list(s.quat),
                             acc_bias=ba.tolist(), gyro_bias=bg.tolist())
    (out / "truth.json").write_text(json.dumps(dict(time_offset=offset_true, imus=truth), indent=2))

    # ---- blind estimation
    est_subject = Subject.from_json(out / "subject.json")
    est_subject.thigh_length += args.length_error
    est_subject.shank_length += args.length_error
    res, _ = run(est_subject, out / "ik.mot",
                 {s.name: out / f"{s.name}.csv" for s in TRUE_IMUS},
                 fc=args.fc, out_dir=out / "estimate")

    print("\n================ error vs truth ================")
    print(f"{'imu':12s} {'segment':10s} {'dt [ms]':>8s} {'pos err [cm] (x,y,z)':>28s} {'rot err [deg]':>14s}")
    for s in TRUE_IMUS:
        e = res[s.name]
        R_true = Rot.from_quat(s.quat, scalar_first=True)
        rot_err = np.degrees((R_true.inv() * Rot.from_matrix(e.R)).magnitude())
        perr = (e.r - np.array(s.pos)) * 100
        ok = "OK " if e.segment == s.body else "BAD"
        print(f"{s.name:12s} {ok}{e.segment:7s} {1000 * (e.time_offset - offset_true):8.2f} "
              f"{'(' + ', '.join(f'{v:6.2f}' for v in perr) + ')':>28s} {rot_err:14.2f}")


if __name__ == "__main__":
    main()
