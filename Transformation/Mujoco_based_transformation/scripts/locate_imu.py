"""Estimate IMU placement(s) from real data.

Example:
  python scripts/locate_imu.py --subject config/subject_example.json \
      --mocap trial01_ik.mot \
      --imu thigh=trial01_thigh_imu.csv --imu shank=trial01_shank_imu.csv \
      --acc-unit g --gyr-unit deg/s --out results/trial01

  --segment thigh=femur_r   restricts an IMU to a known segment (default: auto)
  --prior thigh=0,-0.08,-0.2,0.03   Gaussian prior x,y,z,sigma [m] in segment frame
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from imu_locator import Subject  # noqa: E402
from imu_locator.pipeline import run  # noqa: E402
from imu_locator.plots import plot_fit  # noqa: E402


def kv(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subject", required=True, help="subject JSON (see config/subject_example.json)")
    ap.add_argument("--mocap", required=True, help="OpenSim IK .mot/.sto or CSV of joint coordinates")
    ap.add_argument("--coord-map", help="JSON mapping model joints -> mocap columns (+scale/offset)")
    ap.add_argument("--imu", action="append", required=True, help="name=path.csv (repeatable)")
    ap.add_argument("--segment", action="append", help="name=body (restrict to a segment)")
    ap.add_argument("--prior", action="append", help="name=x,y,z,sigma (metres, segment frame)")
    ap.add_argument("--time-col", default="time")
    ap.add_argument("--acc-cols", default="acc_x,acc_y,acc_z")
    ap.add_argument("--gyr-cols", default="gyr_x,gyr_y,gyr_z")
    ap.add_argument("--acc-unit", default="auto", choices=["auto", "g", "m/s2"])
    ap.add_argument("--gyr-unit", default="auto", choices=["auto", "deg/s", "rad/s"])
    ap.add_argument("--time-unit", type=float, default=1.0, help="multiplier to seconds (1e-3 for ms)")
    ap.add_argument("--fc", type=float, default=6.0, help="low-pass cutoff for mocap AND IMU [Hz]")
    ap.add_argument("--max-lag", type=float, default=0.3,
                    help="max |clock offset| searched [s]; >~0.5 s risks left/right confusion")
    ap.add_argument("--out", default="results")
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args()

    priors = {}
    for k, v in kv(a.prior).items():
        x, y, z, s = map(float, v.split(","))
        priors[k] = (np.array([x, y, z]), s)
    imu_kwargs = dict(time_col=a.time_col, acc_cols=a.acc_cols.split(","),
                      gyr_cols=a.gyr_cols.split(","), acc_unit=a.acc_unit,
                      gyr_unit=a.gyr_unit, time_unit=a.time_unit)
    res, segs = run(Subject.from_json(a.subject), a.mocap, kv(a.imu), fc=a.fc,
                    segment=kv(a.segment), max_lag=a.max_lag, prior=priors, out_dir=a.out,
                    imu_kwargs=imu_kwargs, coord_kwargs=dict(coord_map=a.coord_map))
    if not a.no_plots:
        from imu_locator.io import load_imu_csv
        for name, path in kv(a.imu).items():
            imu = load_imu_csv(path, verbose=False, **imu_kwargs)
            plot_fit(segs[res[name].segment], imu, res[name], a.fc,
                     Path(a.out) / f"fit_{name}.png", title=name)
        print(f"[plots] wrote {a.out}/fit_*.png")


if __name__ == "__main__":
    main()
