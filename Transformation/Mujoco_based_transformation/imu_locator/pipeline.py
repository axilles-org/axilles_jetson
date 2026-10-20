"""End-to-end: mocap file + IMU file + subject -> IMU placement estimate."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .estimate import ImuData, locate_imu
from .io import load_coordinates, load_imu_csv
from .kinematics import Walker
from .model import SEGMENTS, ImuSite, build_mjcf
from .subject import Subject


def segment_kinematics(subject: Subject, t, coords, fc=6.0, verbose=True):
    walker = Walker(build_mjcf(subject))
    missing = [j for j in walker.joint_names if j not in coords]
    if missing and verbose:
        print(f"[pipeline] coordinates not in mocap (held at 0): {', '.join(missing)}")
    if any(j in missing for j in ("pelvis_tx", "pelvis_ty", "pelvis_tz")) and verbose:
        print("[pipeline] WARNING: pelvis translation missing -> accelerations (and IMU "
              "positions) will be wrong; only orientations are trustworthy")
    q = walker.qpos_from_coords(coords, len(t))
    segs, _ = walker.playback(t, q, fc=fc)
    return segs


def estimate_to_site(name, est, rgba="0.1 0.9 0.2 1") -> ImuSite:
    from scipy.spatial.transform import Rotation as Rot
    return ImuSite(name, est.segment, tuple(est.r),
                   tuple(Rot.from_matrix(est.R).as_quat(scalar_first=True)), rgba)


def run(subject: Subject, mocap_path, imu_paths: dict, fc=6.0, segment=None, max_lag=0.3,
        prior=None, out_dir=None, imu_kwargs=None, coord_kwargs=None, verbose=True):
    t, coords = load_coordinates(mocap_path, verbose=verbose, **(coord_kwargs or {}))
    segs = segment_kinematics(subject, t, coords, fc=fc, verbose=verbose)
    results, sites = {}, []
    for name, path in imu_paths.items():
        imu = path if isinstance(path, ImuData) else load_imu_csv(path, verbose=verbose,
                                                                  **(imu_kwargs or {}))
        seg = segment.get(name) if isinstance(segment, dict) else segment
        pr = prior.get(name) if isinstance(prior, dict) else prior
        est = locate_imu(segs, imu, fc=fc, segment=seg, max_lag=max_lag, prior=pr)
        results[name] = est
        sites.append(estimate_to_site(name, est))
        if verbose:
            print_estimate(name, est)
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "imu_estimates.json").write_text(
            json.dumps({k: v.to_dict() for k, v in results.items()}, indent=2))
        (out / "walker_with_imus.xml").write_text(build_mjcf(subject, sites))
        if verbose:
            print(f"[pipeline] wrote {out / 'imu_estimates.json'} and walker_with_imus.xml")
    return results, segs


def print_estimate(name, est):
    d = est.to_dict()
    cm = lambda v: "(" + ", ".join(f"{x * 100:6.2f}" for x in v) + ") cm"
    print(f"\n=== {name} -> {est.segment}   (time offset {est.time_offset:+.4f} s)")
    print(f"  position (segment frame, x fwd / y left / z up): {cm(est.r)}")
    if est.r_std is not None:
        print(f"    +/- (model)  {cm(est.r_std)}")
    if est.r_chunk_std is not None:
        print(f"    +/- (chunks) {cm(est.r_chunk_std)}")
    ax = d["sensor_axes_in_segment_mujoco"]
    print("  sensor axes in segment frame: " +
          "  ".join(f"{k}={np.round(v, 3).tolist()}" for k, v in ax.items()))
    print(f"  residual rms: gyro {est.rms_gyro:.4f} rad/s, acc {est.rms_acc:.3f} m/s^2")
    top = ", ".join(f"{c['segment']}:{c['rel_gyro_residual']:.3f}" for c in est.candidates[:4])
    print(f"  segment ranking (rel. gyro residual): {top}")
    for w in est.warnings:
        print(f"  ! {w}")
