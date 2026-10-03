"""Locate the Camargo-dataset IMUs (foot, shank, thigh, trunk) for one subject.

For every trial: IK -> MuJoCo model converted from the subject's .osim ->
segment kinematics; then estimate segment, clock offset, orientation, position
for each IMU.  Results are aggregated across trials (median + spread).

  python scripts/locate_camargo.py --subject-dir AB09 --modes treadmill levelground ramp stair
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from imu_locator import ImuSite  # noqa: E402
from imu_locator.camargo import IMU_NAMES, load_ik, load_imu, trial_files  # noqa: E402
from imu_locator.estimate import locate_imu  # noqa: E402
from imu_locator.kinematics import Walker  # noqa: E402
from imu_locator.model import C_MJ2OS  # noqa: E402
from imu_locator.osim import osim_to_mjcf  # noqa: E402
from imu_locator.plots import plot_fit  # noqa: E402

# Camargo IMUs (per axis): +-8 g accelerometer, +-16 rad/s gyro; 3% margin
SAT_ACC = 0.97 * 8 * 9.80665
SAT_GYR = 0.97 * 16.0
MAX_REL_ACC = 0.65   # residual RMS / signal RMS above this = recording problem, not mounting
# A strapped IMU never matches IK this well (soft tissue, 1-DOF knee): normal trials are
# 0.14-0.40. Far below that the IMU data are likely not a raw recording (e.g. reconstructed).
MIN_REL_GYRO = 0.12
MAX_ROT_OUTLIER_DEG = 45.0
EXPECTED = dict(foot="calcn_r", shank="tibia_r", thigh="femur_r", trunk="torso")
CANDIDATES = ["pelvis", "torso", "femur_r", "tibia_r", "talus_r", "calcn_r",
              "femur_l", "tibia_l", "talus_l", "calcn_l"]


def rot_mean(Rs):
    return Rot.from_matrix(np.array(Rs)).mean().as_matrix()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject-dir", default="AB09")
    ap.add_argument("--modes", nargs="+", default=["treadmill", "levelground", "ramp", "stair"])
    ap.add_argument("--max-trials", type=int, default=6, help="per mode")
    ap.add_argument("--fc", type=float, default=6.0)
    ap.add_argument("--skip-start", type=float, default=1.5, help="s dropped at trial start (IMU start-up artefact)")
    ap.add_argument("--fix-rot", action="store_true",
                    help="keep the gyro+gravity orientation fixed in the final refinement")
    ap.add_argument("--out", default="results/AB09")
    ap.add_argument("--reaggregate", action="store_true",
                    help="reuse <out>/per_trial.json (no refitting) and only redo the summary")
    a = ap.parse_args()

    sd = Path(a.subject_dir)
    date_dir = next(p for p in sd.iterdir() if p.is_dir() and p.name[0].isdigit())
    osim_path = next((sd / "osimxml").glob("*.osim"))   # e.g. AB07 ships ab07.osim
    osim = osim_to_mjcf(osim_path)
    walker = Walker(osim.xml)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    per_trial = {n: [] for n in IMU_NAMES}
    first = {}
    if a.reaggregate:
        per_trial = json.loads((out / "per_trial.json").read_text())
    for mode in ([] if a.reaggregate else a.modes):
        trials = trial_files(date_dir, mode)
        step = max(1, len(trials) // a.max_trials)
        for trial in trials[::step][: a.max_trials]:
            try:
                t, coords = load_ik(date_dir, mode, trial)
                imus = load_imu(date_dir, mode, trial)
            except FileNotFoundError as e:
                print(f"skip {trial}: {e}")
                continue
            keep = t >= t[0] + a.skip_start
            t = t[keep]
            coords = {k: v[keep] for k, v in coords.items()}
            q = walker.qpos_from_coords(osim.coords_with_dependents(coords), len(t))
            segs, _ = walker.playback(t, q, fc=a.fc, bodies=CANDIDATES)
            for name, imu in imus.items():
                k = imu.t >= imu.t[0] + a.skip_start
                imu.t, imu.acc, imu.gyr = imu.t[k], imu.acc[k], imu.gyr[k]
                try:
                    # segment identification uses the gyro only, so the accel sign does not matter here
                    auto = locate_imu(segs, imu, fc=a.fc, max_lag=0.1, n_chunks=0, acc_sign=1)
                    fits = {sg: locate_imu(segs, imu, fc=a.fc, segment=EXPECTED[name], max_lag=0.1,
                                           n_chunks=0, acc_sign=sg, refine_orientation=not a.fix_rot,
                                           sat_acc=SAT_ACC, sat_gyr=SAT_GYR)
                            for sg in (1, -1)}
                except ValueError as e:
                    print(f"  {trial} {name}: {e}")
                    continue
                row = dict(trial=trial, mode=mode, auto_segment=auto.segment,
                           ranking=auto.candidates[:3], fits={})
                for sg, est in fits.items():
                    row["fits"][str(sg)] = dict(
                        offset=est.time_offset, r=est.r.tolist(), R=est.R.tolist(),
                        rms_acc=est.rms_acc, rms_gyro=est.rms_gyro,
                        rel_gyro=est.candidates[0]["rel_gyro_residual"], rel_acc=est.rel_acc,
                        sat_frac=est.sat_frac, warnings=est.warnings)
                per_trial[name].append(row)
                if name not in first:
                    first[name] = (segs[EXPECTED[name]], imu, fits, trial)
                print(f"{trial:34s} {name:6s} auto={auto.segment:8s} acc rms  +1: {fits[1].rms_acc:5.2f}"
                      f"  -1: {fits[-1].rms_acc:5.2f}  saturated {100 * fits[1].sat_frac:4.1f}%")

    # The accel sign is a hardware property: decide it once per IMU for this subject
    subject_sign = {}
    for name, rows in per_trial.items():
        if rows:
            usable = [r for r in rows if r["fits"]["1"]["rel_gyro"] >= MIN_REL_GYRO] or rows
            med = {sg: np.median([r["fits"][sg]["rms_acc"] for r in usable]) for sg in ("1", "-1")}
            subject_sign[name] = "1" if med["1"] <= med["-1"] else "-1"
            print(f"[sign] {name}: median acc rms +1 {med['1']:.2f} / -1 {med['-1']:.2f} -> {subject_sign[name]}")
    for name, rows in per_trial.items():
        for r in rows:
            r.update(r["fits"][subject_sign[name]], acc_sign=int(subject_sign[name]))
    for name, (seg, imu, fits, trial) in first.items():
        plot_fit(seg, imu, fits[int(subject_sign[name])], a.fc, out / f"fit_{name}_{trial}.png",
                 title=f"{name} ({trial})")

    summary, sites = {}, []
    colors = dict(foot="1 0.3 0.1 1", shank="0.1 0.8 0.2 1", thigh="0.1 0.5 1 1", trunk="0.9 0.8 0.1 1")
    for name, rows in per_trial.items():
        if not rows:
            continue
        # quality gate: the accel model must explain the signal (rel_acc = residual RMS /
        # signal RMS), and the gyro fit must not be much worse than typical for this IMU
        normal = [r for r in rows if r["rel_gyro"] >= MIN_REL_GYRO] or rows
        med = np.median([r["rel_gyro"] for r in normal])
        reasons = {}
        for r in rows:
            if r["rel_gyro"] < MIN_REL_GYRO:
                reasons[r["trial"]] = f"gyro matches mocap implausibly well ({r['rel_gyro']:.3f}): not a raw recording?"
            elif r["rel_gyro"] >= 1.5 * med:
                reasons[r["trial"]] = f"poor gyro fit ({r['rel_gyro']:.2f})"
            elif r["rel_acc"] > MAX_REL_ACC:
                reasons[r["trial"]] = f"accelerometer not explained (rel {r['rel_acc']:.2f}): saturated/damped/corrupt"
        good = [r for r in rows if r["trial"] not in reasons]
        # orientation outliers: distance to the medoid rotation (robust to a few flipped fits)
        if len(good) >= 3:
            Rg = Rot.from_matrix(np.array([r["R"] for r in good]))
            D = np.array([[np.degrees((Rg[i].inv() * Rg[j]).magnitude()) for j in range(len(good))]
                          for i in range(len(good))])
            medoid = int(np.argmin(np.median(D, axis=1)))
            for r, d in zip(good, D[medoid]):
                if d > MAX_ROT_OUTLIER_DEG:
                    reasons[r["trial"]] = f"orientation {d:.0f} deg from the other trials"
            good = [r for r in good if r["trial"] not in reasons]
        if len(good) < 3:
            print(f"[quality] {name}: only {len(good)} trials pass; result unreliable")
            good = good or rows
        rr = np.array([r["r"] for r in good])
        Rm = rot_mean([r["R"] for r in good])
        ang = [np.degrees((Rot.from_matrix(Rm).inv() * Rot.from_matrix(r["R"])).magnitude()) for r in good]
        r_med = np.median(rr, axis=0)
        seg = EXPECTED[name]
        autos = [r["auto_segment"] for r in rows]
        summary[name] = dict(
            segment=seg,
            auto_segment_votes={s: autos.count(s) for s in sorted(set(autos))},
            acc_sign=int(subject_sign[name]),
            n_trials_total=len(rows),
            rejected_trials=[r["trial"] for r in rows if r not in good],
            rejection_reasons=reasons,
            saturated_frame_pct_median=round(100 * float(np.median([r["sat_frac"] for r in rows])), 2),
            n_trials=len(good),
            position_m_mujoco_median=np.round(r_med, 4).tolist(),
            position_m_opensim_median=np.round(C_MJ2OS @ r_med, 4).tolist(),
            position_spread_m_iqr=np.round(np.subtract(*np.percentile(rr, [75, 25], axis=0)), 4).tolist(),
            position_spread_m_std=np.round(rr.std(0, ddof=1), 4).tolist() if len(rr) > 1 else None,
            R_sensor_to_segment_mujoco=np.round(Rm, 4).tolist(),
            R_sensor_to_segment_opensim=np.round(C_MJ2OS @ Rm, 4).tolist(),
            quat_wxyz_sensor_to_segment_mujoco=np.round(Rot.from_matrix(Rm).as_quat(scalar_first=True), 5).tolist(),
            sensor_axes_in_segment_mujoco={ax: np.round(Rm[:, i], 3).tolist() for i, ax in enumerate("xyz")},
            rotation_spread_deg_rms=round(float(np.sqrt(np.mean(np.square(ang)))), 2),
            time_offset_ms_median=round(1000 * float(np.median([r["offset"] for r in good])), 2),
            rms_acc_median=round(float(np.median([r["rms_acc"] for r in good])), 3),
            rel_gyro_residual_median=round(float(np.median([r["rel_gyro"] for r in good])), 3),
        )
        sites.append(ImuSite(f"imu_{name}", seg, tuple(r_med),
                             tuple(Rot.from_matrix(Rm).as_quat(scalar_first=True)), colors[name]))
    (out / "per_trial.json").write_text(json.dumps(per_trial, indent=1))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / f"{sd.name}_with_imus.xml").write_text(
        osim_to_mjcf(osim_path, imus=sites).xml)

    print("\n================= SUMMARY (median over trials) =================")
    for name, s in summary.items():
        p, sp = np.array(s["position_m_mujoco_median"]) * 100, np.array(s["position_spread_m_iqr"]) * 100
        print(f"\n{name} on {s['segment']}  (auto-identified: {s['auto_segment_votes']}, n={s['n_trials']})")
        print(f"  position [cm] x fwd / y left / z up : ({p[0]:6.1f}, {p[1]:6.1f}, {p[2]:6.1f})"
              f"   IQR ({sp[0]:.1f}, {sp[1]:.1f}, {sp[2]:.1f})")
        print(f"  sensor axes in segment frame: {s['sensor_axes_in_segment_mujoco']}")
        print(f"  orientation spread {s['rotation_spread_deg_rms']} deg, clock offset {s['time_offset_ms_median']} ms, "
              f"rel gyro residual {s['rel_gyro_residual_median']}, acc rms {s['rms_acc_median']} m/s2")
    print(f"\nwrote {out}/summary.json, per_trial.json, {sd.name}_with_imus.xml")


if __name__ == "__main__":
    main()
