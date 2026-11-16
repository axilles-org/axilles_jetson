"""
Build a fake mrsd-exo-ankle root: several subjects, each with THEIR OWN IMU
mount pose, slightly different gaits, plus one exo log at a different mount.
Run the multi-subject pipeline and check per-subject recovery.
"""
import subprocess, sys, json, shutil
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial.transform import Rotation as Rot
from virtual_imu import BodyKinematics, synthesize_imu, GRAVITY_Y_UP

FS = 200.0
ROOT = Path("/tmp/fake_gt"); OUT = Path("/tmp/fake_gt_tx")


def gait(n_strides, T, stride_len, phi_po, seed):
    rng = np.random.default_rng(seed)
    heel_f = np.array([-0.05, -0.07, 0.0]); toe_f = np.array([0.15, -0.07, 0.0])
    Rs, os_, hs = [], [], []; x0 = 0.0; k = 0
    for _ in range(n_strides):
        ns = int(T * (1 + 0.03 * rng.normal()) * FS); ph = np.arange(ns) / ns
        a, b, c = 0.08, 0.40, 0.62; hsd = np.deg2rad(-10); po = np.deg2rad(phi_po)
        phi = np.where(ph < a, hsd * (1 - ph / a), 0.0)
        m = (ph >= b) & (ph < c); phi[m] = po * ((ph[m] - b) / (c - b)) ** 1.7
        m = ph >= c; u = (ph[m] - c) / (1 - c); phi[m] = po * (1 - u) ** 2 + hsd * u ** 2
        R = Rot.from_euler("z", phi[:, None]).as_matrix(); o = np.zeros((ns, 3))
        H = np.array([x0, 0, 0])
        Rc = Rot.from_euler("z", [[po]]).as_matrix()[0]
        Rh = Rot.from_euler("z", [[hsd]]).as_matrix()[0]
        for i in range(ns):
            if ph[i] < b: o[i] = H - R[i] @ heel_f
            elif ph[i] < c: o[i] = (H - heel_f + toe_f) - R[i] @ toe_f
            else:
                u = (ph[i] - c) / (1 - c); sm = 3 * u**2 - 2 * u**3
                oc = (H - heel_f + toe_f) - Rc @ toe_f
                on = np.array([x0 + stride_len, 0, 0]) - Rh @ heel_f
                o[i] = (1 - sm) * oc + sm * on; o[i, 1] += 0.055 * np.sin(np.pi * u)
        Rs.append(R); os_.append(o); hs.append(k); k += ns; x0 += stride_len
    R = np.concatenate(Rs); o = np.concatenate(os_)
    return BodyKinematics(np.arange(len(R)) / FS, R, o, GRAVITY_Y_UP, 0.05), np.array(hs), k


def imu(bk, p, R, seed):
    rng = np.random.default_rng(seed)
    a, g = synthesize_imu(bk, p, R)
    return a + rng.normal(0, .15, a.shape), g + rng.normal(0, .015, g.shape)


if __name__ == "__main__":
    shutil.rmtree(ROOT, ignore_errors=True); shutil.rmtree(OUT, ignore_errors=True)
    rng = np.random.default_rng(0)
    p_nom = {"foot": np.array([0.02, 0.03, 0.01]), "shank": np.array([0.05, 0.02, 0.0])}
    R_nom = {"foot": Rot.from_euler("xyz", [0, -150, 5], degrees=True).as_matrix(),
             "shank": Rot.from_euler("xyz", [90, 0, 10], degrees=True).as_matrix()}
    truth, meta = {}, []
    subjects = [f"AB{i:02d}" for i in range(6, 14)]
    for si, subj in enumerate(subjects):
        # each subject: own mount (placement scatter) and own gait
        mounts = {s: (p_nom[s] + rng.normal(0, .008, 3),
                      R_nom[s] @ Rot.from_rotvec(rng.normal(0, np.deg2rad(4), 3)).as_matrix())
                  for s in p_nom}
        truth[subj] = mounts
        d = ROOT / "subjects" / subj; d.mkdir(parents=True)
        for ti in range(2):
            bk, hs, n = gait(30, 1.05 + 0.04 * rng.normal(), 0.62, 55 + 3 * rng.normal(),
                             seed=100 * si + ti)
            cols = {"time_s": np.arange(n) / FS}
            for s, (p, R) in mounts.items():
                a, g = imu(bk, p, R, seed=1000 * si + ti)
                for i, ax in enumerate("XYZ"):
                    cols[f"{s}_Accel_{ax}"] = a[:, i]; cols[f"{s}_Gyro_{ax}"] = g[:, i]
            for i, ax in enumerate("XYZ"):          # trunk, for unit detection
                cols[f"trunk_Accel_{ax}"] = (-9.81 if ax == "Y" else 0.0) + 0.3 * np.random.randn(n)
                cols[f"trunk_Gyro_{ax}"] = 0.05 * np.random.randn(n)
            tr = f"treadmill_{ti+1:02d}_01"
            pd.DataFrame(cols).to_parquet(d / f"{tr}__imu.parquet")
            pct = np.zeros(n)
            for a_, b_ in zip(hs, list(hs[1:]) + [n]):
                pct[a_:b_] = np.linspace(0, 100, b_ - a_, endpoint=False)
            pd.DataFrame({"time_s": np.arange(n) / FS, "HeelStrike": pct}).to_parquet(
                d / f"{tr}__gcRight.parquet")
            pd.DataFrame({"time_s": np.arange(n) / FS, "ankle_angle_r_moment": np.zeros(n)}
                         ).to_parquet(d / f"{tr}__id.parquet")
            meta.append(dict(subject=subj, trial=tr, speed_mean_mps=1.0, weight_kg=70.0))
    pd.DataFrame(meta).to_parquet(ROOT / "metadata.parquet")

    # exo: different person-ish gait, different mounts, binary FSRs
    exo_m = {"foot": (np.array([0.07, 0.08, -0.015]),
                      Rot.from_euler("xyz", [-8, 40, 95], degrees=True).as_matrix()),
             "shank": (np.array([0.02, 0.10, 0.03]),
                       Rot.from_euler("xyz", [0, 90, -20], degrees=True).as_matrix())}
    bk, hs, n = gait(60, 1.08, 0.60, 53, seed=999)
    d = {"timestamp_s": np.arange(n) / FS}
    for s, (p, R) in exo_m.items():
        a, g = imu(bk, p, R, seed=4242)
        for i, ax in enumerate("xyz"):
            d[f"{s}_a{ax}"] = a[:, i]; d[f"{s}_g{ax}"] = g[:, i]
    ph = np.zeros(n)
    for a_, b_ in zip(hs, list(hs[1:]) + [n]):
        ph[a_:b_] = np.linspace(0, 1, b_ - a_, endpoint=False)
    d["heel_fsr_raw"] = (ph < 0.42).astype(float)
    d["toe_fsr_raw"] = ((ph > 0.10) & (ph < 0.62)).astype(float)
    d["ankle_encoder_deg"] = 80 + 5 * np.sin(2 * np.pi * ph)
    pd.DataFrame(d).to_csv("/tmp/fake_exo_multi.csv", index=False)

    r = subprocess.run([sys.executable, "run_alignment.py", "--gatech-root", str(ROOT),
                        "--exo-log", "/tmp/fake_exo_multi.csv", "--out-root", str(OUT),
                        "--report", "/tmp/tx.json"], capture_output=True, text=True)
    print(r.stdout[-4500:]); print(r.stderr[-2000:])

    rep = json.loads(Path("/tmp/tx.json").read_text())
    print("\nPER-SUBJECT RECOVERY vs TRUTH")
    for s in ("foot", "shank"):
        errs_R, errs_p = [], []
        for subj, v in rep[s]["per_subject"].items():
            pA, RA = truth[subj][s]; pB, RB = exo_m[s]
            dR_t = RB.T @ RA; dp_t = RA.T @ (pB - pA)
            dR = np.array(v["dR"]); dp = np.array(v["dp"])
            errs_R.append(np.rad2deg(np.linalg.norm(Rot.from_matrix(dR.T @ dR_t).as_rotvec())))
            ax = np.array([0, 0, 1.0]); ax = RA.T @ ax          # rotation axis in A frame
            e = dp - dp_t; errs_p.append(np.linalg.norm(e - (e @ ax) * ax) * 1000)
        print(f"  {s:<6} dR err: median {np.median(errs_R):.2f} deg, max {np.max(errs_R):.2f}"
              f"   |  dp_perp err: median {np.median(errs_p):.1f} mm, max {np.max(errs_p):.1f}")
    f = next((OUT / "subjects" / "AB06").glob("*__imu.parquet"))
    print(f"\noutput sample: {f.name}, cols={list(pd.read_parquet(f).columns)[:5]}...")
    print("linked streams:", sorted(p.name for p in (OUT/'subjects'/'AB06').iterdir())[:4])