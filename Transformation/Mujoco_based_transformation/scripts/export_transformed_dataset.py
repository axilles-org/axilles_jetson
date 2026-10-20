"""Export the transformed Georgia Tech dataset and the exo recordings.

Georgia Tech: every trial of every activity, foot + shank IMUs, in 4 variants:
    <out>/georgia_tech_transformed/<frame>/<version>/<subject>/<mode>/<trial>.parquet
      frame   = exo_frame | imu_raw_frame
      version = v1_average | v2_own
    columns: time_s, foot_ax..foot_gz, shank_ax..shank_gz  (m/s^2, rad/s; transformed),
             heel_strike_pct, toe_off_pct (Georgia Tech gait-cycle labels, if present),
             foot_saturated, shank_saturated (raw sample at/near the +-8 g / +-16 rad/s range)
    plus <out>/georgia_tech_transformed/index.csv: one row per trial with duration and
    the placement-analysis quality verdict for that trial (ok / rejected: reason / not checked).

Exo: <out>/exo/{exo_frame,imu_raw_frame}/<recording>.csv — the walking recordings,
held IMU samples removed and re-interpolated to 200 Hz (no filtering), rotated with
the exo IMU calibration; raw copies of the recordings are kept next to them.

  python scripts/export_transformed_dataset.py --out axilles_imu_transform/5_data
"""
import argparse
import json
import re
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from imu_locator.camargo import _table, load_imu, trial_files  # noqa: E402
from imu_locator.exo_transform import Transform  # noqa: E402

SEGS = ("foot", "shank")
MODES = ("treadmill", "levelground", "ramp", "stair")
G = 9.80665
SAT_ACC, SAT_GYR = 0.97 * 8 * G, 0.97 * 16.0
FS = 200.0
COLS = [f"{s}_{k}{c}" for s in SEGS for k in ("a", "g") for c in "xyz"]
FRAMES = {"exo_frame": "results/exo_transform", "imu_raw_frame": "results/exo_transform_raw_frame"}


def load_transforms(subject, frames):
    out = {}
    for frame, d in frames.items():
        v1 = json.loads((Path(d) / "transform_version1_average.json").read_text())
        v2 = json.loads((Path(d) / "transform_version2_per_subject.json").read_text())[subject]
        for ver, src in (("v1_average", v1), ("v2_own", v2)):
            out[(frame, ver)] = {s: Transform(R=np.array(src[s]["R_gt_sensor_to_exo"]), r_gt=np.zeros(3),
                                              r_exo=np.zeros(3), acc_sign=int(src[s]["acc_sign"]),
                                              d_override=np.array(src[s]["lever_arm_d_m"])) for s in SEGS}
    return out


def export_subject(args):
    subject, camargo, placement, out, frames = args
    sd = Path(camargo) / subject
    dd = next(p for p in sd.iterdir() if p.is_dir() and p.name[0].isdigit())
    T = load_transforms(subject, frames)
    summ = json.loads((Path(placement) / subject / "summary.json").read_text())
    checked = {r["trial"] for r in json.loads((Path(placement) / subject / "per_trial.json").read_text())["foot"]}
    reasons = {s: summ[s].get("rejection_reasons", {}) for s in SEGS}
    index = []
    for mode in MODES:
        for trial in trial_files(dd, mode):
            try:
                imu = load_imu(dd, mode, trial)
            except Exception as e:  # noqa: BLE001
                print(f"[{subject}] skip {mode}/{trial}: {e}", flush=True)
                continue
            t = imu["foot"].t
            base = {"time_s": t.astype(np.float64)}
            gcf = dd / mode / "gcRight" / f"{trial}.mat"
            if gcf.exists():
                gc = _table(gcf)
                for col, name in (("HeelStrike", "heel_strike_pct"), ("ToeOff", "toe_off_pct")):
                    base[name] = np.interp(t, gc["Header"].to_numpy(float), gc[col].to_numpy(float)).astype(np.float32)
            for s in SEGS:
                base[f"{s}_saturated"] = ((np.abs(imu[s].acc) >= SAT_ACC).any(1) |
                                          (np.abs(imu[s].gyr) >= SAT_GYR).any(1))
            for (frame, ver), tr in T.items():
                cols = dict(base)
                for s in SEGS:
                    a, g = tr[s].apply(imu[s].acc, imu[s].gyr, FS, acc_in_g=False)
                    for i, c in enumerate("xyz"):
                        cols[f"{s}_a{c}"] = a[:, i].astype(np.float32)
                        cols[f"{s}_g{c}"] = g[:, i].astype(np.float32)
                df = pd.DataFrame(cols)[["time_s"] + COLS + [c for c in base if c != "time_s"]]
                p = Path(out) / "georgia_tech_transformed" / frame / ver / subject / mode / f"{trial}.parquet"
                p.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(p, compression="zstd", index=False)

            def verdict(s):
                if trial not in checked:
                    return "not checked"
                return f"rejected: {reasons[s][trial]}" if trial in reasons[s] else "ok"
            index.append(dict(subject=subject, mode=mode, trial=trial, n_samples=len(t),
                              duration_s=round(float(t[-1] - t[0]), 2),
                              foot_quality=verdict("foot"), shank_quality=verdict("shank"),
                              foot_saturated_pct=round(100 * float(base["foot_saturated"].mean()), 2),
                              shank_saturated_pct=round(100 * float(base["shank_saturated"].mean()), 2)))
    print(f"[{subject}] {len(index)} trials", flush=True)
    return index


def export_exo(out, exo_dir, calib_dir, frames=FRAMES):
    """Exo walking recordings: raw copies + exo-frame and raw-axes versions."""
    cal = json.loads((Path(frames["exo_frame"]) / "exo_imu_calibration.json").read_text())
    C = {ses: {s: np.array(m) for s, m in v.items()} for ses, v in cal["R_exo_frame_from_exo_sensor"].items()}
    sept, april = "Sept 11 (2×60 s)", "Apr 4, 2 km/h (5×10 s)"
    exo = Path(out) / "exo"
    (exo / "raw_recordings" / "calibration").mkdir(parents=True, exist_ok=True)
    for f in sorted(Path(exo_dir).glob("data_collection_*.csv")):
        shutil.copy2(f, exo / "raw_recordings" / f.name)
    for f in sorted(Path(calib_dir).glob("*.npz")):
        shutil.copy2(f, exo / "raw_recordings" / "calibration" / f.name)

    def resample(t, x):
        """Drop held IMU samples, linear-interpolate to 200 Hz (no filtering)."""
        ok = np.isfinite(x).all(1)
        t_, x_ = t[ok], x[ok]
        new = np.r_[True, np.abs(np.diff(x_, axis=0)).sum(1) > 0]
        t_, x_ = t_[new], x_[new]
        return np.column_stack([np.interp(tu, t_, x_[:, j]) for j in range(x_.shape[1])])

    recs = []
    raw = np.load(Path(calib_dir) / "raw_20260911_175501.npz")
    w2 = np.load(Path(calib_dir) / "walk_20260911_183138.npz")
    for name, src, p in (("calibration_20260911_175501_walking", raw, "walking__"),
                         ("walk_20260911_183138", w2, "")):
        t = src[p + "time"]
        recs.append((name, sept, t, {s: (src[p + s + "_accel"], src[p + s + "_gyro"]) for s in SEGS},
                     {"ankle_encoder_deg": src[p + "encoder"], "toe_fsr_raw": src[p + "toe"],
                      "heel_fsr_raw": src[p + "heel"]}))
    for f in sorted(Path(exo_dir).glob("data_collection_20260404_*.csv")):
        df = pd.read_csv(f)
        t = df["timestamp_s"].to_numpy()
        recs.append((f.stem, april, t, {s: (df[[f"{s}_a{c}" for c in "xyz"]].to_numpy(),
                                            df[[f"{s}_g{c}" for c in "xyz"]].to_numpy()) for s in SEGS},
                     {c: df[c].to_numpy(float) for c in ("ankle_encoder_deg", "toe_fsr_raw", "heel_fsr_raw")}))
    for name, ses, t, imu, aux in recs:
        global tu
        tu = np.arange(np.nanmin(t), np.nanmax(t), 1 / FS)
        for frame in ("exo_frame", "imu_raw_frame"):
            cols = {"time_s": tu}
            for s in SEGS:
                M = C[ses][s] if frame == "exo_frame" else C[sept][s].T @ C[ses][s]
                for k, x in zip("ag", imu[s]):
                    v = resample(t, x) @ M.T
                    for i, c in enumerate("xyz"):
                        cols[f"{s}_{k}{c}"] = np.round(v[:, i], 5)
            for c, x in aux.items():
                ok = np.isfinite(x)
                cols[c] = np.round(np.interp(tu, t[ok], x[ok]), 3) if ok.any() else np.nan
            p = exo / frame / f"{name}.csv"
            p.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(cols).to_csv(p, index=False)
    print(f"[exo] {len(recs)} walking recordings exported", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="axilles_imu_transform/5_data")
    ap.add_argument("--camargo-root", default=".")
    ap.add_argument("--placement-dir", default="results")
    ap.add_argument("--exo-dir", default="data/exo")
    ap.add_argument("--calib-dir", default="data/exo/calibration")
    ap.add_argument("--transforms-exo", default=FRAMES["exo_frame"], help="folder with the exo-frame transform files")
    ap.add_argument("--transforms-raw", default=FRAMES["imu_raw_frame"], help="folder with the raw-frame transform files")
    ap.add_argument("--subjects", nargs="*")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--exo-only", action="store_true")
    a = ap.parse_args()
    frames = {"exo_frame": a.transforms_exo, "imu_raw_frame": a.transforms_raw}
    export_exo(a.out, a.exo_dir, a.calib_dir, frames)
    if a.exo_only:
        return
    subjects = a.subjects or sorted(p.parent.name for p in Path(a.placement_dir).glob("AB*/summary.json")
                                    if re.fullmatch(r"AB\d+", p.parent.name))
    with ProcessPoolExecutor(a.workers) as ex:
        rows = [r for res in ex.map(export_subject, [(s, a.camargo_root, a.placement_dir, a.out, frames) for s in subjects])
                for r in res]
    idx = Path(a.out) / "georgia_tech_transformed" / "index.csv"
    pd.DataFrame(rows).to_csv(idx, index=False)
    print(f"wrote {len(rows)} trials x 4 variants; index {idx}")


if __name__ == "__main__":
    main()
