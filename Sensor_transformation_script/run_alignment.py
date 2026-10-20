"""
run_alignment.py  (multi-subject)
================================
Fit one IMU transform PER GaTech SUBJECT against ONE exo recording, report how
well the fits agree, and write a transformed copy of the dataset ready for
training.

    python3 run_alignment.py \
        --gatech-root mrsd-exo-ankle \
        --exo-log calib_poweroff_1p0mps.csv \
        --speed 0.85 1.10 \
        --out-root mrsd-exo-ankle-tx

Pipeline, per segment (foot, shank), per GaTech subject:
  1. stride template of the subject (their speed-matched trials, gcRight HS)
  2. stride template of your exo (FSR heel strikes)
  3. init  : Kabsch on gyro templates (closed form) + phase-offset search
  4. refine: Kang-style joint 6-DOF least squares on accel + gyro templates
  5. every trial of that subject transformed with ITS OWN dR, dp

Record the exo trial for this with: assistance OFF, walking speed inside the
--speed band, cadence metronome-matched to the GaTech subjects at that speed,
3+ minutes. That is what keeps the gait-difference confound small.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import align_core as ac
import align_io as aio

SEGMENTS = ("foot", "shank")


def exo_templates(exo, segs):
    hs = exo.events.get("heel_strike", np.array([], int))
    if len(hs) < 5:
        raise SystemExit(
            f"only {len(hs)} heel strikes detected on the exo log. Check the "
            f"FSR columns with: python3 sniff.py <exo_log>. The template needs many strides.")
    out = {}
    for s in segs:
        if s in exo.accel:
            out[s] = ac.Template(ac.stride_ensemble(
                exo.accel[s], exo.gyro[s], exo.fs, hs))
    return out, hs


def subject_templates(root, subject, trials, segs):
    ens = defaultdict(list)
    used = []
    for tr in trials:
        try:
            rec = aio.load_gatech(root, subject, tr, segments=segs)
        except Exception as e:                                # noqa
            print(f"      {tr}: load failed ({e})")
            continue
        hs = rec.events.get("heel_strike", np.array([], int))
        if len(hs) < 5:
            continue
        for s in segs:
            e = ac.stride_ensemble(rec.accel[s], rec.gyro[s], rec.fs, hs)
            if len(e):
                ens[s].append(e)
        used.append(tr)
    tpl = {}
    for s in segs:
        if ens[s]:
            try:
                tpl[s] = ac.Template.pooled(ens[s])
            except ValueError:
                pass
    return tpl, used


def write_transformed(root, out_root, subject, all_trials, fits, segs):
    """Write <out_root>/subjects/<subj>/<trial>__imu.parquet in the exo frame."""
    root, out_root = Path(root), Path(out_root)
    dst = out_root / "subjects" / subject
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for tr in all_trials:
        src_imu = root / "subjects" / subject / f"{tr}__imu.parquet"
        if not src_imu.exists():
            continue
        rec = aio.load_gatech(root, subject, tr, segments=segs, debias=False)
        t = pd.read_parquet(src_imu, columns=["time_s"])["time_s"].to_numpy()
        cols = {"time_s": t}
        for s in segs:
            if s not in fits:
                continue
            a, g = rec.accel[s], rec.gyro[s]
            if fits[s].get("mirror_P") is not None:
                P = fits[s]["mirror_P"]
                a, g = a @ P.T, -(g @ P.T)
            a2, g2 = ac.transform_imu(a, g, rec.fs, fits[s]["dR"], fits[s]["dp"])
            for i, ax in enumerate("XYZ"):
                cols[f"{s}_Accel_{ax}"] = a2[:, i]
                cols[f"{s}_Gyro_{ax}"] = g2[:, i]
        pd.DataFrame(cols).to_parquet(dst / f"{tr}__imu.parquet", index=False)
        # every other stream is unchanged: link it (copy if links unsupported)
        for f in (root / "subjects" / subject).glob(f"{tr}__*.parquet"):
            if f.name.endswith("__imu.parquet"):
                continue
            tgt = dst / f.name
            if tgt.exists():
                continue
            try:
                os.symlink(f.resolve(), tgt)
            except OSError:
                shutil.copy2(f, tgt)
        n += 1
    return n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gatech-root", required=True)
    p.add_argument("--exo-log", required=True)
    p.add_argument("--speed", nargs=2, type=float, default=[0.85, 1.10],
                   help="speed band (m/s) of GaTech trials used for FITTING. "
                        "Transformed output covers all trials.")
    p.add_argument("--segments", nargs="+", default=list(SEGMENTS))
    p.add_argument("--subjects", nargs="*", default=None,
                   help="restrict to these subjects (default: all)")
    p.add_argument("--fs", type=float, default=None, help="exo log rate override")
    p.add_argument("--mirror", action="store_true",
                   help="exo is on the LEFT leg (GaTech IMUs are right-side)")
    p.add_argument("--max-shift", type=float, default=10,
                   help="phase-offset search range, percent of gait cycle")
    p.add_argument("--drop-outliers", action="store_true",
                   help="exclude flagged subjects from the transformed output")
    p.add_argument("--out-root", default=None,
                   help="write transformed dataset here (omit to only report)")
    p.add_argument("--report", default="transforms.json")
    a = p.parse_args()
    segs = tuple(a.segments)

    # ---------------------------------------------------------------- exo --
    print("=" * 70 + "\nTARGET: your exo\n" + "=" * 70)
    exo = aio.load_exo(a.exo_log, fs=a.fs)
    odr = exo.meta["effective_odr_hz"]
    print(f"  {exo.meta['file']}: {exo.meta['n_rows']} rows, {exo.fs:.0f} Hz "
          f"logged, {odr:.0f} Hz effective")
    if odr < 0.6 * exo.fs:
        print("  [WARN] low ODR -- held samples were splined, but alpha (and so "
              "dp) will be degraded. Rotation is unaffected.")
    tgt, hs = exo_templates(exo, segs)
    for s, t in tgt.items():
        print(f"  {s:<6} template: {t.n_strides} strides")
    if a.mirror:
        tgt = {s: t.mirrored() for s, t in tgt.items()}
        print("  mirrored target across its medial-lateral axis (--mirror)")

    # ------------------------------------------------------------ GaTech --
    meta = pd.read_parquet(Path(a.gatech_root) / "metadata.parquet")
    if a.subjects:
        meta = meta[meta.subject.isin(a.subjects)]
    all_trials = meta.groupby("subject")["trial"].apply(list).to_dict()
    fit_meta = meta
    if "speed_mean_mps" in meta:
        lo, hi = a.speed
        fit_meta = meta[(meta.speed_mean_mps >= lo) & (meta.speed_mean_mps <= hi)]
    fit_trials = fit_meta.groupby("subject")["trial"].apply(list).to_dict()
    print("\n" + "=" * 70 + f"\nSOURCE: {len(all_trials)} GaTech subjects, "
          f"{len(fit_meta)} trials in {a.speed[0]}-{a.speed[1]} m/s for fitting\n"
          + "=" * 70)
    if not fit_trials:
        raise SystemExit("no GaTech trials in that speed band; widen --speed")

    # -------------------------------------------------------------- fits --
    fits = {s: {} for s in segs}
    for subj in sorted(fit_trials):
        tpl, used = subject_templates(a.gatech_root, subj, fit_trials[subj], segs)
        line = [f"  {subj}  ({len(used)} trials)"]
        # ONE phase offset for all segments, from rotation-invariant |w|
        try:
            ph, ph_score = ac.phase_offset(tpl, tgt, max_shift=a.max_shift)
        except ValueError:
            ph, ph_score = 0.0, float("nan")
        tgt_s = {k: v.shifted(ph) for k, v in tgt.items()}
        line.append(f"phase {ph:+.1f}% (r={ph_score:.2f})")
        for s in segs:
            if s not in tpl or s not in tgt:
                line.append(f"{s}: --")
                continue
            f = ac.kang_refine(tpl[s], tgt_s[s], max_shift=0)
            f["shift"] = ph
            f["mirror_P"] = getattr(tgt[s], "mirror_P", None)
            fits[s][subj] = f
            b, af = f["rmse_before"], f["rmse_after"]
            line.append(f"{s}: acc {b[0]:.2f}->{af[0]:.2f}  gyr {b[1]:.2f}->"
                        f"{af[1]:.2f}"
                        + ("  [dp at bound]" if f["at_bound"] else ""))
        print("   ".join(line))

    # ------------------------------------------------------------ report --
    summary = {}
    for s in segs:
        F = fits[s]
        if len(F) < 2:
            continue
        cr = ac.cluster_report(F)
        acc_b = np.array([f["rmse_before"][0] for f in F.values()])
        acc_a = np.array([f["rmse_after"][0] for f in F.values()])
        gyr_b = np.array([f["rmse_before"][1] for f in F.values()])
        gyr_a = np.array([f["rmse_after"][1] for f in F.values()])
        print("\n" + "-" * 70 + f"\n{s.upper()}  ({len(F)} subjects)\n" + "-" * 70)
        print(f"  template RMSE, median over subjects   (Kang: acc -82.6%, gyr -40.6%)")
        print(f"    accel : {np.median(acc_b):.2f} -> {np.median(acc_a):.2f} m/s^2"
              f"   ({100*(1-np.median(acc_a)/np.median(acc_b)):+.0f}%)")
        print(f"    gyro  : {np.median(gyr_b):.3f} -> {np.median(gyr_a):.3f} rad/s"
              f"   ({100*(1-np.median(gyr_a)/np.median(gyr_b)):+.0f}%)")
        print(f"  dR agreement: median {cr['angle_median']:.1f} deg from mean, "
              f"90th pct {cr['angle_p90']:.1f} deg")
        print(f"  dp_perp spread: {cr['dp_spread_mm']:.0f} mm   "
              f"median dp_perp {np.round(cr['dp_perp_median']*1000,0)} mm")
        shifts = [f["shift"] for f in F.values()]
        print(f"  phase offset (shared): median {np.median(shifts):+.1f}%  "
              f"(range {min(shifts):+.1f} to {max(shifts):+.1f})")
        if cr["outliers"]:
            print(f"  OUTLIERS: {', '.join(cr['outliers'])}")
        if cr["angle_p90"] > 10:
            print("  [WARN] dR scatter >10 deg: fits are picking up gait, not "
                  "geometry. Check speed/cadence matching of the exo trial.")
        summary[s] = dict(
            outliers=cr["outliers"],
            R_mean=cr["R_mean"].tolist(),
            dp_perp_median=cr["dp_perp_median"].tolist(),
            angle_median_deg=cr["angle_median"],
            angle_p90_deg=cr["angle_p90"],
            dp_spread_mm=cr["dp_spread_mm"],
            per_subject={k: dict(dR=v["dR"].tolist(), dp=v["dp"].tolist(),
                                 shift=float(v["shift"]),
                                 rmse_before=v["rmse_before"],
                                 rmse_after=v["rmse_after"],
                                 angle_to_mean=cr["angle_to_mean"][k])
                         for k, v in F.items()})

    Path(a.report).write_text(json.dumps(summary, indent=1))
    print(f"\nreport -> {a.report}")

    # ------------------------------------------------------------ output --
    if a.out_root:
        out = Path(a.out_root)
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(a.gatech_root) / "metadata.parquet", out / "metadata.parquet")
        drop = set()
        if a.drop_outliers:
            for s in summary:
                drop |= set(summary[s]["outliers"])
        n_tr = 0
        for subj, trials in sorted(all_trials.items()):
            sf = {s: fits[s][subj] for s in segs if subj in fits[s]}
            if not sf or subj in drop:
                continue
            n_tr += write_transformed(a.gatech_root, out, subj, trials, sf, segs)
        print(f"transformed dataset -> {out}  ({n_tr} trials"
              + (f", dropped {sorted(drop)}" if drop else "") + ")")
        print("IMU columns are now in YOUR exo's sensor frames, m/s^2 and rad/s. "
              "Other streams are linked unchanged.")


if __name__ == "__main__":
    main()