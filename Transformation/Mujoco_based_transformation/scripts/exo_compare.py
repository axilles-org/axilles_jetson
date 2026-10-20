"""Georgia Tech -> Axilles exo frame: build the transforms and compare with exo data.

Version 1: one transform from the subject-average IMU pose and anatomy.
Version 2: each subject gets their own transform.

Both are applied to the GT foot/shank IMU data (treadmill trials that passed the
placement quality gate) and compared, stride-averaged, with the exo's own walking
recordings rotated into the exo frame by a calibration that uses only exo data.

  python scripts/exo_compare.py
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from imu_locator.camargo import _table, load_imu  # noqa: E402
from imu_locator.exo_transform import (ExoGeometry, Transform, exo_frame_rotation,  # noqa: E402
                                       exo_imu_positions, principal_axis, rotation_mean, standing_up)
from imu_locator.signal_utils import lowpass  # noqa: E402

FS = 200.0
FC = 6.0                     # same low-pass on both datasets (exo IMUs update at ~20-30 Hz)
NPTS = 101
SEGS = ("foot", "shank")
GT_BODY = dict(foot="calcn_r", shank="tibia_r")
CH = [("acc", i, f"acc {c} [m/s²]") for i, c in enumerate("xyz")] + \
     [("gyr", i, f"gyro {c} [rad/s]") for i, c in enumerate("xyz")]
# set from the command line in main(); defaults work from the project root
CALIB = Path("data/exo/calibration")
EXO_DIR = Path("data/exo")
PLACEMENT = Path("results")      # results/<subject>/{summary,per_trial}.json from locate_camargo.py
CAMARGO = Path(".")              # folder holding the raw Camargo subject folders (AB06/, ...)
INK, MUTED, GRID = "#0b0b0b", "#8a8984", "#ebeae6"
C_GT, C_SEP, C_APR = "#2a78d6", "#eb6834", "#1baf7a"
# Output frame: OUT[seg] maps exo-frame vectors to the output frame (identity = exo
# frame; C_sept^T = the exo IMU's raw sensor axes, Sept 11 mounting). Gait events
# are always detected in the exo frame.
OUT = {"foot": np.eye(3), "shank": np.eye(3)}
FRAME_NAME = "exo frame: x fwd, y left, z up"
RAW_SESSION = "Sept 11 (2×60 s)"


# ------------------------------------------------------------------ helpers
def uniform(t, x, fs=FS):
    """Drop held (repeated) IMU samples, interpolate to a uniform grid, low-pass."""
    ok = np.isfinite(x).all(1) & np.isfinite(t)
    t, x = t[ok], x[ok]
    new = np.r_[True, np.abs(np.diff(x, axis=0)).sum(1) > 0]
    t, x = t[new], x[new]
    tu = np.arange(t[0], t[-1], 1 / fs)
    xu = np.column_stack([np.interp(tu, t, x[:, j]) for j in range(3)])
    return tu, lowpass(xu, fs, FC)


def midswing(shank_wy, fs=FS, thr=-1.5, min_gap=0.6):
    """Shank mid-swing peaks (large negative rotation about +y, left): the most
    robust gait event in both datasets."""
    idx = []
    n = int(min_gap * fs)
    i = 1
    while i < len(shank_wy) - 1:
        if shank_wy[i] < thr and shank_wy[i] <= shank_wy[max(0, i - n):i + n].min():
            idx.append(i)
            i += n
        else:
            i += 1
    return np.array(idx)


def cycles(sig, ev, fs=FS, dmin=0.7, dmax=2.5):
    out, dur = [], []
    for a, b in zip(ev[:-1], ev[1:]):
        if dmin <= (b - a) / fs <= dmax:
            seg = sig[a:b]
            src = np.linspace(0, 1, len(seg))
            out.append(np.column_stack([np.interp(np.linspace(0, 1, NPTS), src, seg[:, j])
                                        for j in range(seg.shape[1])]))
            dur.append((b - a) / fs)
    return np.array(out), np.array(dur)


def rephase(c, phi):
    """Roll cycles so 0% is heel strike (phi = HS phase after mid-swing, 0-1)."""
    k = int(round(phi * (NPTS - 1)))
    return np.concatenate([c[..., k:-1, :], c[..., :k, :], c[..., k:k + 1, :]], axis=-2)


# ------------------------------------------------------------------ exo data
def load_exo():
    """Exo walking sessions, each IMU in its own sensor frame, uniform 200 Hz."""
    raw = np.load(CALIB / "raw_20260911_175501.npz")
    w2 = np.load(CALIB / "walk_20260911_183138.npz")
    sessions = {"Sept 11 (2×60 s)": [], "Apr 4, 2 km/h (5×10 s)": []}
    for src, p in [(raw, "walking__"), (w2, "")]:
        t = src[p + "time"]
        rec = {}
        for s in SEGS:
            ta, a = uniform(t, src[p + s + "_accel"])
            tg, g = uniform(t, src[p + s + "_gyro"])
            n = min(len(a), len(g))
            rec[s] = (a[:n], g[:n])
        sessions["Sept 11 (2×60 s)"].append(rec)
    for f in sorted(EXO_DIR.glob("data_collection_20260404_*.csv")):
        df = pd.read_csv(f)
        t = df["timestamp_s"].to_numpy()
        rec = {}
        for s in SEGS:
            _, a = uniform(t, df[[f"{s}_a{c}" for c in "xyz"]].to_numpy())
            _, g = uniform(t, df[[f"{s}_g{c}" for c in "xyz"]].to_numpy())
            n = min(len(a), len(g))
            rec[s] = (a[:n], g[:n])
        sessions["Apr 4, 2 km/h (5×10 s)"].append(rec)
    standing = {s: raw[f"standing__{s}_accel"] for s in SEGS}
    hipswing = {s: raw[f"hipswing__{s}_gyro"] for s in SEGS}
    return sessions, standing, hipswing


def calibrate_exo(sessions, standing, hipswing):
    """Exo frame <- exo sensor, per session, from exo data only:
    z = 'up' (standing gravity for Sept; stride-mean specific force for April,
    validated on Sept), y = walking flexion axis; signs: shank mid-swing rotation
    is negative about +y; foot y agrees with shank y during the rigid hip swing."""
    fin = lambda x: x[np.isfinite(x).all(1)]
    out, report = {}, {}
    for name, recs in sessions.items():
        R = {}
        for s in SEGS:
            g = np.vstack([r[s][1] for r in recs])
            axis, frac = principal_axis(g)
            if name.startswith("Sept"):
                up = fin(standing[s]).mean(0)
                up_src = "standing gravity"
            else:
                # no standing phase: walking-mean specific force, corrected by the
                # walking-mean -> standing rotation measured in the Sept session
                # (the bias comes from average posture during gait, ~10 deg)
                from scipy.spatial.transform import Rotation as Rot
                sept = sessions["Sept 11 (2×60 s)"]
                u_w = np.vstack([r[s][0] for r in sept]).mean(0)
                u_s = fin(standing[s]).mean(0)
                fix, _ = Rot.align_vectors([u_s / np.linalg.norm(u_s)], [u_w / np.linalg.norm(u_w)])
                up = fix.apply(np.vstack([r[s][0] for r in recs]).mean(0))
                up_src = "walking mean specific force + Sept walking->standing correction"
            R[s] = exo_frame_rotation(up, axis, 1.0)
            report.setdefault(name, {})[s] = dict(up_source=up_src, flex_axis_var=round(frac, 3))
        # sign: shank's dominant rotation (mid-swing) must be negative about +y
        wy = np.concatenate([r["shank"][1] @ R["shank"][1] for r in recs])
        if wy[np.argmax(np.abs(wy))] > 0:
            R["shank"] = exo_frame_rotation(R["shank"][2], -R["shank"][1], 1.0)
        # foot sign: during the hip swing foot and shank rotate together
        hs_f, hs_s = fin(hipswing["foot"]), fin(hipswing["shank"])
        n = min(len(hs_f), len(hs_s))
        c = np.corrcoef(hs_f[:n] @ R["foot"][1], hs_s[:n] @ R["shank"][1])[0, 1]
        if c < 0:
            R["foot"] = exo_frame_rotation(R["foot"][2], -R["foot"][1], 1.0)
        report[name]["hipswing_corr_foot_shank_y"] = round(abs(float(c)), 3)
        out[name] = R
    # validate the April 'up' method on the Sept walks
    up_walk = {s: np.vstack([r[s][0] for r in sessions["Sept 11 (2×60 s)"]]).mean(0) for s in SEGS}
    for s in SEGS:
        u1, u2 = fin(standing[s]).mean(0), up_walk[s]
        report.setdefault("check", {})[f"{s}_walk_mean_vs_standing_up_deg"] = round(float(np.degrees(
            np.arccos(u1 @ u2 / np.linalg.norm(u1) / np.linalg.norm(u2)))), 2)
    return out, report


def exo_cycles(sessions, Rexo):
    res = {}
    for name, recs in sessions.items():
        cyc = {s: {"acc": [], "gyr": []} for s in SEGS}
        durs = []
        for r in recs:
            a = {s: r[s][0] @ (OUT[s] @ Rexo[name][s]).T for s in SEGS}
            g = {s: r[s][1] @ (OUT[s] @ Rexo[name][s]).T for s in SEGS}
            ev = midswing((g["shank"] @ OUT["shank"])[:, 1])
            for s in SEGS:
                ca, d = cycles(a[s], ev)
                cg, _ = cycles(g[s], ev)
                if len(ca):
                    cyc[s]["acc"].append(ca)
                    cyc[s]["gyr"].append(cg)
            durs.append(d)
        res[name] = dict(cyc={s: {k: np.concatenate(v) for k, v in cyc[s].items()} for s in SEGS},
                         dur=np.concatenate(durs))
    return res


# ------------------------------------------------------------------ GT data
def gt_strides(subject, transforms, dur_range):
    """Transformed, stride-normalised GT cycles for one subject (treadmill trials
    that passed the placement quality gate for both foot and shank)."""
    summ = json.loads((PLACEMENT / subject / "summary.json").read_text())
    per = json.loads((PLACEMENT / subject / "per_trial.json").read_text())
    bad = set(summ["foot"]["rejected_trials"]) | set(summ["shank"]["rejected_trials"])
    trials = [r["trial"] for r in per["foot"] if r["mode"] == "treadmill" and r["trial"] not in bad]
    sd = CAMARGO / subject
    dd = next(p for p in sd.iterdir() if p.is_dir() and p.name[0].isdigit())
    out = {v: {s: {"acc": [], "gyr": []} for s in SEGS} for v in transforms}
    phis, durs = [], []
    for tr in trials:
        imu = load_imu(dd, "treadmill", tr)
        gc = _table(dd / "treadmill" / "gcRight" / f"{tr}.mat")
        k = imu["foot"].t >= imu["foot"].t[0] + 1.5
        hs = np.where(np.diff(gc["HeelStrike"].to_numpy()) < -50)[0] + 1
        hs_t = gc["Header"].to_numpy()[hs]
        for v, T in transforms.items():
            sig = {}
            for s in SEGS:
                a, g = T[s].apply(imu[s].acc[k], imu[s].gyr[k], FS, acc_in_g=False)
                sig[s] = (lowpass(a, FS, FC), lowpass(g, FS, FC))
            t = imu["foot"].t[k]
            ev = midswing((sig["shank"][1] @ OUT["shank"])[:, 1])
            keep = []
            for a_, b_ in zip(ev[:-1], ev[1:]):
                dd_ = (b_ - a_) / FS
                keep.append(dur_range[0] <= dd_ <= dur_range[1])
            for s in SEGS:
                for key, x in (("acc", sig[s][0]), ("gyr", sig[s][1])):
                    c, d = cycles(x, ev, dmin=dur_range[0], dmax=dur_range[1])
                    if len(c):
                        out[v][s][key].append(c)
            if v == next(iter(transforms)):
                for a_, b_ in zip(ev[:-1], ev[1:]):
                    dd_ = (b_ - a_) / FS
                    if dur_range[0] <= dd_ <= dur_range[1]:
                        h = hs_t[(hs_t > t[a_]) & (hs_t < t[b_])]
                        if len(h):
                            phis.append((h[0] - t[a_]) / (t[b_] - t[a_]))
                        durs.append(dd_)
    for v in out:
        for s in SEGS:
            for key in ("acc", "gyr"):
                out[v][s][key] = np.concatenate(out[v][s][key]) if out[v][s][key] else np.zeros((0, NPTS, 3))
    return out, np.array(phis), np.array(durs), trials


# ------------------------------------------------------------------ plots
def style(ax):
    ax.grid(color=GRID, lw=0.8)
    ax.tick_params(colors="#52514e", labelsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#c9c8c2")


def overview(path, title, gt_subj_means, exo_means, ph):
    fig, axes = plt.subplots(4, 3, figsize=(13, 12), sharex=True)
    for r, s in enumerate(SEGS):
        for j, (key, i, lab) in enumerate(CH):
            ax = axes[2 * r + j // 3, j % 3]
            M = np.array([m[s][key][:, i] for m in gt_subj_means.values()])
            for row in M:
                ax.plot(ph, row, color=C_GT, lw=0.6, alpha=0.25)
            mu, sd = M.mean(0), M.std(0)
            ax.fill_between(ph, mu - sd, mu + sd, color=C_GT, alpha=0.18, lw=0)
            ax.plot(ph, mu, color=C_GT, lw=2, label=f"Georgia Tech (n={len(M)} subjects, mean ± SD)")
            for (name, em), col in zip(exo_means.items(), (C_SEP, C_APR)):
                ax.plot(ph, em[s][key][:, i], color=col, lw=2, label=f"exo {name}")
            ax.set_title(f"{s} {lab}", fontsize=10, color=INK, loc="left")
            style(ax)
    for ax in axes[-1]:
        ax.set_xlabel("gait cycle [%] (0 = heel strike)", fontsize=9)
    axes[0, 0].legend(fontsize=8, frameon=False, loc="best")
    fig.suptitle(title, fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def subject_plot(path, subject, gt, exo_means, ph):
    fig, axes = plt.subplots(4, 3, figsize=(13, 11), sharex=True)
    for r, s in enumerate(SEGS):
        for j, (key, i, lab) in enumerate(CH):
            ax = axes[2 * r + j // 3, j % 3]
            c = gt[s][key][:, :, i]
            mu, sd = c.mean(0), c.std(0)
            ax.fill_between(ph, mu - sd, mu + sd, color=C_GT, alpha=0.2, lw=0)
            ax.plot(ph, mu, color=C_GT, lw=2, label=f"{subject} (own transform, {len(c)} strides, mean ± SD)")
            for (name, em), col in zip(exo_means.items(), (C_SEP, C_APR)):
                ax.plot(ph, em[s][key][:, i], color=col, lw=2, label=f"exo {name}")
            ax.set_title(f"{s} {lab}", fontsize=10, color=INK, loc="left")
            style(ax)
    for ax in axes[-1]:
        ax.set_xlabel("gait cycle [%] (0 = heel strike)", fontsize=9)
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle(f"Version 2 — {subject} with its own transform vs exo ({FRAME_NAME})", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=100)
    plt.close(fig)


def metrics(gt_mean, exo_mean):
    out = {}
    for s in SEGS:
        for key, i, _ in CH:
            a, b = gt_mean[s][key][:, i], exo_mean[s][key][:, i]
            out[f"{s}_{key}_{'xyz'[i]}"] = (float(np.corrcoef(a, b)[0, 1]), float(np.sqrt(np.mean((a - b) ** 2))))
    return out


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/exo_transform")
    ap.add_argument("--encoder-x", choices=["heel", "ankle"], default="heel",
                    help="encoder fore-aft position: from the heel measurement, or on the ankle joint centre")
    ap.add_argument("--frame", choices=["exo", "raw"], default="exo",
                    help="exo: exo frame (x fwd, y left, z up); raw: the exo IMUs' own sensor axes "
                         "(Sept 11 mounting)")
    ap.add_argument("--exo-dir", default="data/exo", help="exo recordings (data_collection_*.csv)")
    ap.add_argument("--calib-dir", default="data/exo/calibration", help="exo calibration recordings (*.npz)")
    ap.add_argument("--placement-dir", default="results", help="per-subject IMU placement results")
    ap.add_argument("--camargo-root", default=".", help="folder with the raw Camargo subject folders")
    a = ap.parse_args()
    global FRAME_NAME, CALIB, EXO_DIR, PLACEMENT, CAMARGO
    CALIB, EXO_DIR, PLACEMENT, CAMARGO = (Path(a.calib_dir), Path(a.exo_dir), Path(a.placement_dir),
                                          Path(a.camargo_root))
    if a.frame == "raw" and a.out == "results/exo_transform":
        a.out = "results/exo_transform_raw_frame"
    out = Path(a.out)
    (out / "version2_per_subject_plots").mkdir(parents=True, exist_ok=True)
    subjects = sorted(p.parent.name for p in PLACEMENT.glob("AB*/summary.json")
                      if re.fullmatch(r"AB\d+", p.parent.name))
    geo = ExoGeometry(encoder_x_anchor=a.encoder_x)

    # ---- exo side
    sessions, standing, hipswing = load_exo()
    Rexo, calib_report = calibrate_exo(sessions, standing, hipswing)
    if a.frame == "raw":
        for s in SEGS:
            OUT[s] = Rexo[RAW_SESSION][s].T          # exo frame -> raw sensor axes
        FRAME_NAME = f"exo IMU raw sensor axes ({RAW_SESSION.split(' (')[0]} mounting)"
    exo = exo_cycles(sessions, Rexo)
    all_dur = np.concatenate([e["dur"] for e in exo.values()])
    dur_range = (0.85 * np.percentile(all_dur, 5), 1.15 * np.percentile(all_dur, 95))

    # ---- transforms
    per = {}
    for s_ in subjects:
        summ = json.loads((PLACEMENT / s_ / "summary.json").read_text())
        osim = next((CAMARGO / s_ / "osimxml").glob("*.osim"))
        dd = next(p for p in (CAMARGO / s_).iterdir() if p.is_dir() and p.name[0].isdigit())
        # exo frame on the GT subject, defined like the exo calibration:
        # z = up during the subject's static standing, y = walking flexion axis
        up = standing_up(osim, next((dd / "static" / "ik").glob("*.mat")))
        tr0 = next(r["trial"] for r in json.loads((PLACEMENT / s_ / "per_trial.json").read_text())["foot"]
                   if r["mode"] == "treadmill")
        imu0 = load_imu(dd, "treadmill", tr0)
        R_s2e = {}
        for s in SEGS:
            Rg = np.array(summ[s]["R_sensor_to_segment_mujoco"])
            ax, _ = principal_axis(imu0[s].gyr @ Rg.T)
            R_s2e[s] = exo_frame_rotation(up[GT_BODY[s]], ax, 1.0 if ax[1] > 0 else -1.0)
        pos = exo_imu_positions(osim, geo, R_s2e)
        per[s_] = {s: Transform(R=R_s2e[s] @ np.array(summ[s]["R_sensor_to_segment_mujoco"]),
                                r_gt=np.array(summ[s]["position_m_mujoco_median"]),
                                r_exo=np.array(pos[s]), acc_sign=summ[s]["acc_sign"], R_s2e=R_s2e[s],
                                meta=dict(subject=s_, segment=GT_BODY[s],
                                          encoder_segment_m=np.round(pos["encoder_calcn" if s == "foot" else "encoder_tibia"], 4).tolist(),
                                          encoder_minus_ankle_joint_m=np.round(pos["encoder_minus_ankle"], 4).tolist()))
                   for s in SEGS}
    avg = {s: Transform(R=rotation_mean([per[x][s].R for x in subjects]),
                        r_gt=np.mean([per[x][s].r_gt for x in subjects], 0),
                        r_exo=np.mean([per[x][s].r_exo for x in subjects], 0), acc_sign=1,
                        d_override=np.mean([per[x][s].d for x in subjects], 0),
                        meta=dict(subject="average of " + ", ".join(subjects), segment=GT_BODY[s]))
           for s in SEGS}

    # rotation-only baseline (no lever arm), to show what the position correction adds
    avg_rot = {s: Transform(R=avg[s].R, r_gt=avg[s].r_gt, r_exo=avg[s].r_exo, d_override=np.zeros(3),
                            meta=dict(subject="average, rotation only")) for s in SEGS}

    if a.frame == "raw":
        def to_out(T, s):
            return Transform(R=OUT[s] @ T.R, r_gt=T.r_gt, r_exo=T.r_exo, acc_sign=T.acc_sign, R_s2e=T.R_s2e,
                             d_override=OUT[s] @ T.d,
                             meta={**T.meta, "frame": FRAME_NAME,
                                   "R_raw_from_exo_frame": np.round(OUT[s], 6).tolist()})
        per = {x: {s: to_out(per[x][s], s) for s in SEGS} for x in per}
        avg = {s: to_out(avg[s], s) for s in SEGS}
        avg_rot = {s: to_out(avg_rot[s], s) for s in SEGS}

    # ---- GT side
    gt = {}
    phis = []
    for s_ in subjects:
        cyc, ph_, du_, trials = gt_strides(s_, {"v1": avg, "v2": per[s_], "v1_rot_only": avg_rot}, dur_range)
        gt[s_] = dict(cyc=cyc, trials=trials, n=len(du_), dur=du_)
        phis.append(ph_)
        print(f"[gt] {s_}: {len(trials)} treadmill trials, {len(du_)} duration-matched strides", flush=True)
    phi = float(np.median(np.concatenate(phis)))
    ph = np.linspace(0, 100, NPTS)

    def mean_of(c):
        return {s: {k: rephase(c[s][k], phi).mean(0) for k in ("acc", "gyr")} for s in SEGS}

    exo_means = {n: mean_of(e["cyc"]) for n, e in exo.items()}
    v1_means = {s_: mean_of(gt[s_]["cyc"]["v1"]) for s_ in subjects if gt[s_]["n"] >= 5}
    v2_means = {s_: mean_of(gt[s_]["cyc"]["v2"]) for s_ in subjects if gt[s_]["n"] >= 5}
    v1r_means = {s_: mean_of(gt[s_]["cyc"]["v1_rot_only"]) for s_ in subjects if gt[s_]["n"] >= 5}
    # stride-averaged curves, for scripts/evaluate_match.py: key = version|who|segment|acc/gyr, (101, 3)
    flat = {}
    for ver, means in (("exo", exo_means), ("v1", v1_means), ("v2", v2_means), ("v1_rot_only", v1r_means)):
        for who, m in means.items():
            for s in SEGS:
                for k in ("acc", "gyr"):
                    flat[f"{ver}|{who}|{s}|{k}"] = m[s][k]
    np.savez(out / "mean_cycles.npz", **flat)

    overview(out / "version1_average_transform.png",
             f"Version 1 — every subject with the AVERAGE transform vs exo ({FRAME_NAME})",
             v1_means, exo_means, ph)
    overview(out / "version2_own_transform_overview.png",
             f"Version 2 — every subject with its OWN transform vs exo ({FRAME_NAME})",
             v2_means, exo_means, ph)
    for s_ in v2_means:
        g = {s: {k: rephase(gt[s_]["cyc"]["v2"][s][k], phi) for k in ("acc", "gyr")} for s in SEGS}
        subject_plot(out / "version2_per_subject_plots" / f"{s_}.png", s_, g, exo_means, ph)

    # ---- metrics: correlation and RMSE of mean cycles vs the Sept exo mean (longest session)
    ref = next(iter(exo_means))
    rows = []
    for ver, means in (("v1_average", v1_means), ("v2_own", v2_means), ("v1_rotation_only", v1r_means)):
        for s_, m in means.items():
            for ch, (r, e) in metrics(m, exo_means[ref]).items():
                rows.append(dict(version=ver, subject=s_, channel=ch, r=round(r, 3), rmse=round(e, 3)))
    with open(out / "match_metrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # heatmap of correlation per subject x channel (version 2) vs the exo
    chans = [f"{s}_{k}_{c}" for s in SEGS for k in ("acc", "gyr") for c in "xyz"]
    subs2 = list(v2_means)
    Rm = np.array([[next(r["r"] for r in rows if r["version"] == "v2_own" and r["subject"] == s_ and r["channel"] == ch)
                    for ch in chans] for s_ in subs2])
    fig, ax = plt.subplots(figsize=(11, 0.32 * len(subs2) + 2))
    im = ax.imshow(Rm, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(chans)), [c.replace("_", " ") for c in chans], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(subs2)), subs2, fontsize=8)
    for i in range(len(subs2)):
        for j in range(len(chans)):
            ax.text(j, i, f"{Rm[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if abs(Rm[i, j]) > 0.6 else INK)
    fig.colorbar(im, ax=ax, fraction=0.025, label="correlation with exo mean cycle")
    ax.set_title(f"Version 2: shape match per subject and channel (Pearson r vs exo {ref})", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(out / "version2_match_heatmap.png", dpi=110)
    plt.close(fig)

    # ---- write transforms
    (out / "transform_version1_average.json").write_text(json.dumps(
        {s: avg[s].to_dict() for s in SEGS}, indent=2))
    (out / "transform_version2_per_subject.json").write_text(json.dumps(
        {s_: {s: per[s_][s].to_dict() for s in SEGS} for s_ in subjects}, indent=2))
    with open(out / "transform_version2_per_subject.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["subject", "segment"] + [f"R{i}{j}" for i in range(3) for j in range(3)] +
                   ["d_x_m", "d_y_m", "d_z_m", "r_gt_x", "r_gt_y", "r_gt_z", "r_exo_x", "r_exo_y", "r_exo_z"])
        for s_ in ["average"] + subjects:
            for s in SEGS:
                T = avg[s] if s_ == "average" else per[s_][s]
                w.writerow([s_, s] + np.round(T.R, 6).ravel().tolist() + np.round(T.d, 5).tolist() +
                           np.round(T.r_gt, 5).tolist() + np.round(T.r_exo, 5).tolist())
    (out / "exo_imu_calibration.json").write_text(json.dumps(dict(
        R_exo_frame_from_exo_sensor={n: {s: np.round(R[s], 5).tolist() for s in SEGS} for n, R in Rexo.items()},
        report=calib_report, stride_duration_range_s=np.round(dur_range, 3).tolist(),
        exo_strides={n: int(len(e["dur"])) for n, e in exo.items()},
        exo_stride_duration_median_s={n: round(float(np.median(e["dur"])), 3) for n, e in exo.items()},
        heel_strike_phase_after_midswing=round(phi, 3)), indent=2))

    # ---- summary table (median over subjects)
    summ_rows = []
    for ver in ("v1_average", "v2_own", "v1_rotation_only"):
        for ch in chans:
            rs = [r for r in rows if r["version"] == ver and r["channel"] == ch]
            summ_rows.append((ver, ch, np.median([r["r"] for r in rs]), np.median([r["rmse"] for r in rs])))
    print("\nmedian over subjects, vs exo", ref)
    for ver, ch, r, e in summ_rows:
        print(f"  {ver:10s} {ch:14s} r={r:5.2f} rmse={e:5.2f}")
    json.dump({"summary": [dict(version=v, channel=c, median_r=round(r, 3), median_rmse=round(e, 3))
                           for v, c, r, e in summ_rows]},
              open(out / "match_summary.json", "w"), indent=2)
    print(f"\nexo strides: { {n: len(e['dur']) for n, e in exo.items()} }, duration range {np.round(dur_range, 2)} s, "
          f"HS phase after mid-swing {phi:.3f}")
    print(json.dumps(calib_report, indent=1))


if __name__ == "__main__":
    main()
