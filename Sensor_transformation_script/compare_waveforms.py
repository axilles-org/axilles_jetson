"""
compare_waveforms.py
====================
Waveform comparison of GaTech IMU data against your exo, BEFORE and AFTER the
transform, using standard gait-biomechanics methods:

  CMC  Coefficient of Multiple Correlation (Kadaba 1989; Ferrari 2010).
       One 0-1 similarity score per channel. > 0.9 very good, 0.75-0.9 good,
       0.6-0.75 moderate, < 0.6 poor. NaN when offsets are so large the
       formula goes complex -- itself a sign of poor agreement.

  LFM  Linear Fit Method (Iosa et al. 2014).  exo ~= a1 * gatech + a0
         a1 != 1  -> SCALE mismatch   (speed, lever arm, sensor gain)
         a0 != 0  -> OFFSET mismatch  (gravity orientation, bias)
         R2 < 1   -> SHAPE mismatch   (genuinely different motion)

  SPM  Statistical Parametric Mapping (Pataky; spm1d). Two-sample t-test at
       every point of the gait cycle, corrected for the whole curve. Reports
       WHERE in the stride the two differ significantly, not just by how much.

Usage (after run_alignment.py):

    python3 compare_waveforms.py \\
        --gatech-root mrsd-exo-ankle \\
        --tx-root mrsd-exo-ankle-tx \\
        --exo-log calib_poweroff_1p0mps_don1.csv \\
        --transforms transforms.json \\
        --out-dir comparison

Outputs in --out-dir:
  comparison_<segment>.png   mean strides before/after vs exo, SPM regions shaded
  comparison_report.json     every number printed to the console

Interpretation caveats
  * Channels are compared axis-by-axis. BEFORE the transform, GaTech's X is not
    your X, so "before" numbers mostly show how misaligned the frames are.
  * CMC and LFM compare each subject's mean stride with your mean stride, then
    summarise across subjects (median and IQR).
  * SPM compares the 22 subject-mean strides against your individual strides.
    With one exo subject, a significant region means "this person on this exo
    differs from the GaTech population here". That includes ordinary person-
    to-person differences and exo-induced gait change, not only sensor
    placement. It cannot separate those; it tells you where to look.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import align_core as ac
import align_io as aio

CH = ["ax", "ay", "az", "gx", "gy", "gz"]
UNIT = ["m/s²"] * 3 + ["rad/s"] * 3


# ---------------------------------------------------------------- metrics --

def cmc(y1, y2):
    """CMC between two waveforms of equal length (G = 2 groups)."""
    Y = np.vstack([y1, y2])
    G, F = Y.shape
    num = np.sum((Y - Y.mean(0)) ** 2) / (F * (G - 1))
    den = np.sum((Y - Y.mean()) ** 2) / (F * G - 1)
    if den <= 0:
        return np.nan
    r = num / den
    return float(np.sqrt(1 - r)) if r <= 1 else np.nan


def lfm(ref, test):
    """Linear Fit Method: test ~= a1 * ref + a0. Returns (a1, a0, R2)."""
    ref, test = np.asarray(ref, float), np.asarray(test, float)
    if np.std(ref) < 1e-9:
        return np.nan, np.nan, np.nan
    a1, a0 = np.polyfit(ref, test, 1)
    r = np.corrcoef(ref, test)[0, 1]
    return float(a1), float(a0), float(r * r)


def spm_ttest(YA, YB, alpha=0.05):
    """
    Two-sample SPM t-test (unequal variance). Returns (mask, method, zstar).
    Falls back to pointwise Welch t with Bonferroni if spm1d is missing
    (more conservative; not a true SPM).
    """
    YA, YB = np.asarray(YA, float), np.asarray(YB, float)
    try:
        import spm1d
        t = spm1d.stats.ttest2(YA, YB, equal_var=False)
        ti = t.inference(alpha=alpha, two_tailed=True)
        return np.abs(ti.z) > ti.zstar, "spm1d", float(ti.zstar)
    except ImportError:
        from scipy.stats import ttest_ind
        _, p = ttest_ind(YA, YB, axis=0, equal_var=False)
        return p < alpha / YA.shape[1], "pointwise-bonferroni", float("nan")


def clusters(mask):
    """Boolean mask over 0..100 % -> list of (start %, end %)."""
    return [(int(a), int(b - 1)) for a, b in ac.runs(mask)]


# ------------------------------------------------------------------ data --

def exo_strides(exo, seg):
    hs = exo.events.get("heel_strike", np.array([], int))
    e = ac.stride_ensemble(exo.accel[seg], exo.gyro[seg], exo.fs, hs)
    return e[:, :, :6]


def subject_mean(root, subj, trials, seg, shift):
    """Subject's mean stride (100, 6), moved onto the exo's timing."""
    ens = []
    for tr in trials:
        try:
            rec = aio.load_gatech(root, subj, tr, segments=(seg,))
        except Exception:                                   # noqa
            continue
        hs = rec.events.get("heel_strike", np.array([], int))
        if len(hs) < 5:
            continue
        e = ac.stride_ensemble(rec.accel[seg], rec.gyro[seg], rec.fs, hs)
        if len(e):
            ens.append(e[:, :, :6])
    if not ens:
        return None
    m = np.concatenate(ens).mean(0)
    # the fit aligned (source) with (target shifted by +shift); moving the
    # source by -shift puts it on the exo's own heel-strike timing instead
    return ac.circ_shift(m, -shift)


# ------------------------------------------------------------------ main --

def low_signal(E, frac=0.10):
    """
    Channels whose exo mean stride barely moves (range < frac of the largest
    channel of the same sensor type). CMC/LFM on a near-flat line are noise.
    """
    rng = np.ptp(E.mean(0), axis=0)
    flag = np.zeros(6, bool)
    for sl in (slice(0, 3), slice(3, 6)):
        top = rng[sl].max()
        flag[sl] = rng[sl] < frac * max(top, 1e-12)
    return flag, rng


def compare(label, S, E, alpha):
    """S: (n_subj, 100, 6) subject means. E: (n_str, 100, 6) exo strides."""
    Em = E.mean(0)
    flat, rng = low_signal(E)
    diff = np.abs(S.mean(0) - Em)                     # population vs exo
    out = {}
    for c in range(6):
        cm = np.array([cmc(s[:, c], Em[:, c]) for s in S])
        lf = np.array([lfm(s[:, c], Em[:, c]) for s in S])
        mask, method, _ = spm_ttest(S[:, :, c], E[:, :, c], alpha)
        # effect size: largest mean difference inside significant regions,
        # as % of the channel's range. Significant-but-tiny shows up here.
        eff = float(100 * diff[mask, c].max() / max(rng[c], 1e-12)) \
            if mask.any() else 0.0
        out[CH[c]] = dict(
            low_signal=bool(flat[c]),
            exo_range=float(rng[c]),
            spm_max_diff_pct=eff,
            cmc_median=float(np.nanmedian(cm)) if np.isfinite(cm).any() else None,
            cmc_iqr=[float(x) for x in np.nanpercentile(cm, [25, 75])]
            if np.isfinite(cm).any() else None,
            cmc_nan_frac=float(np.mean(~np.isfinite(cm))),
            a1_median=float(np.nanmedian(lf[:, 0])),
            a0_median=float(np.nanmedian(lf[:, 1])),
            r2_median=float(np.nanmedian(lf[:, 2])),
            spm_sig_pct=float(100 * mask.mean()),
            spm_clusters=clusters(mask),
            spm_method=method,
            _mask=mask)
    return out


def print_table(seg, before, after):
    print(f"\n{'-'*78}\n{seg.upper()}\n{'-'*78}")
    print(f"  {'ch':<3} {'':<7}{'CMC':>7} {'a1':>7} {'a0':>8} {'R²':>6} "
          f"{'SPM sig':>8} {'max Δ':>7}   significant regions (% cycle)")
    for c in CH:
        if after[c]["low_signal"]:
            print(f"  {c:<3} -- low signal (exo range {after[c]['exo_range']:.2f}"
                  f"); metrics not meaningful, skipped")
            continue
        for tag, d in (("before", before[c]), ("after", after[c])):
            cm = "  nan " if d["cmc_median"] is None else f"{d['cmc_median']:6.2f}"
            cl = ", ".join(f"{a}-{b}" for a, b in d["spm_clusters"][:4]) or "-"
            print(f"  {c if tag == 'before' else '':<3} {tag:<7}{cm:>7} "
                  f"{d['a1_median']:7.2f} {d['a0_median']:8.2f} "
                  f"{d['r2_median']:6.2f} {d['spm_sig_pct']:7.0f}% "
                  f"{d['spm_max_diff_pct']:6.0f}%   {cl}")
    ok = [c for c in CH if not after[c]["low_signal"]]
    if ok:
        med = lambda k, D: np.nanmedian([D[c][k] if D[c][k] is not None
                                         else np.nan for c in ok])
        print(f"  summary ({len(ok)} active channels): CMC "
              f"{med('cmc_median', before):.2f} -> {med('cmc_median', after):.2f}"
              f",  R² {med('r2_median', before):.2f} -> "
              f"{med('r2_median', after):.2f}")


def plot(seg, S0, S1, E, after, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.linspace(0, 100, E.shape[1])
    fig, axs = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for c, ax in enumerate(axs.ravel()):
        Em, Es = E[:, :, c].mean(0), E[:, :, c].std(0)
        m1, s1 = S1[:, :, c].mean(0), S1[:, :, c].std(0)
        ax.fill_between(x, Em - Es, Em + Es, color="C0", alpha=.15)
        ax.plot(x, Em, "C0", lw=2, label="your exo")
        ax.plot(x, S0[:, :, c].mean(0), "C3", lw=1, ls="--", label="GaTech before")
        ax.fill_between(x, m1 - s1, m1 + s1, color="C2", alpha=.15)
        ax.plot(x, m1, "C2", lw=2, label="GaTech after")
        for a, b in ac.runs(after[CH[c]]["_mask"]):
            ax.axvspan(x[a], x[min(b, len(x) - 1)], color="k", alpha=.07)
        d = after[CH[c]]
        cm = "nan" if d["cmc_median"] is None else f"{d['cmc_median']:.2f}"
        ttl = (f"{CH[c]}   LOW SIGNAL" if d["low_signal"] else
               f"{CH[c]}   CMC {cm}   R² {d['r2_median']:.2f}   "
               f"a1 {d['a1_median']:.2f}")
        ax.set_title(ttl, fontsize=9)
        ax.set_ylabel(UNIT[c]); ax.grid(alpha=.3)
        if c >= 3:
            ax.set_xlabel("gait cycle (%)")
    axs[0, 0].legend(fontsize=8)
    fig.suptitle(f"{seg}: mean ± SD.  Grey = SPM-significant difference (after)")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gatech-root", required=True, help="ORIGINAL dataset")
    p.add_argument("--tx-root", required=True, help="TRANSFORMED dataset")
    p.add_argument("--exo-log", required=True)
    p.add_argument("--transforms", default="transforms.json")
    p.add_argument("--speed", nargs=2, type=float, default=[0.85, 1.10])
    p.add_argument("--segments", nargs="+", default=["foot", "shank"])
    p.add_argument("--fs", type=float, default=None)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--out-dir", default="comparison")
    a = p.parse_args()

    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    tx_info = json.loads(Path(a.transforms).read_text())
    meta = pd.read_parquet(Path(a.gatech_root) / "metadata.parquet")
    if "speed_mean_mps" in meta:
        meta = meta[(meta.speed_mean_mps >= a.speed[0]) &
                    (meta.speed_mean_mps <= a.speed[1])]
    trials = meta.groupby("subject")["trial"].apply(list).to_dict()

    exo = aio.load_exo(a.exo_log, fs=a.fs, verbose=False)
    print(f"exo: {exo.meta['file']}  "
          f"{len(exo.events.get('heel_strike', []))} heel strikes")

    report = {}
    for seg in a.segments:
        if seg not in tx_info or seg not in exo.accel:
            print(f"\n{seg}: no transform or no exo data, skipped")
            continue
        E = exo_strides(exo, seg)
        per = tx_info[seg]["per_subject"]
        S0, S1, used = [], [], []
        for subj, info in per.items():
            if subj not in trials:
                continue
            if not (Path(a.tx_root) / "subjects" / subj).exists():
                continue                      # dropped as an outlier
            sh = float(info.get("shift", 0.0))
            m0 = subject_mean(a.gatech_root, subj, trials[subj], seg, sh)
            m1 = subject_mean(a.tx_root, subj, trials[subj], seg, sh)
            if m0 is not None and m1 is not None:
                S0.append(m0); S1.append(m1); used.append(subj)
        if len(used) < 2 or len(E) < 3:
            print(f"\n{seg}: not enough data ({len(used)} subjects, "
                  f"{len(E)} exo strides)")
            continue
        S0, S1 = np.array(S0), np.array(S1)
        before = compare("before", S0, E, a.alpha)
        after = compare("after", S1, E, a.alpha)
        print_table(f"{seg}  ({len(used)} subjects vs {len(E)} exo strides)",
                    before, after)
        plot(seg, S0, S1, E, after, out / f"comparison_{seg}.png")

        strip = lambda d: {c: {k: v for k, v in d[c].items() if k != "_mask"}
                           for c in d}
        report[seg] = dict(subjects=used, n_exo_strides=int(len(E)),
                           before=strip(before), after=strip(after))

    method = next((v["after"]["ax"]["spm_method"] for v in report.values()), "-")
    (out / "comparison_report.json").write_text(json.dumps(report, indent=1))
    print(f"\nSPM method: {method}.  Plots + report -> {out}/")
    print("Read: CMC up toward 1, a1 toward 1, a0 toward 0, R² up, SPM % down.")
    print("'max Δ' = biggest mean difference inside significant regions, as % of "
          "the channel's range. Significant but < ~10% is usually unimportant.")


if __name__ == "__main__":
    main()