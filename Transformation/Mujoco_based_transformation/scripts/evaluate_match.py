"""Numeric evaluation of the Georgia Tech -> exo transform (uses mean_cycles.npz
written by scripts/exo_compare.py).

Per channel (12 = foot/shank x acc/gyro x xyz), comparing the exo's stride-averaged
curve with the transformed Georgia Tech subjects:

  r        Pearson correlation: shape/timing similarity (ignores offset and scale)
  RMSE     absolute error [m/s^2 or rad/s]
  NRMSE    RMSE / RMS of the Georgia Tech mean curve's variation (unitless, comparable
           across channels; ~1 means "error as large as the signal itself")
  gain     least-squares slope exo ~ GT (1 = same amplitude)
  offset   mean difference exo - GT
  inband   % of the cycle where the exo curve lies inside the GT mean +- 2 SD (across subjects)
  LOO      exo-to-GT RMSE divided by the median subject-to-rest RMSE (leave one out):
           <= 1 means the exo differs from the GT average no more than a typical GT
           subject does, i.e. it is indistinguishable from "one more subject"

and two direct measures of the rotation, in degrees:
  stance gravity angle   angle between the exo and GT mean accel vectors in mid-stance
                         (15-40% of the cycle, foot flat): a pure orientation error
  rotation-axis angle    angle between the dominant gyro axis of the exo and GT curves

  python scripts/evaluate_match.py [--version v1|v2|v1_rot_only] [--exo "Sept 11 (2×60 s)"]
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

SEGS = ("foot", "shank")
KEYS = ("acc", "gyr")


def load(path):
    d = np.load(path)
    out = {}
    for k in d.files:
        ver, who, s, key = k.split("|")
        out.setdefault(ver, {}).setdefault(who, {}).setdefault(s, {})[key] = d[k]
    return out


def ang(a, b):
    return float(np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/exo_transform")
    ap.add_argument("--version", default="v2")
    ap.add_argument("--exo", default="Sept 11 (2×60 s)")
    a = ap.parse_args()
    D = load(Path(a.dir) / "mean_cycles.npz")
    exo, gt = D["exo"][a.exo], D[a.version]
    subs = sorted(gt)
    rows = []
    for s in SEGS:
        for k in KEYS:
            G = np.array([gt[x][s][k] for x in subs])          # (n_subj, 101, 3)
            E = exo[s][k]
            mu, sd = G.mean(0), G.std(0)
            for i, c in enumerate("xyz"):
                g, e = mu[:, i], E[:, i]
                rmse = float(np.sqrt(np.mean((e - g) ** 2)))
                sig = float(np.sqrt(np.mean((g - g.mean()) ** 2)))
                gc = g - g.mean()
                gain = float(gc @ (e - e.mean()) / (gc @ gc)) if sig > 0 else np.nan
                loo = [np.sqrt(np.mean((G[j, :, i] - np.delete(G, j, 0)[:, :, i].mean(0)) ** 2))
                       for j in range(len(subs))]
                rows.append(dict(
                    channel=f"{s} {k} {c}", r=float(np.corrcoef(g, e)[0, 1]), rmse=rmse,
                    nrmse=rmse / sig if sig > 0 else np.nan, gain=gain, offset=float(np.mean(e - g)),
                    inband_pct=float(100 * np.mean(np.abs(e - g) <= 2 * sd[:, i])),
                    loo_ratio=rmse / float(np.median(loo)), gt_signal_rms=sig))
    # direct rotation checks, per subject then median
    geo = {}
    st = slice(15, 41)
    for s in SEGS:
        gang = [ang(exo[s]["acc"][st].mean(0), gt[x][s]["acc"][st].mean(0)) for x in subs]

        def axis(w):
            w = w - w.mean(0)
            v = np.linalg.eigh(w.T @ w)[1][:, -1]
            return v
        aang = [min(ang(axis(exo[s]["gyr"]), axis(gt[x][s]["gyr"])),
                    ang(-axis(exo[s]["gyr"]), axis(gt[x][s]["gyr"]))) for x in subs]
        geo[s] = dict(stance_gravity_angle_deg_median=round(float(np.median(gang)), 2),
                      stance_gravity_angle_deg_range=[round(min(gang), 2), round(max(gang), 2)],
                      rotation_axis_angle_deg_median=round(float(np.median(aang)), 2),
                      rotation_axis_angle_deg_range=[round(min(aang), 2), round(max(aang), 2)])

    out = Path(a.dir) / f"evaluation_{a.version}.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    (Path(a.dir) / f"evaluation_{a.version}_rotation.json").write_text(json.dumps(geo, indent=2))
    print(f"version {a.version} vs exo {a.exo}, {len(subs)} subjects")
    print(f"{'channel':15s} {'r':>5s} {'RMSE':>6s} {'NRMSE':>6s} {'gain':>5s} {'offset':>7s} {'in band':>8s} {'LOO':>5s}")
    for r in rows:
        print(f"{r['channel']:15s} {r['r']:5.2f} {r['rmse']:6.2f} {r['nrmse']:6.2f} {r['gain']:5.2f} "
              f"{r['offset']:7.2f} {r['inband_pct']:7.0f}% {r['loo_ratio']:5.2f}")
    print(json.dumps(geo, indent=1))


if __name__ == "__main__":
    main()
