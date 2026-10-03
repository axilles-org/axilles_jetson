"""Correlation heatmaps (subject x channel) for the transform versions, from
results/exo_transform/match_metrics.csv written by scripts/exo_compare.py.

Writes:
  version1_match_heatmap.png          average transform
  version2_match_heatmap.png          each subject's own transform
  heatmap_v1_vs_v2.png                side by side, plus v2 - v1 difference

  python scripts/plot_heatmaps.py
"""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

CHANS = [f"{s}_{k}_{c}" for s in ("foot", "shank") for k in ("acc", "gyr") for c in "xyz"]
LABEL = lambda c: c.replace("_gyr_", " gyro ").replace("_acc_", " acc ")
INK = "#0b0b0b"
VERSIONS = {"v1_average": "Version 1 — average transform", "v2_own": "Version 2 — own transform"}


def matrix(rows, ver):
    subs = sorted({r["subject"] for r in rows if r["version"] == ver})
    M = np.full((len(subs), len(CHANS)), np.nan)
    for r in rows:
        if r["version"] == ver:
            M[subs.index(r["subject"]), CHANS.index(r["channel"])] = float(r["r"])
    return subs, M


def draw(ax, M, subs, title, cmap="RdBu", vmin=-1, vmax=1, fmt="{:.2f}", signed=False):
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(CHANS)), [LABEL(c) for c in CHANS], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(subs)), subs, fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            txt = ("{:+.2f}" if signed else fmt).format(v)
            ax.text(j, i, txt, ha="center", va="center", fontsize=6.5,
                    color="white" if abs(v) > 0.6 * max(abs(vmin), abs(vmax)) else INK)
    ax.set_title(title, fontsize=10, loc="left", color=INK)
    return im


def summary_row(M):
    return np.nanmedian(M, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/exo_transform")
    ap.add_argument("--exo", default="Sept 11 (2×60 s)")
    a = ap.parse_args()
    d = Path(a.dir)
    frame = ("exo IMU raw sensor axes" if "raw" in d.name else "exo frame: x fwd, y left, z up")
    rows = list(csv.DictReader(open(d / "match_metrics.csv")))

    mats = {}
    for ver, title in VERSIONS.items():
        subs, M = matrix(rows, ver)
        subs_m = subs + ["median"]
        Mm = np.vstack([M, summary_row(M)])
        mats[ver] = (subs, M)
        fig, ax = plt.subplots(figsize=(11, 0.32 * len(subs_m) + 2))
        im = draw(ax, Mm, subs_m, f"{title} [{frame}]: shape match per subject and channel (Pearson r vs exo {a.exo})")
        ax.axhline(len(subs) - 0.5, color=INK, lw=1)
        fig.colorbar(im, ax=ax, fraction=0.025, label="correlation with exo mean cycle")
        fig.tight_layout()
        name = "version1_match_heatmap.png" if ver == "v1_average" else "version2_match_heatmap.png"
        fig.savefig(d / name, dpi=110)
        plt.close(fig)

    (s1, M1), (s2, M2) = mats["v1_average"], mats["v2_own"]
    assert s1 == s2
    subs_m = s1 + ["median"]
    D = M2 - M1
    fig, axes = plt.subplots(1, 3, figsize=(25, 0.32 * len(subs_m) + 2.6), layout="constrained")
    for ax, (ver, title), M in zip(axes[:2], VERSIONS.items(), (M1, M2)):
        im = draw(ax, np.vstack([M, summary_row(M)]), subs_m, f"{title} (r)")
        ax.axhline(len(s1) - 0.5, color=INK, lw=1)
    fig.colorbar(im, ax=axes[1], fraction=0.04, label="correlation with exo mean cycle")
    lim = max(0.2, float(np.nanmax(np.abs(D))))
    im2 = draw(axes[2], np.vstack([D, summary_row(D)]), s1 + ["median of diff."],
               "Difference: version 2 − version 1 (blue = own transform matches better)",
               cmap="RdBu", vmin=-lim, vmax=lim, signed=True)
    axes[2].axhline(len(s1) - 0.5, color=INK, lw=1)
    fig.colorbar(im2, ax=axes[2], fraction=0.03, label="change in r")
    fig.suptitle(f"Average vs own transform — correlation with exo {a.exo}  [{frame}]", fontsize=12, color=INK)
    fig.savefig(d / "heatmap_v1_vs_v2.png", dpi=100, bbox_inches="tight")
    plt.close(fig)

    print("median r per channel:   v1    v2    v2-v1")
    for c, x, y in zip(CHANS, summary_row(M1), summary_row(M2)):
        print(f"  {LABEL(c):14s} {x:5.2f} {y:5.2f} {y - x:+6.2f}")
    print(f"cells where v2 is better by >0.05: {(D > 0.05).sum()}, worse by >0.05: {(D < -0.05).sum()}, "
          f"of {D.size}")
    print(f"wrote {d}/version1_match_heatmap.png, version2_match_heatmap.png, heatmap_v1_vs_v2.png")


if __name__ == "__main__":
    main()
