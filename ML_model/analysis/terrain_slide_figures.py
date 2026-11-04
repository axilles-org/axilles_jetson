"""Slide figures for the multi-terrain architecture comparison.

    python analysis/terrain_slide_figures.py --csv ~/PycharmProjects/exo-data/csv_output \
        --out runs/terrain/slides

architectures.png  the four compared models side by side
windowing.png      one level-ground trial: torque exists only on force-plate steps,
                   and which part of a 3 s window the loss scores
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

BLUE, ORANGE, AQUA, VIOLET, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#8a8984"
INK, INK2, LINE = "#0b0b0b", "#52514e", "#c9c8c3"


def box(ax, x, y, w, h, text, color, fill=0.14, size=10, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                fc=matplotlib.colors.to_rgba(color, fill), ec=color, lw=1.6))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=INK,
            weight=weight, linespacing=1.3)


def arrow(ax, x0, y0, x1, y1, color=INK2, style="-|>", lw=1.4, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=12,
                                 color=color, lw=lw, linestyle=ls, shrinkA=0, shrinkB=0))


def trunk(ax, x, y, w, h, n=5, film=False):
    """Five temporal blocks stacked left to right; returns block centres."""
    gap = 0.022
    bw = (w - gap * (n - 1)) / n
    centres = []
    for i in range(n):
        bx = x + i * (bw + gap)
        box(ax, bx, y, bw, h, f"block {i + 1}\nd={2 ** i}", BLUE, size=8.5)
        centres.append(bx + bw / 2)
        if i:
            arrow(ax, bx - gap + 0.003, y + h / 2, bx - 0.003, y + h / 2)
    ax.text(x + w / 2, y - 0.045, "causal TCN trunk: 5 blocks × 80 ch, kernel 5, receptive field 2.5 s",
            ha="center", va="top", fontsize=8.5, color=INK2)
    return centres, bw


def panel(ax, title, subtitle):
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ax.text(0.0, 0.98, title, fontsize=13, weight="bold", color=INK, va="top")
    ax.text(0.0, 0.905, subtitle, fontsize=9.5, color=INK2, va="top")


def inputs(ax, y, emb=False):
    box(ax, 0.015, y, 0.14, 0.2, "sensor window\n14 ch × 300\n(3 s @ 100 Hz)", GRAY, size=8.5)
    if emb:
        box(ax, 0.015, y - 0.3, 0.14, 0.2, "subject\nembedding\n4-d (nearest\ntraining subject)", VIOLET, size=7.5)
    return y + 0.1


def draw_architectures(path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 8.6))
    y_t, h_t = 0.42, 0.16

    # plain TCN
    ax = axes[0, 0]
    panel(ax, "1  Plain TCN", "no wearer or terrain information beyond the sensors")
    mid = inputs(ax, 0.4)
    trunk(ax, 0.2, y_t, 0.56, h_t)
    arrow(ax, 0.15, mid, 0.2, y_t + h_t / 2)
    box(ax, 0.8, y_t, 0.08, h_t, "1×1\nhead", BLUE, size=8.5)
    arrow(ax, 0.76, y_t + h_t / 2, 0.8, y_t + h_t / 2)
    ax.text(0.9, y_t + h_t / 2, "τ̂(t)\nN·m/kg", fontsize=10, va="center", color=INK)
    arrow(ax, 0.88, y_t + h_t / 2, 0.9, y_t + h_t / 2)

    # TCN + embedding
    ax = axes[0, 1]
    panel(ax, "2  TCN + subject embedding  (deployed today)",
          "a learned 4-d vector per training subject, concatenated to every timestep")
    mid = inputs(ax, 0.4, emb=True)
    trunk(ax, 0.2, y_t, 0.56, h_t)
    arrow(ax, 0.15, mid, 0.2, y_t + h_t / 2)
    arrow(ax, 0.155, 0.2, 0.2, y_t + 0.03, color=VIOLET)
    ax.text(0.17, 0.24, "concat", fontsize=8, color=VIOLET, rotation=55)
    box(ax, 0.8, y_t, 0.08, h_t, "1×1\nhead", BLUE, size=8.5)
    arrow(ax, 0.76, y_t + h_t / 2, 0.8, y_t + h_t / 2)
    ax.text(0.9, y_t + h_t / 2, "τ̂(t)\nN·m/kg", fontsize=10, va="center", color=INK)
    arrow(ax, 0.88, y_t + h_t / 2, 0.9, y_t + h_t / 2)

    # FiLM
    ax = axes[1, 0]
    panel(ax, "3  TCN + FiLM on the subject embedding", "the same learnable 4-d embedding as model 2, "
          "but it rescales every trunk block\ninstead of being concatenated to the input:  "
          "h ← h · (1 + γ) + β")
    y_f = 0.32
    mid = inputs(ax, y_f)
    centres, bw = trunk(ax, 0.2, y_f, 0.56, h_t)
    arrow(ax, 0.15, mid, 0.2, y_f + h_t / 2)
    box(ax, 0.015, 0.67, 0.14, 0.13, "subject\nembedding 4-d\n(nearest training\nsubject)", VIOLET, size=7.5)
    box(ax, 0.22, 0.67, 0.28, 0.13, "linear 4 → 800\n(γ, β) for each block", ORANGE, size=8.5)
    arrow(ax, 0.155, 0.735, 0.22, 0.735, color=VIOLET)
    ax.add_line(plt.Line2D([0.5, max(centres)], [0.735, 0.735], color=ORANGE, lw=1.1))
    for c in centres:
        arrow(ax, c, 0.735 if c > 0.5 else 0.67, c, y_f + h_t, color=ORANGE, lw=1.1)
    box(ax, 0.8, y_f, 0.08, h_t, "1×1\nhead", BLUE, size=8.5)
    arrow(ax, 0.76, y_f + h_t / 2, 0.8, y_f + h_t / 2)
    ax.text(0.9, y_f + h_t / 2, "τ̂(t)\nN·m/kg", fontsize=10, va="center", color=INK)
    arrow(ax, 0.88, y_f + h_t / 2, 0.9, y_f + h_t / 2)
    ax.text(0.52, 0.81, "zero-initialised: starts as the plain TCN", fontsize=8, color=INK2)

    # MoE
    ax = axes[1, 1]
    panel(ax, "4  TCN + mixture of experts", "4 expert heads share the trunk; a per-timestep softmax gate "
          "mixes them\n(no terrain labels given; load-balancing loss keeps all experts in use)")
    y_m = 0.32
    mid = inputs(ax, y_m)
    centres, _ = trunk(ax, 0.2, y_m, 0.44, h_t)
    arrow(ax, 0.15, mid, 0.2, y_m + h_t / 2)
    feat = (0.64, y_m + h_t / 2)
    ys = [0.72, 0.6, 0.48, 0.36]
    for i, ey in enumerate(ys):
        box(ax, 0.69, ey - 0.045, 0.1, 0.09, f"expert {i + 1}\n80→32→1", AQUA, size=7.5)
        arrow(ax, feat[0], feat[1], 0.69, ey, color=AQUA, lw=1)
        arrow(ax, 0.79, ey, 0.82, y_m + h_t / 2 + 0.02, color=AQUA, lw=1)
    box(ax, 0.69, 0.1, 0.1, 0.1, "gate\nsoftmax (4)", AQUA, size=8)
    arrow(ax, feat[0], feat[1], 0.69, 0.15, color=AQUA, lw=1)
    arrow(ax, 0.79, 0.15, 0.835, y_m + h_t / 2 - 0.03, color=AQUA, lw=1, ls="--")
    ax.add_patch(plt.Circle((0.845, y_m + h_t / 2), 0.03, fc="white", ec=AQUA, lw=1.6))
    ax.text(0.845, y_m + h_t / 2, "Σ", ha="center", va="center", fontsize=12, color=INK)
    ax.text(0.92, y_m + h_t / 2, "τ̂(t)", fontsize=10, va="center", color=INK)
    arrow(ax, 0.875, y_m + h_t / 2, 0.915, y_m + h_t / 2)
    ax.text(0.845, y_m + h_t / 2 - 0.08, "Σ gₖ·yₖ", ha="center", fontsize=8.5, color=INK2)

    fig.text(0.5, 0.015, "All four: same 14 inputs, same trunk, same training recipe; causal, so only the "
             "last-timestep output is used on the exo.", ha="center", fontsize=10, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1), h_pad=2.5, w_pad=3)
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def draw_windowing(csv_root: Path, path: Path, subject="AB06", trial="levelground_ccw_normal_01_01") -> None:
    base = csv_root / subject / "levelground"
    tq = pd.read_csv(base / "id" / trial / "data.csv", usecols=["Header", "ankle_angle_r_moment"])
    lab = pd.read_csv(base / "conditions" / trial / "labels.csv")
    t = tq.Header.to_numpy() - tq.Header.iloc[0]
    y = tq.ankle_angle_r_moment.to_numpy()
    valid = ~np.isnan(y)
    labels = lab.Label.astype(str).to_numpy()[:len(t)]

    fig, (ax, axl) = plt.subplots(2, 1, figsize=(12, 4.6), sharex=True,
                                  gridspec_kw={"height_ratios": [4, 0.6], "hspace": 0.08})
    ax.fill_between(t, 0, 1, where=~valid, color="#ecebe7", step="mid", lw=0,
                    transform=ax.get_xaxis_transform(), label="no force plate → no torque label")
    ax.plot(t, y, color=BLUE, lw=1.6, label="ankle moment, inverse dynamics (N·m)")
    walk = np.flatnonzero(valid & (labels == "walk"))
    end = t[walk[len(walk) // 2]]
    ax.axvspan(end - 3.0, end, color=ORANGE, alpha=0.08, lw=0)
    ax.axvspan(end - 0.5, end, color=ORANGE, alpha=0.3, lw=0)
    top = np.nanmax(y)
    ax.annotate("", (end - 3.0, top * 1.12), (end, top * 1.12),
                arrowprops=dict(arrowstyle="<->", color=ORANGE, lw=1.4))
    ax.text(end - 1.5, top * 1.3, "model input: 3 s (300 samples), sensors must be complete;\n"
            "torque may be missing here", ha="center", va="bottom", color=INK, fontsize=9.5)
    ax.annotate("scored by the loss: last 0.5 s (50 samples)\nmust have a torque label",
                (end - 0.25, np.nanmin(y) * 0.55), (end + 1.4, np.nanmin(y) * 0.55), color=INK,
                fontsize=9.5, va="center", bbox=dict(fc="white", ec="none", alpha=0.9),
                arrowprops=dict(arrowstyle="-", color=ORANGE, lw=1.2))
    ax.set(ylabel="N·m", ylim=(np.nanmin(y) * 1.1, top * 2.6),
           title=f"Level-ground trial ({subject}): torque exists only on force-plate steps")
    ax.legend(loc="lower left", fontsize=9, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    cats = {"walk": BLUE, "turn": VIOLET, "stand": GRAY, "transition": AQUA}
    def cat(lbl):
        return "transition" if "-" in lbl else "turn" if lbl.startswith("turn") else lbl if lbl in cats else None
    for i in range(len(t) - 1):
        c = cat(labels[i])
        if c:
            axl.axvspan(t[i], t[i + 1], color=cats[c], lw=0)
    axl.set_yticks([])
    axl.set_xlabel("Time (s)")
    axl.spines[["top", "right", "left"]].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in cats.values()]
    axl.legend(handles, cats.keys(), ncol=4, fontsize=8.5, frameon=False, loc="upper center",
               bbox_to_anchor=(0.5, -0.9))
    axl.text(-0.01, 0.5, "dataset\nlabel", transform=axl.transAxes, ha="right", va="center", fontsize=8.5, color=INK2)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    draw_architectures(a.out / "architectures.png")
    draw_windowing(a.csv.expanduser(), a.out / "windowing.png")
    print(f"figures -> {a.out}")


if __name__ == "__main__":
    main()
