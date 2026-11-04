"""One architecture diagram per terrain_experiment variant, as SVG (for slides) and a
3x PNG rendered with headless Chrome. All five share one canvas and trunk position so
they line up slide to slide.

    python analysis/architecture_diagrams.py [--out runs/terrain/slides/architectures]
"""
from __future__ import annotations

import argparse
import math
import subprocess
from pathlib import Path

import numpy as np

W, H = 1560, 640
Y = 290                                   # main signal line
BX0, BW, BH, PITCH = 330, 80, 200, 140    # trunk blocks
BLOCKS = [BX0 + i * PITCH for i in range(5)]
NODES = [b + BW + (PITCH - BW) / 2 for b in BLOCKS]   # FiLM nodes between blocks
TRUNK_END = BLOCKS[-1] + BW
OUT_X, OUT_W = 1300, 210
BUS_Y = 432
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

INK, MUTED, LINE = "#1e293b", "#64748b", "#475569"
C = {
    "trunk": ("#eaf0f8", "#34507e"), "head": ("#f1f5f9", "#64748b"),
    "emb": ("#f0ebfb", "#6a4dbf"), "film": ("#fdf1de", "#bd7410"),
    "ctx": ("#e1f3f0", "#1d8475"), "exp": ("#e6f0fb", "#2e69b0"),
    "gate": ("#fbe9ed", "#b3405a"), "in": ("#ffffff", "#334155"),
}
SANS = "'Helvetica Neue', Helvetica, Arial, sans-serif"
SERIF = "'Times New Roman', Times, serif"


class Svg:
    def __init__(self, title: str):
        self.el = [f'<text x="48" y="74" font-size="34" font-weight="600" fill="{INK}" '
                   f'letter-spacing="-0.3">{title}</text>',
                   f'<line x1="48" y1="94" x2="{W - 48}" y2="94" stroke="#e2e8f0" stroke-width="1.5"/>']
        self.markers = set()

    def add(self, s: str) -> None:
        self.el.append(s)

    def text(self, x, y, s, size=16, color=INK, weight=400, anchor="middle", family=SANS, italic=False, rot=None):
        style = ' font-style="italic"' if italic else ""
        tr = f' transform="rotate({rot} {x} {y})"' if rot is not None else ""
        self.add(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}" '
                 f'text-anchor="{anchor}" dominant-baseline="central" font-family="{family}"{style}{tr}>{s}</text>')

    def box(self, x, y, w, h, kind, rx=10, shadow=True):
        fill, stroke = C[kind]
        f = ' filter="url(#sh)"' if shadow else ""
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
                 f'stroke="{stroke}" stroke-width="1.8"{f}/>')

    def arrow(self, pts, color=LINE, width=2.0, head=True, dash=None):
        self.markers.add(color)
        d = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        m = f' marker-end="url(#ah{color[1:]})"' if head else ""
        ds = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" '
                 f'stroke-linejoin="round"{m}{ds}/>')

    def dot(self, x, y, color=LINE, r=4):
        self.add(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{color}"/>')

    def circle(self, x, y, r, kind, label, size=15, italic=False, family=SANS):
        fill, stroke = C[kind]
        self.add(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="1.8" filter="url(#sh)"/>')
        self.text(x, y + 0.5, label, size, stroke, 600, family=family, italic=italic)

    def render(self) -> str:
        marks = "".join(
            f'<marker id="ah{c[1:]}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
            f'orient="auto-start-reverse"><path d="M0,0.5 L10,5 L0,9.5 z" fill="{c}"/></marker>'
            for c in sorted(self.markers))
        defs = (f'<defs>{marks}<filter id="sh" x="-20%" y="-20%" width="140%" height="140%">'
                '<feDropShadow dx="0" dy="1.5" stdDeviation="2" flood-color="#0f172a" flood-opacity="0.10"/>'
                '</filter></defs>')
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
                f'font-family="{SANS}"><rect width="{W}" height="{H}" fill="#ffffff"/>{defs}'
                + "".join(self.el) + "</svg>")


def _poly(xs, ys) -> str:
    return " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))


def draw_input(s: Svg) -> None:
    x, y, w, h = 52, 232, 140, 116
    for k in (2, 1):
        s.add(f'<rect x="{x + 8 * k}" y="{y - 8 * k}" width="{w}" height="{h}" rx="8" fill="#f8fafc" '
              f'stroke="#94a3b8" stroke-width="1.4"/>')
    s.box(x, y, w, h, "in", rx=8)
    t = np.linspace(0, 1, 80)
    waves = [(np.sin(2 * np.pi * 2 * t) + 0.35 * np.sin(2 * np.pi * 6 * t), "#2e69b0"),
             (np.sin(2 * np.pi * 2 * t + 1.3) * 0.8 + 0.3 * np.sin(2 * np.pi * 4 * t), "#1d8475"),
             ((np.sin(2 * np.pi * 2 * t + 2.4) > 0) * 1.6 - 0.8, "#6a4dbf"),
             (np.sin(2 * np.pi * 2 * t - 0.6) * 0.9, "#bd7410")]
    for i, (v, col) in enumerate(waves):
        cy = y + 20 + i * 25.5
        s.add(f'<polyline points="{_poly(x + 12 + t * (w - 24), cy - v * 8)}" fill="none" '
              f'stroke="{col}" stroke-width="1.6" stroke-linejoin="round"/>')
    s.text(x + w / 2 + 8, y + h + 28, '<tspan font-style="italic">x</tspan> ∈ ℝ'
           '<tspan dy="-8" font-size="13">14×300</tspan>', 21, INK, family=SERIF)


def draw_trunk(s: Svg, film: bool = False) -> None:
    s.add(f'<path d="M{BX0},176 L{BX0},166 L{TRUNK_END},166 L{TRUNK_END},176" fill="none" '
          f'stroke="{MUTED}" stroke-width="1.4"/>')
    s.text((BX0 + TRUNK_END) / 2, 146, "causal dilated conv  ·  k = 5  ·  80 ch", 16, MUTED)
    for i, bx in enumerate(BLOCKS):
        s.box(bx, Y - BH / 2, BW, BH, "trunk")
        s.text(bx + BW / 2, Y - 14, "Temporal Block", 16, C["trunk"][1], 600, rot=-90)
        s.text(bx + BW / 2, Y + BH / 2 - 22, f'<tspan font-style="italic" font-family="Times New Roman">d</tspan> = {2 ** i}',
               16, C["trunk"][1])
        if i < 4:
            if film:
                s.arrow([(bx + BW, Y), (NODES[i] - 17, Y)])
                s.circle(NODES[i], Y, 16, "film", "γ,β", 13, italic=True, family=SERIF)
                s.arrow([(NODES[i] + 17, Y), (BLOCKS[i + 1], Y)])
            else:
                s.arrow([(bx + BW, Y), (BLOCKS[i + 1], Y)])
    if film:
        s.arrow([(TRUNK_END, Y), (NODES[4] - 17, Y)])
        s.circle(NODES[4], Y, 16, "film", "γ,β", 13, italic=True, family=SERIF)


def draw_film_bus(s: Svg, x_from: float, label: str) -> None:
    fs = C["film"][1]
    s.arrow([(x_from, BUS_Y), (NODES[-1], BUS_Y)], fs, 2.2, head=False)
    for n in NODES:
        s.arrow([(n, BUS_Y), (n, Y + 18)], fs, 2.0)
    s.text(x_from + 14, BUS_Y + 20, label, 19, fs, 600, anchor="start", family=SERIF, italic=True)


def draw_head_output(s: Svg, start: float) -> None:
    s.arrow([(start, Y), (1060, Y)])
    s.box(1060, Y - 30, 120, 60, "head")
    s.text(1120, Y, "1×1 conv", 16, INK, 500)
    s.arrow([(1180, Y), (OUT_X, Y)])
    draw_output(s)


def draw_output(s: Svg) -> None:
    x, y, w, h = OUT_X, Y - 62, OUT_W, 124
    s.box(x, y, w, h, "in", rx=8)
    base = y + h - 26
    s.add(f'<line x1="{x + 14}" y1="{base}" x2="{x + w - 14}" y2="{base}" stroke="#cbd5e1" stroke-width="1.2"/>')
    t = np.linspace(0, 1, 160)
    rise = np.clip(t / 0.48, 0, 1) ** 2.4
    fall = np.where(t > 0.48, np.cos(np.clip((t - 0.48) / 0.14, 0, 1) * np.pi / 2) ** 2, 1)
    v = rise * fall - 0.06 * np.sin(np.clip(t / 0.12, 0, 1) * np.pi)
    s.add(f'<polyline points="{_poly(x + 14 + t * (w - 28), base - v * 78)}" fill="none" '
          f'stroke="#c2410c" stroke-width="2.6" stroke-linejoin="round"/>')
    tx = x + w / 2 - 62
    s.text(tx, y + h + 30, "τ", 22, INK, anchor="start", family=SERIF, italic=True)
    hy = y + h + 21
    s.add(f'<path d="M{tx + 3},{hy + 4} L{tx + 7.5},{hy} L{tx + 12},{hy + 4}" fill="none" '
          f'stroke="{INK}" stroke-width="1.4" stroke-linejoin="round"/>')
    s.text(tx + 18, y + h + 29, "ankle moment", 18, INK, anchor="start")


def draw_embedding(s: Svg) -> tuple[float, float]:
    """Subject ID -> embedding table; returns the table's right-edge anchor."""
    fill, stroke = C["emb"]
    hx, hy = 76, 486
    s.add(f'<circle cx="{hx}" cy="{hy}" r="12" fill="{fill}" stroke="{stroke}" stroke-width="1.8"/>')
    s.add(f'<path d="M{hx - 21},{hy + 40} Q{hx - 21},{hy + 16} {hx},{hy + 16} Q{hx + 21},{hy + 16} '
          f'{hx + 21},{hy + 40} Z" fill="{fill}" stroke="{stroke}" stroke-width="1.8"/>')
    s.text(hx, hy + 66, "subject", 15, MUTED)
    tx, ty, cell, rows = 128, 452, 10, 13
    for r in range(rows):
        for c in range(4):
            on = r == 7
            s.add(f'<rect x="{tx + c * cell}" y="{ty + r * cell}" width="{cell - 1.5}" height="{cell - 1.5}" '
                  f'rx="1.5" fill="{stroke if on else fill}" stroke="{stroke}" stroke-width="{0 if on else 0.8}"/>')
    s.arrow([(hx + 26, hy + 18), (tx - 8, ty + 7.5 * cell)], stroke, 1.8)
    s.text(tx + 2 * cell, ty + rows * cell + 22, '<tspan font-style="italic">E</tspan> ∈ ℝ'
           '<tspan dy="-7" font-size="12">13×4</tspan>', 18, INK, family=SERIF)
    return tx + 4 * cell, ty + 7.5 * cell


def plain_tcn() -> Svg:
    s = Svg("Plain TCN")
    draw_input(s)
    s.arrow([(208, Y), (BX0, Y)])
    draw_trunk(s)
    draw_head_output(s, TRUNK_END)
    return s


def tcn_emb() -> Svg:
    s = Svg("TCN + Subject Embedding")
    draw_input(s)
    cx = 269
    s.arrow([(208, Y), (cx - 17, Y)])
    s.circle(cx, Y, 16, "head", "‖", 17)
    s.arrow([(cx + 17, Y), (BX0, Y)])
    ex, ey = draw_embedding(s)
    fill, stroke = C["emb"]
    s.arrow([(ex + 4, ey), (cx - 44, ey)], stroke, 1.8)
    s.box(cx - 44, ey - 24, 88, 48, "emb")
    s.text(cx, ey, "repeat ×<tspan font-style='italic' font-family='Times New Roman'>T</tspan>", 15, stroke, 600)
    s.arrow([(cx, ey - 24), (cx, Y + 18)], stroke, 2.0)
    draw_trunk(s)
    draw_head_output(s, TRUNK_END)
    return s


def film_subject() -> Svg:
    s = Svg("TCN + FiLM on Subject Embedding")
    draw_input(s)
    s.arrow([(208, Y), (BX0, Y)])
    draw_trunk(s, film=True)
    ex, ey = draw_embedding(s)
    fs = C["film"][1]
    lx = 214
    s.arrow([(ex + 4, ey), (lx, ey)], C["emb"][1], 1.8)
    s.box(lx, ey - 26, 112, 52, "film")
    s.text(lx + 56, ey - 8, "Linear", 16, fs, 600)
    s.text(lx + 56, ey + 12, "4 → 800", 13, fs)
    s.arrow([(lx + 56, ey - 26), (lx + 56, BUS_Y)], fs, 2.2, head=False)
    draw_film_bus(s, lx + 56, "")
    s.text(lx + 72, BUS_Y - 16, "γ, β", 19, fs, 600, anchor="start", family=SERIF, italic=True)
    draw_head_output(s, NODES[-1] + 17)
    return s


def film_context() -> Svg:
    s = Svg("TCN + Context FiLM")
    draw_input(s)
    s.arrow([(208, Y), (BX0, Y)])
    draw_trunk(s, film=True)
    ts = C["ctx"][1]
    bx, by, bw, bh = BX0, 480, 70, 76
    s.dot(262, Y, ts)
    s.arrow([(262, Y), (262, by + bh / 2), (bx, by + bh / 2)], ts, 2.0)
    for i, d in enumerate((8, 16, 32)):
        x = bx + i * 100
        s.box(x, by, bw, bh, "ctx")
        s.text(x + bw / 2, by + 26, "TB", 16, ts, 600)
        s.text(x + bw / 2, by + 50, f'<tspan font-style="italic" font-family="Times New Roman">d</tspan> = {d}', 15, ts)
        s.arrow([(x + bw, by + bh / 2), (x + 100, by + bh / 2)], ts, 2.0)
    s.text(bx + 135, by + bh + 26, "context encoder  ·  32 ch", 15, MUTED)
    cx = bx + 300
    s.box(cx, by + 10, 96, bh - 20, "film")
    s.text(cx + 48, by + bh / 2, "1×1 conv", 15, C["film"][1], 600)
    jx = cx + 48
    fs = C["film"][1]
    s.arrow([(jx, by + 10), (jx, BUS_Y)], fs, 2.2, head=False)
    draw_film_bus(s, NODES[0], "")
    s.text(jx + 12, BUS_Y + 22, "γ(t), β(t)", 19, fs, 600, anchor="start", family=SERIF, italic=True)
    draw_head_output(s, NODES[-1] + 17)
    return s


def moe() -> Svg:
    s = Svg("TCN + Mixture of Experts")
    draw_input(s)
    s.arrow([(208, Y), (BX0, Y)])
    draw_trunk(s)
    sx, ex, ew, eh = TRUNK_END + 38, TRUNK_END + 80, 128, 46
    centers = [Y - 84, Y - 28, Y + 28, Y + 84]
    gy = Y + 178
    s.arrow([(TRUNK_END, Y), (sx, Y)], head=False)
    s.dot(sx, Y)
    s.arrow([(sx, centers[0]), (sx, gy)], head=False)
    es = C["exp"][1]
    sig_x, sig_r = 1240, 22
    for k, cy in enumerate(centers):
        s.arrow([(sx, cy), (ex, cy)])
        s.box(ex, cy - eh / 2, ew, eh, "exp", rx=8)
        s.text(ex + 30, cy, f'E<tspan dy="5" font-size="12">{k + 1}</tspan>', 18, es, 700, family=SERIF, italic=True)
        s.text(ex + 86, cy, "80→32→1", 13, es)
        dx, dy = sig_x - (ex + ew), Y - cy
        n = math.hypot(dx, dy)
        s.arrow([(ex + ew, cy), (sig_x - dx / n * (sig_r + 2), Y - dy / n * (sig_r + 2))], es, 1.8)
    gs = C["gate"][1]
    s.arrow([(sx, gy), (ex, gy)])
    s.box(ex, gy - 26, ew, 52, "gate", rx=8)
    s.text(ex + ew / 2, gy - 8, "Gate", 16, gs, 600)
    s.text(ex + ew / 2, gy + 12, "softmax", 13, gs)
    s.arrow([(ex + ew, gy), (sig_x, gy), (sig_x, Y + sig_r + 3)], gs, 2.0)
    s.text(sig_x + 12, Y + 120, '<tspan font-style="italic">w</tspan><tspan dy="5" font-size="12">1…4</tspan>',
           19, gs, 600, anchor="start", family=SERIF)
    s.circle(sig_x, Y, sig_r, "head", "Σ", 20)
    s.arrow([(sig_x + sig_r, Y), (OUT_X, Y)])
    draw_output(s)
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parents[1] / "runs/terrain/slides/architectures")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    figs = {"1_tcn": plain_tcn, "2_tcn_emb": tcn_emb, "3_film_subject": film_subject,
            "4_film_context": film_context, "5_moe": moe}
    for name, fn in figs.items():
        svg = fn().render()
        (a.out / f"{name}.svg").write_text(svg)
        page = a.out / f".{name}.html"
        page.write_text(f"<html><body style='margin:0'>{svg}</body></html>")
        subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                        "--force-device-scale-factor=3", f"--window-size={W},{H}",
                        f"--screenshot={a.out / f'{name}.png'}", page.as_uri()],
                       check=True, capture_output=True)
        page.unlink()
        print(a.out / f"{name}.png")


if __name__ == "__main__":
    main()
