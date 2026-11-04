"""Terrain-experiment results as a booktabs LaTeX table plus a PNG of the same table.

    python analysis/results_table.py runs/terrain_stairs/<stamp> [--out runs/terrain/slides]
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
MODELS = [("tcn", "Plain TCN"), ("tcn_emb", "TCN + Subject Embedding"),
          ("film", "TCN + FiLM (Subject Embedding)"), ("moe", "TCN + Mixture of Experts")]
MODES = [("treadmill", "Treadmill"), ("walk", "Level"), ("turn", "Turn"),
         ("rampascent", "Ramp ↑"), ("rampdescent", "Ramp ↓"),
         ("stairascent", "Stair ↑"), ("stairdescent", "Stair ↓"), ("transition", "Transition")]


def rows(summary: dict) -> list[dict]:
    global MODES
    MODES = [(m, lbl) for m, lbl in MODES if m in summary["tcn"]["nearest"]]
    out = []
    for key, label in MODELS:
        r = summary[key]["nearest"]
        out.append({"label": label, "overall": r["overall"]["r2"], "rmse": r["overall"]["rmse"],
                    **{m: r[m]["r2"] for m, _ in MODES}, "params": summary[key]["params"]})
    return out


def best(data, col, low=False):
    vals = [d[col] for d in data]
    return min(vals) if low else max(vals)


def latex(data) -> str:
    cols = ["overall"] + [m for m, _ in MODES]
    tex_modes = [lbl.replace("↑", r"$\uparrow$").replace("↓", r"$\downarrow$") for _, lbl in MODES]
    lines = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{l c " + "c" * len(MODES) + " c r}", r"\toprule",
             r" & & \multicolumn{" + str(len(MODES)) + r"}{c}{$R^2$ by locomotion mode} & & \\",
             r"\cmidrule(lr){3-" + str(len(MODES) + 2) + "}",
             r"Model & $R^2$ & " + " & ".join(tex_modes) + r" & RMSE & Params \\", r"\midrule"]
    for d in data:
        cells = []
        for c in cols:
            v = f"{d[c]:.3f}"
            cells.append(rf"\textbf{{{v}}}" if abs(d[c] - best(data, c)) < 5e-4 else v)
        rm = f"{d['rmse']:.3f}"
        rm = rf"\textbf{{{rm}}}" if abs(d["rmse"] - best(data, "rmse", True)) < 5e-4 else rm
        lines.append(f"{d['label']} & " + " & ".join(cells) + f" & {rm} & {d['params'] / 1e3:.0f}k \\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Ankle moment estimation on six held-out subjects (treadmill, level ground, "
              + ("ramps and stairs" if any(m.startswith("stair") for m, _ in MODES) else "ramps")
              + r"). $R^2$ of the causal last-timestep prediction; RMSE in N$\cdot$m/kg. "
              r"Best per column in bold.}",
              r"\label{tab:terrain_results}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def html(data) -> str:
    cols = ["overall"] + [m for m, _ in MODES]
    th = "".join(f"<th>{lbl}</th>" for _, lbl in MODES)
    body = ""
    for d in data:
        tds = "".join(f"<td class='{'b' if abs(d[c] - best(data, c)) < 5e-4 else ''}'>{d[c]:.3f}</td>" for c in cols)
        rb = "b" if abs(d["rmse"] - best(data, "rmse", True)) < 5e-4 else ""
        body += (f"<tr><td class='l'>{d['label']}</td>{tds}<td class='{rb} sep'>{d['rmse']:.3f}</td>"
                 f"<td>{d['params'] / 1e3:.0f}k</td></tr>")
    n = len(MODES)
    return f"""<html><head><style>
body {{ margin:0; background:#fff; font-family:'Times New Roman',Times,serif; color:#111; }}
.wrap {{ display:inline-block; padding:28px 34px; }}
table {{ border-collapse:collapse; font-size:19px; }}
th, td {{ padding:7px 11px; text-align:center; white-space:nowrap; }}
th {{ font-weight:600; }}
td.l, th.l {{ text-align:left; padding-right:22px; }}
td.b {{ font-weight:700; }}
.top {{ border-top:2px solid #111; }} .bot {{ border-bottom:2px solid #111; }}
.mid td {{ border-top:1px solid #111; }}
.span {{ border-bottom:1px solid #111; font-weight:500; font-style:italic; }}
.sep {{ padding-left:18px; }}
</style></head><body><div class="wrap" id="t"><table>
<tr class="top"><th></th><th></th><th colspan="{n}" class="span"><i>R</i><sup>2</sup> by locomotion mode</th><th></th><th></th></tr>
<tr><th class="l">Model</th><th>Overall <i>R</i><sup>2</sup></th>{th}<th class="sep">RMSE</th><th>Params</th></tr>
{body.replace('<tr>', '<tr class="mid">', 1)}
<tr class="bot"><td colspan="{n + 4}" style="padding:0"></td></tr>
</table></div></body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--name", default="results_table")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "runs/terrain/slides")
    a = ap.parse_args()
    data = rows(json.loads((a.run / "summary.json").read_text()))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"{a.name}.tex").write_text(latex(data))
    page = a.out / f".{a.name}.html"
    page.write_text(html(data))
    png = a.out / f"{a.name}.png"
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                    "--force-device-scale-factor=3", "--window-size=1500,300",
                    f"--screenshot={png}", page.as_uri()], check=True, capture_output=True)
    page.unlink()
    from PIL import Image, ImageChops
    im = Image.open(png).convert("RGB")
    box = ImageChops.difference(im, Image.new("RGB", im.size, "white")).getbbox()
    im.crop((max(box[0] - 60, 0), max(box[1] - 60, 0), box[2] + 60, box[3] + 60)).save(png)
    print(a.out / f"{a.name}.tex"); print(png)


if __name__ == "__main__":
    main()
