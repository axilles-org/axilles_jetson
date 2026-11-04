"""Booktabs LaTeX + PNG tables of the embedding ablation (run embedding_ablation.py first).

    python analysis/embedding_ablation_table.py runs/terrain/<stamp> [--name embedding_ablation]
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
MODELS = [("tcn_emb", "TCN + Subject Embedding"), ("film", "TCN + FiLM (Subject Embedding)")]
MODES = [("overall", "Overall"), ("treadmill", "Treadmill"), ("walk", "Level"), ("turn", "Turn"),
         ("rampascent", "Ramp ↑"), ("rampdescent", "Ramp ↓"), ("stairascent", "Stair ↑"),
         ("stairdescent", "Stair ↓"), ("transition", "Transition")]
ROWS = [("nearest", "Nearest training subject (deployed)"), ("mean", "Average embedding"),
        ("best", "Single training embedding: best"), ("median", "Single training embedding: median"),
        ("worst", "Single training embedding: worst")]


def tex_label(s: str) -> str:
    return s.replace("↑", r"$\uparrow$").replace("↓", r"$\downarrow$")


def stats(ab: dict, cols: list[str]) -> dict:
    train = [v for k, v in ab.items() if k.startswith("train")]
    out = {"nearest": {c: ab["nearest"][c] for c in cols}, "mean": {c: ab["mean"][c] for c in cols}}
    for name, f in (("best", np.max), ("median", np.median), ("worst", np.min)):
        out[name] = {c: float(f([t[c] for t in train])) for c in cols}
    return out


def subject_stats(ab: dict) -> dict:
    train = [v for k, v in ab.items() if k.startswith("train")]
    subs = sorted(ab["nearest"]["per_subject"])
    return {z: {"nearest": ab["nearest"]["per_subject"][z], "mean": ab["mean"]["per_subject"][z],
                "best": max(t["per_subject"][z] for t in train),
                "worst": min(t["per_subject"][z] for t in train)} for z in subs}


def latex_modes(cols, labels, tcn, blocks) -> str:
    n = len(cols)
    L = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{l " + "c" * n + "}", r"\toprule",
         r" & \multicolumn{" + str(n) + r"}{c}{$R^2$ on held-out subjects} \\",
         r"\cmidrule(lr){2-" + str(n + 1) + "}",
         r"Embedding given to the test subject & " + " & ".join(tex_label(l) for l in labels) + r" \\",
         r"\midrule",
         r"Plain TCN (no embedding) & " + " & ".join(f"{tcn[c]:.3f}" for c in cols) + r" \\"]
    for title, st in blocks:
        L += [r"\midrule", r"\multicolumn{" + str(n + 1) + r"}{l}{\textit{" + title + r"}} \\"]
        for key, lab in ROWS:
            L.append(r"\quad " + lab + " & " + " & ".join(f"{st[key][c]:.3f}" for c in cols) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Embedding ablation. The trained model is scored on the six held-out subjects "
          r"with the nearest training subject's embedding (by demographics, as deployed), the average "
          r"of all training embeddings, and each of the 13 training subjects' embeddings in turn "
          r"(best, median and worst taken per column).}",
          r"\label{tab:embedding_ablation}", r"\end{table}"]
    return "\n".join(L) + "\n"


def latex_subjects(subj: dict) -> str:
    keys = [("nearest", "Nearest"), ("mean", "Average"), ("best", "Best"), ("worst", "Worst")]
    L = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{5pt}",
         r"\begin{tabular}{l cccc cccc}", r"\toprule",
         r" & \multicolumn{4}{c}{" + MODELS[0][1].replace("+", r"+") + r"} & \multicolumn{4}{c}{"
         + MODELS[1][1] + r"} \\", r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
         r"Test subject & " + " & ".join(l for _, l in keys) + " & " + " & ".join(l for _, l in keys) + r" \\",
         r"\midrule"]
    for z in subj[MODELS[0][0]]:
        cells = [f"{subj[m][z][k]:.3f}" for m, _ in MODELS for k, _ in keys]
        L.append(f"{z} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Per-subject $R^2$ with the nearest training subject's embedding, the average "
          r"embedding, and the best and worst of the 13 training subjects' embeddings.}",
          r"\label{tab:embedding_ablation_subjects}", r"\end{table}"]
    return "\n".join(L) + "\n"


CSS = """body{margin:0;background:#fff;font-family:'Times New Roman',Times,serif;color:#111}
.wrap{display:inline-block;padding:28px 34px}table{border-collapse:collapse;font-size:19px}
th,td{padding:6px 12px;text-align:center;white-space:nowrap}th{font-weight:600}
.l{text-align:left;padding-right:24px}.top{border-top:2px solid #111}.bot td{border-top:2px solid #111;padding:0}
.rule td,.rule th{border-top:1px solid #111}.span{border-bottom:1px solid #111;font-style:italic;font-weight:500}
.grp td{font-style:italic;text-align:left;padding-top:12px}.ind{padding-left:30px}.dep{font-weight:700}
.diag td{color:#555}.gap{padding-left:26px}"""


def html_modes(cols, labels, tcn, blocks) -> str:
    n = len(cols)
    h = (f"<tr class='top'><th></th><th colspan='{n}' class='span'><i>R</i><sup>2</sup> on held-out subjects</th></tr>"
         f"<tr><th class='l'>Embedding given to the test subject</th>" + "".join(f"<th>{l}</th>" for l in labels) + "</tr>"
         f"<tr class='rule'><td class='l'>Plain TCN (no embedding)</td>" + "".join(f"<td>{tcn[c]:.3f}</td>" for c in cols) + "</tr>")
    for title, st in blocks:
        h += f"<tr class='grp rule'><td colspan='{n + 1}'>{title}</td></tr>"
        for key, lab in ROWS:
            cls = "dep" if key == "nearest" else ""
            row = "diag" if key in ("best", "median", "worst") else ""
            h += (f"<tr class='{row}'><td class='l ind {cls}'>{lab}</td>"
                  + "".join(f"<td class='{cls}'>{st[key][c]:.3f}</td>" for c in cols) + "</tr>")
    h += f"<tr class='bot'><td colspan='{n + 1}'></td></tr>"
    return f"<html><head><style>{CSS}</style></head><body><div class='wrap'><table>{h}</table></div></body></html>"


def html_subjects(subj) -> str:
    keys = [("nearest", "Nearest"), ("mean", "Average"), ("best", "Best"), ("worst", "Worst")]
    h = ("<tr class='top'><th></th>" + "".join(f"<th colspan='4' class='span {'gap' if i else ''}'>{t}</th>"
                                                for i, (_, t) in enumerate(MODELS)) + "</tr>"
         "<tr><th class='l'>Test subject</th>" + "".join(
             f"<th class='{'gap' if i == 4 else ''}'>{l}</th>" for i, (_, l) in enumerate(keys * 2)) + "</tr>")
    first = True
    for z in subj[MODELS[0][0]]:
        cells = [(subj[m][z][k], k) for m, _ in MODELS for k, _ in keys]
        h += (f"<tr class='{'rule' if first else ''}'><td class='l'>{z}</td>" + "".join(
            f"<td class='{'gap' if i == 4 else ''}'>{v:.3f}</td>".replace(">-", ">\u2212")
            for i, (v, k) in enumerate(cells)) + "</tr>")
        first = False
    h += "<tr class='bot'><td colspan='9'></td></tr>"
    return f"<html><head><style>{CSS}</style></head><body><div class='wrap'><table>{h}</table></div></body></html>"


def render(page_html: str, png: Path, width: int) -> None:
    page = png.with_name(f".{png.stem}.html")
    page.write_text(page_html)
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=3",
                    f"--window-size={width},700", f"--screenshot={png}", page.as_uri()], check=True, capture_output=True)
    page.unlink()
    from PIL import Image, ImageChops
    im = Image.open(png).convert("RGB")
    box = ImageChops.difference(im, Image.new("RGB", im.size, "white")).getbbox()
    im.crop((max(box[0] - 60, 0), max(box[1] - 60, 0), box[2] + 60, box[3] + 60)).save(png)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--name", default="embedding_ablation")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "runs/terrain/slides")
    a = ap.parse_args()
    summary = json.loads((a.run / "summary.json").read_text())
    abl = {m: json.loads((a.run / f"embedding_ablation_{m}.json").read_text()) for m, _ in MODELS}
    cols = [c for c, _ in MODES if c in abl["tcn_emb"]["nearest"]]
    labels = [l for c, l in MODES if c in cols]
    tcn = {c: summary["tcn"]["nearest"][c]["r2"] for c in cols}
    blocks = [(t, stats(abl[m], cols)) for m, t in MODELS]
    subj = {m: subject_stats(abl[m]) for m, _ in MODELS}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"{a.name}.tex").write_text(latex_modes(cols, labels, tcn, blocks) + "\n" + (latex_subjects(subj)).replace("& -", "& $-$"))
    render(html_modes(cols, labels, tcn, blocks), a.out / f"{a.name}_by_mode.png", 1500)
    render(html_subjects(subj), a.out / f"{a.name}_by_subject.png", 1300)
    print(a.out / f"{a.name}.tex")


if __name__ == "__main__":
    main()
