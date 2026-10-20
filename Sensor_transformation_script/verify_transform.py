"""Score a fitted GaTech -> exo transform with the waveform_metrics package.

For every subject, segment, sensor (accel/gyro) and axis, the GaTech mean stride is
compared with the exo mean stride twice: in GaTech's own frame ("before") and after
applying that subject's fitted dR, dp ("after").

    python3 verify_transform.py \
      --gatech-root data/mrsd-exo-ankle --exo-log data/<log>.csv \
      --transforms data/transforms.json --out-dir data/verification
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import align_io as aio
from run_alignment import SEGMENTS, exo_templates, subject_templates

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "ML_model" / "analysis"))
from waveform_metrics import compute_all

METRICS = ("rmse", "nrmse", "pearson_r", "r2", "peak_timing_error_pct", "dtw_distance")


def subject_strides(subject: str, gatech_tpl: dict, exo_tpl: dict, fits: dict) -> dict:
    """{segment: {sensor: {"exo" | "before" | "after": (n_phase, 3) array}}}"""
    out = {}
    for seg in SEGMENTS:
        fit = fits.get(seg, {}).get(subject)
        if fit is None or seg not in gatech_tpl or seg not in exo_tpl:
            continue
        target = exo_tpl[seg].shifted(fit["shift"])
        src = gatech_tpl[seg]
        tx_accel, tx_gyro = src.transformed(np.asarray(fit["dR"]), np.asarray(fit["dp"]))
        out[seg] = {
            "accel": {"exo": target.A, "before": src.A, "after": tx_accel},
            "gyro": {"exo": target.G, "before": src.G, "after": tx_gyro},
        }
    return out


def score(subject: str, strides: dict) -> list[dict]:
    rows = []
    for seg, sensors in strides.items():
        for sensor, s in sensors.items():
            for stage in ("before", "after"):
                for i, axis in enumerate("XYZ"):
                    m = compute_all(s["exo"][:, i], s[stage][:, i])
                    rows.append({"subject": subject, "segment": seg, "sensor": sensor,
                                 "axis": axis, "stage": stage,
                                 **{k: getattr(m, k) for k in METRICS}})
    return rows


def _to_lists(strides: dict) -> dict:
    return {seg: {sensor: {k: np.round(v, 4).tolist() for k, v in s.items()}
                  for sensor, s in sensors.items()}
            for seg, sensors in strides.items()}


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    per_group = (df.groupby(["segment", "sensor", "stage", "subject"])[list(METRICS)].mean()
                 .groupby(["segment", "sensor", "stage"]).median())
    wide = per_group.unstack("stage")
    wide.columns = [f"{metric}_{stage}" for metric, stage in wide.columns]
    wide["rmse_change_%"] = 100 * (wide["rmse_after"] - wide["rmse_before"]) / wide["rmse_before"]
    order = [f"{m}_{s}" for m in METRICS for s in ("before", "after")]
    return wide[order[:2] + ["rmse_change_%"] + order[2:]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gatech-root", required=True)
    ap.add_argument("--exo-log", required=True)
    ap.add_argument("--transforms", required=True)
    ap.add_argument("--out-dir", default=None,
                    help="default: ML_model/runs/transform_verification/<transforms stem>")
    a = ap.parse_args()

    fits_json = json.loads(Path(a.transforms).read_text())
    fits = {seg: fits_json[seg]["per_subject"] for seg in SEGMENTS if seg in fits_json}
    subjects = sorted(set().union(*(f.keys() for f in fits.values())))

    exo_tpl, _ = exo_templates(aio.load_exo(a.exo_log, verbose=False), SEGMENTS)
    meta = pd.read_parquet(Path(a.gatech_root) / "metadata.parquet")
    trials = meta.groupby("subject")["trial"].apply(list).to_dict()

    rows, templates = [], {}
    for subject in subjects:
        gatech_tpl, _ = subject_templates(a.gatech_root, subject, trials[subject], SEGMENTS)
        strides = subject_strides(subject, gatech_tpl, exo_tpl, fits)
        rows.extend(score(subject, strides))
        templates[subject] = _to_lists(strides)

    out = Path(a.out_dir) if a.out_dir else (
        REPO_ROOT / "ML_model" / "runs" / "transform_verification" / Path(a.transforms).stem)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "per_axis_metrics.csv", index=False)
    summary = summarize(df)
    summary.to_csv(out / "summary.csv")
    (out / "templates.json").write_text(json.dumps(templates))
    (out / "meta.json").write_text(json.dumps({
        "exo_log": Path(a.exo_log).name,
        "transforms": str(Path(a.transforms).resolve()),
        "gatech_root": str(Path(a.gatech_root).resolve()),
        "subjects": subjects,
        "outliers": {seg: fits_json[seg].get("outliers", []) for seg in fits},
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
    }, indent=2))
    pd.set_option("display.width", 200)
    print(summary.round(3).to_string())
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
