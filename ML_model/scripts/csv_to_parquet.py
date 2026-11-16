"""Convert the raw GaTech CSV tree into the Hugging Face Parquet layout.

Output:
    <out>/metadata.parquet
    <out>/subjects/<subject>/<trial>__<sensor>.parquet

Only the sensors used by this project are kept (imu, id, fp, gon, gcRight) plus a
per-trial conditions summary folded into metadata. Overground trials also get
``<trial>__labels.parquet``, the per-sample locomotion mode (walk, rampascent, ...).
Signals stay at native rate.

    python scripts/csv_to_parquet.py --src <csv_output> --out <parquet_dir>
    python scripts/csv_to_parquet.py --src ... --out ... --terrains treadmill levelground ramp
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

_SIGNAL_SENSORS = ("imu", "id", "fp", "gon", "gcRight")


def _read_conditions(trial_dir: Path) -> dict:
    out: dict = {}
    for name in ("speed", "trialStarts", "trialEnds"):
        path = trial_dir / f"{name}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, header=None if name != "speed" else 0)
        if name == "speed":
            out["speed_mean_mps"] = float(pd.to_numeric(df["Speed"], errors="coerce").dropna().mean())
        else:
            out[f"{name.lower()}_s"] = float(df.iloc[0, -1])
    return out


def convert(src: Path, out: Path, terrains: list[str]) -> None:
    subjects_out = out / "subjects"
    subjects_out.mkdir(parents=True, exist_ok=True)

    demo = pd.read_csv(src / "subject_info.csv")
    demo["Subject"] = demo["Subject"].str.strip()
    demo_by_id = demo.set_index("Subject").to_dict("index")

    rows: list[dict] = []
    subjects = sorted(d.name for d in src.iterdir() if d.is_dir() and d.name.startswith("AB"))

    for subject, terrain in ((s, t) for s in subjects for t in terrains):
        imu_root = src / subject / terrain / "imu"
        if not imu_root.is_dir():
            continue
        (subjects_out / subject).mkdir(exist_ok=True)

        for trial_dir in sorted(p for p in imu_root.iterdir() if p.is_dir()):
            trial = trial_dir.name
            n_samples = 0
            for sensor in _SIGNAL_SENSORS:
                csv = src / subject / terrain / sensor / trial / "data.csv"
                if not csv.exists():
                    continue
                df = pd.read_csv(csv).rename(columns={"Header": "time_s"})
                df = df.astype("float32", errors="ignore")
                df.to_parquet(subjects_out / subject / f"{trial}__{sensor}.parquet",
                              index=False, compression="zstd")
                if sensor == "imu":
                    n_samples = len(df)

            cond_dir = src / subject / terrain / "conditions" / trial
            labels = cond_dir / "labels.csv"
            if labels.exists():
                pd.read_csv(labels).rename(columns={"Header": "time_s"}).to_parquet(
                    subjects_out / subject / f"{trial}__labels.parquet", index=False,
                    compression="zstd")
            cond = _read_conditions(cond_dir)
            d = demo_by_id.get(subject, {})
            rows.append({
                "subject": subject, "trial": trial, "terrain": terrain,
                "n_imu_samples": n_samples,
                "age": d.get("Age"), "gender": d.get("Gender"),
                "height_m": d.get("Height"), "weight_kg": d.get("Weight"),
                **cond,
            })
            print(f"  {subject}/{trial}")

    meta = pd.DataFrame(rows)
    meta.to_parquet(out / "metadata.parquet", index=False)
    print(f"\n{len(meta)} trials, {meta.subject.nunique()} subjects -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="raw GaTech csv_output/ root")
    ap.add_argument("--out", required=True, help="destination Parquet directory")
    ap.add_argument("--terrains", nargs="+", default=["treadmill"],
                    help="mode folders to convert (treadmill levelground ramp stair)")
    args = ap.parse_args()
    convert(Path(args.src).expanduser(), Path(args.out).expanduser(), args.terrains)


if __name__ == "__main__":
    main()
