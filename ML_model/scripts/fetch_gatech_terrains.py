"""Download GaTech (Camargo 2021) overground modes from Mendeley and write the CSV tree.

Only the members we use are pulled out of the three ~10 GB zips (HTTP range reads),
then each MATLAB table is written as CSV in the same layout as the treadmill tree:

    <out>/<subject>/<mode>/<sensor>/<trial>/data.csv
    <out>/<subject>/<mode>/conditions/<trial>/{labels.csv, trialStarts.csv, trialEnds.csv, info.json}

    python scripts/fetch_gatech_terrains.py --out ~/PycharmProjects/exo-data/csv_output
    python scripts/fetch_gatech_terrains.py --out ... --modes levelground ramp --workers 8
"""
from __future__ import annotations

import argparse
import json
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from remotezip import RemoteZip

import matio

DATASETS = ("fcgm3chfff", "k9kvm5tn3f", "jj3r5f9pnf")      # Mendeley parts 1-3, version 2
SENSORS = ("imu", "id", "gon", "gcRight", "conditions")


def zip_urls() -> list[str]:
    urls = []
    for ds in DATASETS:
        files = requests.get(f"https://data.mendeley.com/public-api/datasets/{ds}/files",
                             params={"folder_id": "root", "version": 2}, timeout=60).json()
        urls += [f["content_details"]["download_url"] for f in files if f["filename"].startswith("Subjects")]
    return urls


def write_conditions(mat: dict, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    labels = mat["labels"].copy()
    labels["Label"] = labels["Label"].map(lambda v: str(np.ravel(v)[0]) if np.size(v) else "")
    labels.to_csv(dst / "labels.csv", index=False)
    info = {}
    for k, v in mat.items():
        if k.startswith("__") or k == "labels":
            continue
        if isinstance(v, pd.DataFrame):
            v.to_csv(dst / f"{k}.csv", index=False)
            continue
        arr = np.ravel(np.asarray(v, dtype=object))
        info[k] = [x.item() if hasattr(x, "item") else str(x) for x in arr]
    for name in ("trialStarts", "trialEnds"):
        if name in info:
            pd.DataFrame([info[name]]).to_csv(dst / f"{name}.csv", header=False, index=False)
    (dst / "info.json").write_text(json.dumps(info, indent=1, default=str))


_local = threading.local()


def convert(url: str, member: str, out: Path, tmp: Path) -> str:
    subject, _date, mode, sensor, fname = member.split("/")
    trial = fname[:-4]
    dst = out / subject / mode / sensor / trial
    done = dst / ("labels.csv" if sensor == "conditions" else "data.csv")
    if done.exists():
        return f"skip {member}"
    if getattr(_local, "url", None) != url:            # one zip handle per worker thread
        _local.zip, _local.url = RemoteZip(url), url
    local = Path(_local.zip.extract(member, tmp))
    mat = matio.load_from_mat(str(local))
    if sensor == "conditions":
        write_conditions(mat, dst)
    else:
        dst.mkdir(parents=True, exist_ok=True)
        mat["data"].to_csv(dst / "data.csv.part", index=False)
        (dst / "data.csv.part").rename(done)
    local.unlink()
    return member


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--modes", nargs="+", default=["levelground", "ramp"])
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    out = a.out.expanduser()

    for url in zip_urls():
        with RemoteZip(url) as probe:
            members = [m for m in probe.namelist()
                       if m.endswith(".mat") and len(m.split("/")) == 5
                       and m.split("/")[2] in a.modes and m.split("/")[3] in SENSORS]
        print(f"{len(members)} members from {url.split('/')[-1][:12]}", flush=True)
        with tempfile.TemporaryDirectory() as tmp, ThreadPoolExecutor(a.workers) as pool:
            futs = [pool.submit(convert, url, m, out, Path(tmp)) for m in members]
            for n, f in enumerate(as_completed(futs), 1):
                try:
                    r = f.result()
                except Exception as e:                       # keep going; re-run resumes
                    r = f"FAILED {e!r}"
                if n % 50 == 0 or r.startswith("FAILED"):
                    print(f"  {n}/{len(members)} {r}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
