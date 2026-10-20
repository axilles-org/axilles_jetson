"""Process many Camargo subjects: wait for each download, unzip, analyse, compare.

For every subject zip in --zips-dir (e.g. ~/Downloads/AB11.zip):
  1. wait until the browser has finished it (no matching *.zip.part, zip valid)
  2. unzip into the project folder (-> ./AB11/...)
  3. scripts/locate_camargo.py and scripts/write_placement_table.py
Finally scripts/compare_subjects.py over every subject that has results.

  python scripts/batch_camargo.py --zips-dir ~/Downloads --parallel 4
"""
import argparse
import re
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def zip_ready(z: Path) -> bool:
    if any(z.parent.glob(f"{z.stem}.*.zip.part")) or not z.exists() or z.stat().st_size == 0:
        return False
    try:
        with zipfile.ZipFile(z) as f:
            return f.testzip() is None
    except zipfile.BadZipFile:
        return False


def extract(z: Path, subject: str) -> Path:
    dest = ROOT / subject
    if dest.exists() and any(dest.glob("osimxml/*.osim")):
        return dest
    with zipfile.ZipFile(z) as f:
        names = f.namelist()
        top = {n.split("/")[0] for n in names if n.strip("/")}
        f.extractall(ROOT if top == {subject} else dest)
    if not any(dest.glob("osimxml/*.osim")):
        raise RuntimeError(f"{subject}: no osimxml/*.osim after unzip")
    return dest


def run(cmd, log):
    with open(log, "a") as fh:
        fh.write(f"\n$ {' '.join(map(str, cmd))}\n")
        fh.flush()
        return subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT).returncode


def process(subject, z, logdir, max_trials, poll, timeout):
    log = logdir / f"{subject}.log"
    t0 = time.time()
    while not zip_ready(z):
        if time.time() - t0 > timeout:
            print(f"[{subject}] download not finished after {timeout / 3600:.1f} h, skipped", flush=True)
            return subject, "timeout"
        time.sleep(poll)
    print(f"[{subject}] download complete, extracting", flush=True)
    try:
        extract(z, subject)
    except Exception as e:  # noqa: BLE001
        print(f"[{subject}] extract failed: {e}", flush=True)
        return subject, "extract-failed"
    out = ROOT / "results" / subject
    rc = run([PY, "scripts/locate_camargo.py", "--subject-dir", subject, "--max-trials", str(max_trials),
              "--out", str(out)], log)
    if rc == 0:
        rc = run([PY, "scripts/write_placement_table.py", "--results", str(out), "--subject", subject], log)
    status = "ok" if rc == 0 else f"failed (see {log})"
    print(f"[{subject}] {status}", flush=True)
    return subject, status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zips-dir", default=str(Path.home() / "Downloads"))
    ap.add_argument("--subjects", nargs="*", help="default: every AB*.zip in --zips-dir")
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--max-trials", type=int, default=6)
    ap.add_argument("--poll", type=float, default=60.0)
    ap.add_argument("--timeout-h", type=float, default=6.0)
    a = ap.parse_args()

    zdir = Path(a.zips_dir).expanduser()
    subjects = a.subjects or sorted({m.group(1) for p in zdir.glob("AB*.zip")
                                     if (m := re.fullmatch(r"(AB\d+)\.zip", p.name))})
    logdir = ROOT / "results" / "batch_logs"
    logdir.mkdir(parents=True, exist_ok=True)
    print(f"subjects: {' '.join(subjects)}", flush=True)
    with ThreadPoolExecutor(a.parallel) as ex:
        results = list(ex.map(lambda s: process(s, zdir / f"{s}.zip", logdir, a.max_trials,
                                                a.poll, a.timeout_h * 3600), subjects))

    done = sorted(p.parent.name for p in (ROOT / "results").glob("AB*/summary.json")
                  if re.fullmatch(r"AB\d+", p.parent.name))
    print(f"\ncomparing {len(done)} subjects: {' '.join(done)}", flush=True)
    run([PY, "scripts/compare_subjects.py", "--subjects", *done], logdir / "compare.log")
    print("\n".join(f"  {s}: {st}" for s, st in results), flush=True)


if __name__ == "__main__":
    main()
