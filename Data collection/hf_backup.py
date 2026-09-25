#!/usr/bin/env python3
"""
hf_backup.py

Back up recorded sensor data to the Hugging Face dataset repo.

Watched folders (paths are relative to the repository root and kept the same
on Hugging Face):
- Data collection/data/   CSVs from data_collection.py
- calibration/            recordings, plots and results from `exo_frame.py calibrate`

Runs as its own process, separate from the recorders, so WiFi problems never
affect data collection. Local files are never modified or deleted. A file is
uploaded once it has not changed for SETTLE_SECONDS (so a recording is not
uploaded while it is still being written); anything that fails to upload is
retried on the next pass.

One-time setup on the Jetson:
    pip install huggingface_hub
    hf auth login                     # token with write access

Run:
    python3 "Data collection/hf_backup.py"             # keep syncing every 60 s
    python3 "Data collection/hf_backup.py" --once      # single pass, then exit
    python3 "Data collection/hf_backup.py" --dry-run   # list what would be uploaded
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


# ========================= User Config =========================
# Hugging Face dataset repo the data is backed up to
DEFAULT_REPO_ID = "Amilyl/MRSD_Axilles_exo"

REPO_ROOT = Path(__file__).resolve().parent.parent

# Folders to back up
DEFAULT_SOURCES = [
    REPO_ROOT / "Data collection" / "data",
    REPO_ROOT / "calibration",
]

# Folder names that are never uploaded (test artefacts)
SKIP_DIRS = {"_test_output", "__pycache__"}

# File types that are never uploaded (code, not data)
SKIP_SUFFIXES = {".py", ".pyc", ".tmp"}

# A file must be unchanged for this long before it is uploaded (seconds)
SETTLE_SECONDS = 30.0

# Seconds between upload passes, and the longest wait between retries
INTERVAL_S = 60.0
MAX_BACKOFF_S = 300.0

# Record of what has already been uploaded (ignored by git)
MANIFEST_PATH = REPO_ROOT / "Data collection" / ".hf_uploaded.json"

# Max files per commit
BATCH_SIZE = 100
# ==============================================================


def log(msg: str) -> None:
    print(f"[hf_backup][{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_manifest(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def save_manifest(path: Path, manifest: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    os.replace(tmp, path)


def find_pending(sources: list[Path], manifest: dict) -> list[tuple[Path, str, list]]:
    """Return [(local_path, path_in_repo, signature)] for settled files that are new or changed."""
    pending = []
    now = time.time()
    for src in sources:
        if not src.is_dir():
            continue
        for root, dirs, files in os.walk(src):
            dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.startswith("."))
            for name in sorted(files):
                if name.startswith(".") or Path(name).suffix in SKIP_SUFFIXES:
                    continue
                local = Path(root) / name
                try:
                    st = local.stat()
                except OSError:
                    continue
                if now - st.st_mtime < SETTLE_SECONDS:
                    continue  # still being written
                try:
                    path_in_repo = local.relative_to(REPO_ROOT).as_posix()
                except ValueError:
                    path_in_repo = local.relative_to(src.parent).as_posix()
                signature = [st.st_size, st.st_mtime]
                if manifest.get(path_in_repo) != signature:
                    pending.append((local, path_in_repo, signature))
    return pending


def upload_pass(api, repo_id: str, sources: list[Path], manifest_path: Path, dry_run: bool) -> int:
    manifest = load_manifest(manifest_path)
    pending = find_pending(sources, manifest)
    if not pending:
        return 0

    if dry_run:
        for _, path_in_repo, _ in pending:
            log(f"would upload {path_in_repo}")
        return len(pending)

    from huggingface_hub import CommitOperationAdd  # type: ignore[import-not-found]

    uploaded = 0
    for i in range(0, len(pending), BATCH_SIZE):
        batch = pending[i:i + BATCH_SIZE]
        api.create_commit(
            repo_id=repo_id,
            repo_type="dataset",
            operations=[CommitOperationAdd(path_in_repo=p, path_or_fileobj=str(local)) for local, p, _ in batch],
            commit_message=f"Backup {len(batch)} file(s) from the Jetson",
        )
        # Only mark as uploaded after the commit succeeded
        for _, path_in_repo, signature in batch:
            manifest[path_in_repo] = signature
        save_manifest(manifest_path, manifest)
        uploaded += len(batch)
        names = ", ".join(p for _, p, _ in batch[:5]) + (" ..." if len(batch) > 5 else "")
        log(f"uploaded {len(batch)} file(s): {names}")
    return uploaded


def main() -> None:
    ap = argparse.ArgumentParser(description="Back up recorded sensor data to Hugging Face")
    ap.add_argument("--repo-id", default=os.environ.get("HF_REPO_ID", DEFAULT_REPO_ID),
                    help=f"dataset repo (default {DEFAULT_REPO_ID}, or set HF_REPO_ID)")
    ap.add_argument("--dir", action="append", type=Path, default=None,
                    help="folder to back up (repeatable); default: 'Data collection/data' and 'calibration'")
    ap.add_argument("--interval", type=float, default=INTERVAL_S, help="seconds between upload passes")
    ap.add_argument("--once", action="store_true", help="run a single pass and exit")
    ap.add_argument("--dry-run", action="store_true", help="list files that would be uploaded, upload nothing")
    args = ap.parse_args()

    sources = [p.resolve() for p in args.dir] if args.dir else DEFAULT_SOURCES

    api = None
    if not args.dry_run:
        try:
            from huggingface_hub import HfApi  # type: ignore[import-not-found]
        except ImportError:
            sys.exit("huggingface_hub is not installed. Run: pip install huggingface_hub")
        api = HfApi()

    log("watching " + ", ".join(str(p.relative_to(REPO_ROOT)) if p.is_relative_to(REPO_ROOT) else str(p)
                                for p in sources)
        + ("" if args.dry_run else f" -> https://huggingface.co/datasets/{args.repo_id}"))

    repo_ready = args.dry_run
    backoff = args.interval
    while True:
        try:
            if not repo_ready:
                api.create_repo(args.repo_id, repo_type="dataset", exist_ok=True)
                repo_ready = True
            if upload_pass(api, args.repo_id, sources, MANIFEST_PATH, args.dry_run) == 0 and args.once:
                log("nothing new to upload.")
            backoff = args.interval
        except KeyboardInterrupt:
            log("stopped.")
            return
        except Exception as exc:
            # No network, HF outage, auth problem... data stays local and is retried later
            log(f"upload failed ({type(exc).__name__}: {exc})")
            if "401" in str(exc) or "403" in str(exc):
                log("not authorized: run 'hf auth login' with a write token, and check the repo id")
            if args.once:
                sys.exit(1)
            log(f"retrying in {backoff:.0f} s")
            try:
                time.sleep(backoff)
            except KeyboardInterrupt:
                log("stopped.")
                return
            backoff = min(backoff * 2, MAX_BACKOFF_S)
            continue

        if args.once:
            return
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            log("stopped.")
            return


if __name__ == "__main__":
    main()
