"""Fetch the dataset and build the processed cache — one command.

    python scripts/prepare.py                 # fetch (if needed) + ingest
    python scripts/prepare.py --force          # rebuild the cache
    python scripts/prepare.py --no-fetch       # use the local dataset as-is

``scripts/train.py`` calls this automatically when the cache is cold.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from exo.config import Config
from exo.data import ingest
from exo.data.hub import HubConfig, fetch


def prepare(cfg: Config, do_fetch: bool = True, force: bool = False,
            subjects: list[str] | None = None, verbose: bool = True) -> Path:
    dataset_root = cfg.paths.dataset()
    revision = cfg.paths.hub.revision

    if do_fetch and cfg.paths.hub.repo_id:
        if verbose:
            print(f"fetching {cfg.paths.hub.repo_id}@{revision} -> {dataset_root}")
        fetch(HubConfig(repo_id=cfg.paths.hub.repo_id, revision=revision), dataset_root)

    if not (dataset_root / "metadata.parquet").exists():
        raise SystemExit(
            f"no dataset at {dataset_root}. Set paths.hub.repo_id and run without "
            f"--no-fetch, or point paths.dataset_dir at a local Parquet dataset."
        )

    processed = cfg.paths.processed()
    if not force and ingest.is_cached(processed):
        if verbose:
            print(f"processed cache is up to date: {processed}")
        return processed

    return ingest.run(cfg, dataset_root, processed, revision, subjects=subjects, verbose=verbose)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--no-fetch", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--subjects", nargs="*", default=None)
    args = ap.parse_args()

    cfg = Config.load(args.config)
    prepare(cfg, do_fetch=not args.no_fetch, force=args.force, subjects=args.subjects)


if __name__ == "__main__":
    main()
