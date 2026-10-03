"""Build the processed .npz cache from a local Parquet dataset (no fetch).

Thin wrapper around ``exo.data.ingest.run``. For the fetch + ingest flow use
``scripts/prepare.py``.

    python scripts/ingest_raw.py
    python scripts/ingest_raw.py --subjects AB06 AB09
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from exo.config import Config
from exo.data import ingest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    dataset_root = cfg.paths.dataset()
    if not (dataset_root / "metadata.parquet").exists():
        raise SystemExit(f"no dataset at {dataset_root}; run scripts/prepare.py")

    processed = cfg.paths.processed()
    if not args.force and ingest.is_cached(processed) and args.subjects is None:
        print(f"cache up to date: {processed}")
        return
    ingest.run(cfg, dataset_root, processed, cfg.paths.hub.revision, subjects=args.subjects)


if __name__ == "__main__":
    main()
