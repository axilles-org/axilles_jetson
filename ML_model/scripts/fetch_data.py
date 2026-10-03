"""Download the dataset snapshot from Hugging Face.

    python scripts/fetch_data.py
    python scripts/fetch_data.py --subjects AB06 AB09
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from exo.config import Config
from exo.data.hub import HubConfig, fetch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--subjects", nargs="*", default=None, help="download only these subjects")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    if not cfg.paths.hub.repo_id:
        raise SystemExit("set paths.hub.repo_id in the config first")

    patterns = ["metadata.parquet"]
    if args.subjects:
        patterns += [f"subjects/{s}/*" for s in args.subjects]

    dest = cfg.paths.dataset()
    local = fetch(HubConfig(repo_id=cfg.paths.hub.repo_id, revision=cfg.paths.hub.revision),
                  dest, allow_patterns=patterns if args.subjects else None)
    print(f"dataset -> {local}")


if __name__ == "__main__":
    main()
