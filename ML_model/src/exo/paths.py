"""Repo-root-relative path resolution.

Every path in the project is resolved through here so nothing hardcodes an
absolute path to a particular machine.
"""
from __future__ import annotations

import os
from pathlib import Path

# src/exo/paths.py -> repo root is two parents up from this file's package dir
REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve(path: str | os.PathLike) -> Path:
    """Resolve ``path`` to an absolute Path.

    Absolute paths are returned as-is. Relative paths are taken relative to the
    repo root, not the current working directory, so scripts work from anywhere.
    """
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()
