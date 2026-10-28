"""Build (and cache) the list of valid sliding windows for a split.

A window is ``(trial_name, start, end_inclusive)``. Windows only cover contiguous
regions where the target is non-NaN and at least ``min_segment_length`` long.

The index is deterministic given (trial set, window_length, stride,
min_segment_length), so it is hashed and cached to ``<processed>/window_index/``
— building it means scanning every trial's target once; loading the cache is a
single ``np.load``.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def _key(trial_names: list[str], window_length: int, stride: int, min_seg: int) -> str:
    h = hashlib.sha1()
    h.update(json.dumps(sorted(trial_names)).encode())
    h.update(f"|{window_length}|{stride}|{min_seg}".encode())
    return h.hexdigest()[:16]


def _windows_for_trial(target: np.ndarray, window_length: int, stride: int,
                       min_seg: int) -> list[tuple[int, int]]:
    valid = ~np.isnan(target).any(axis=1)
    out: list[tuple[int, int]] = []
    n = len(valid)
    start = None
    for i in range(n + 1):
        v = valid[i] if i < n else False
        if v and start is None:
            start = i
        elif not v and start is not None:
            seg_len = i - start
            if seg_len >= min_seg:
                last_start = i - window_length
                for s in range(start, last_start + 1, stride):
                    out.append((s, s + window_length - 1))
            start = None
    return out


class WindowIndex:
    def __init__(self, entries: np.ndarray, trial_names: list[str]):
        # entries: (N, 3) int32  -> (trial_id, start, end_inclusive)
        self.entries = entries
        self.trial_names = trial_names

    def __len__(self) -> int:
        return len(self.entries)

    @classmethod
    def build(
        cls,
        processed_dir: str | Path,
        trial_names: list[str],
        window_length: int,
        stride: int,
        min_seg: int,
        cache: bool = True,
    ) -> "WindowIndex":
        processed_dir = Path(processed_dir)
        cache_dir = processed_dir / "window_index"
        key = _key(trial_names, window_length, stride, min_seg)
        cache_path = cache_dir / f"{key}.npz"

        if cache and cache_path.exists():
            data = np.load(cache_path, allow_pickle=True)
            return cls(data["entries"], list(data["trial_names"]))

        entries: list[tuple[int, int, int]] = []
        for tid, name in enumerate(trial_names):
            npz = processed_dir / f"{name}.npz"
            if not npz.exists():
                continue
            with np.load(npz) as d:
                target = d["target"]
            for s, e in _windows_for_trial(target, window_length, stride, min_seg):
                entries.append((tid, s, e))

        if not entries:
            raise ValueError(
                f"No valid windows for {len(trial_names)} trials "
                f"(window_length={window_length}, min_seg={min_seg})"
            )

        arr = np.asarray(entries, dtype=np.int32)
        if cache:
            cache_dir.mkdir(parents=True, exist_ok=True)
            np.savez(cache_path, entries=arr, trial_names=np.array(trial_names))
        return cls(arr, list(trial_names))
