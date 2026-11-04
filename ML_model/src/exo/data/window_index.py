"""Build (and cache) the list of valid sliding windows for a split.

A window is ``(trial_name, start, end_inclusive)``. By default windows only cover
contiguous regions where the target is non-NaN and at least ``min_segment_length``
long. With ``valid_tail > 0`` a window instead needs NaN-free inputs throughout and a
valid target on its last ``valid_tail`` samples only, and windows ending on a mode
label in ``drop_modes`` are skipped.

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


def _key(trial_names: list[str], window_length: int, stride: int, min_seg: int,
         valid_tail: int = 0, drop_modes: tuple[str, ...] = ()) -> str:
    h = hashlib.sha1()
    h.update(json.dumps(sorted(trial_names)).encode())
    h.update(f"|{window_length}|{stride}|{min_seg}".encode())
    if valid_tail:
        h.update(f"|tail{valid_tail}|{sorted(drop_modes)}".encode())
    return h.hexdigest()[:16]


def _tail_windows(features: np.ndarray, target: np.ndarray, mode: np.ndarray | None,
                  window_length: int, stride: int, valid_tail: int,
                  drop_modes: tuple[str, ...]) -> list[tuple[int, int]]:
    x_bad = np.isnan(features).any(axis=1).astype(np.int64)
    y_bad = np.isnan(target).any(axis=1).astype(np.int64)
    x_cum, y_cum = np.r_[0, np.cumsum(x_bad)], np.r_[0, np.cumsum(y_bad)]
    out = []
    for e in range(window_length - 1, len(target), stride):
        s = e - window_length + 1
        if x_cum[e + 1] - x_cum[s] or y_cum[e + 1] - y_cum[e + 1 - valid_tail]:
            continue
        if mode is not None and mode[e] in drop_modes:
            continue
        out.append((s, e))
    return out


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
        valid_tail: int = 0,
        drop_modes: tuple[str, ...] = (),
    ) -> "WindowIndex":
        processed_dir = Path(processed_dir)
        cache_dir = processed_dir / "window_index"
        key = _key(trial_names, window_length, stride, min_seg, valid_tail, tuple(drop_modes))
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
                if valid_tail:
                    windows = _tail_windows(d["features"], target,
                                            d["mode"] if "mode" in d.files else None,
                                            window_length, stride, valid_tail, tuple(drop_modes))
                else:
                    windows = _windows_for_trial(target, window_length, stride, min_seg)
            for s, e in windows:
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
