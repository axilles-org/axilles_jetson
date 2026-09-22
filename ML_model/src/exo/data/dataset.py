"""Sliding-window Dataset over the ingested .npz trials.

Per-trial arrays are memory-mapped; the window index is built once and cached.
``__getitem__`` returns ``(x, y, trial_name, subject_id)`` with ``x`` a ``(C, T)``
z-scored feature window and ``y`` a ``(1, T)`` z-scored target.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import AugmentConfig, DataConfig
from .augment import Augmenter
from .feature_pipeline import FeaturePipeline
from .scalers import ScalerBundle
from .window_index import WindowIndex


class WindowDataset(Dataset):
    def __init__(
        self,
        cfg: DataConfig,
        processed_dir: str | Path,
        scalers: ScalerBundle,
        split: str = "train",
        augment_cfg: AugmentConfig | None = None,
        verbose: bool = False,
    ):
        self.cfg = cfg
        self.processed_dir = Path(processed_dir)
        self.scalers = scalers
        self.split = split

        self.feature_names = cfg.feature_names()
        self.pipeline = FeaturePipeline(self.feature_names)

        # z-score params restricted to this feature set (superset scaler -> subset)
        self._x_mean, self._x_scale = scalers.subset_x(self.feature_names)
        self._y_mean = scalers.scaler_y.mean_.astype(np.float32)
        self._y_scale = scalers.scaler_y.scale_.astype(np.float32)

        self.trial_names = self._trials_for_split(split)
        self.index = WindowIndex.build(
            self.processed_dir, self.trial_names,
            cfg.window_length, cfg.stride, cfg.min_segment_length,
            cache=cfg.cache_window_index,
        )

        self._augment = split == "train" and augment_cfg is not None and augment_cfg.enabled
        self._augment_cfg = augment_cfg
        self._augmenter: Augmenter | None = None

        # lazy per-worker mmap handles
        self._feat_mm: dict[int, np.ndarray] = {}
        self._targ_mm: dict[int, np.ndarray] = {}
        self._cols: dict[int, list[str]] = {}

        if verbose:
            print(f"[{split}] trials={len(self.trial_names)}  windows={len(self.index)}  "
                  f"features={self.feature_names}  augment={self._augment}")

    # -- split resolution -------------------------------------------
    def _trials_for_split(self, split: str) -> list[str]:
        groups = {"train": self.cfg.split.train, "val": self.cfg.split.val,
                  "test": self.cfg.split.test}
        if split not in groups:
            raise ValueError(f"unknown split {split!r}")
        subjects = set(groups[split])
        names = sorted(
            p.stem for p in self.processed_dir.glob("*.npz")
            if p.stem.split("_")[0] in subjects
        )
        if not names:
            raise ValueError(f"no ingested trials for split={split} (subjects {sorted(subjects)})")
        return names

    # -- mmap access ---------------------------------------------
    def _load(self, tid: int):
        if tid not in self._feat_mm:
            name = self.index.trial_names[tid]
            path = self.processed_dir / f"{name}.npz"
            d = np.load(path, mmap_mode="r")
            self._feat_mm[tid] = d["features"]
            self._targ_mm[tid] = d["target"]
            with np.load(path, allow_pickle=True) as meta:
                self._cols[tid] = list(meta["feature_columns"])
        return self._feat_mm[tid], self._targ_mm[tid], self._cols[tid]

    # -- Dataset API -------------------------------------------
    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        tid, s, e = (int(v) for v in self.index.entries[i])
        feat_all, targ_all, cols = self._load(tid)

        # Latency DR: shift the input window back a random k samples while keeping
        # the target aligned to the true window end. Teaches "predict the moment k
        # samples ahead", so a fixed deployment lag lands in-distribution.
        lat = 0
        if self._augment and self._augment_cfg.latency_samples > 0:
            lat = int(np.random.randint(0, self._augment_cfg.latency_samples + 1))
            lat = min(lat, s)                             # never read before frame 0
        xs, xe = s - lat, e - lat

        raw = np.asarray(feat_all[xs : xe + 1])                   # (T, F_super)
        feat = self.pipeline.transform(raw, columns=cols)         # (T, C) ordered
        x = (feat - self._x_mean) / self._x_scale                 # z-score
        y = (np.asarray(targ_all[s : e + 1]) - self._y_mean) / self._y_scale

        x_t = torch.from_numpy(np.ascontiguousarray(x.T))         # (C, T)
        y_t = torch.from_numpy(np.ascontiguousarray(y.T))         # (1, T)

        if self._augment:
            if self._augmenter is None:
                self._augmenter = Augmenter(self._augment_cfg, self.feature_names)
                self._augmenter.set_scaler(self._x_mean, self._x_scale)
            x_t = self._augmenter(x_t)

        name = self.index.trial_names[tid]
        return x_t, y_t, name, name.split("_")[0]
