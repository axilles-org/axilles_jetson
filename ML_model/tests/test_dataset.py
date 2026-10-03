"""Windowing, split disjointness, and NaN handling in WindowDataset."""
import numpy as np
import torch

from exo.data.dataset import WindowDataset
from exo.data.scalers import ScalerBundle
from exo.data.window_index import _windows_for_trial


def test_windows_skip_nan_regions():
    target = np.full((100, 1), np.nan)
    target[10:70] = 1.0  # one 60-long valid region
    windows = _windows_for_trial(target, window_length=20, stride=10, min_seg=20)
    assert windows
    for start, end in windows:
        assert 10 <= start and end <= 69
        assert end - start + 1 == 20


def test_shapes_and_normalisation(cfg, processed_dir):
    scalers = ScalerBundle.load(processed_dir)
    ds = WindowDataset(cfg.data, processed_dir, scalers, split="val")
    x, y, name, subject = ds[0]
    assert x.shape == (cfg.data.num_features, cfg.data.window_length)
    assert y.shape == (1, cfg.data.window_length)
    assert x.dtype == torch.float32
    assert subject in cfg.data.split.val

    batch = torch.stack([ds[i][0] for i in range(0, min(len(ds), 2000), 20)])
    per_channel_std = batch.permute(1, 0, 2).reshape(cfg.data.num_features, -1).std(1)
    assert per_channel_std.mean().item() < 1.5  # roughly unit scale


def test_splits_are_subject_disjoint(cfg, processed_dir):
    scalers = ScalerBundle.load(processed_dir)
    subjects = {}
    for split in ("train", "val", "test"):
        ds = WindowDataset(cfg.data, processed_dir, scalers, split=split)
        subjects[split] = {n.split("_")[0] for n in ds.trial_names}
    assert not subjects["train"] & subjects["val"]
    assert not subjects["train"] & subjects["test"]
    assert not subjects["val"] & subjects["test"]


def test_window_index_cache_is_reused(cfg, processed_dir):
    scalers = ScalerBundle.load(processed_dir)
    first = WindowDataset(cfg.data, processed_dir, scalers, split="val")
    second = WindowDataset(cfg.data, processed_dir, scalers, split="val")
    np.testing.assert_array_equal(first.index.entries, second.index.entries)
