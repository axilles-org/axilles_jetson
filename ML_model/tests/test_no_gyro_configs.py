"""No-gyro experiments keep targets/settings fixed and receive unique names."""
from pathlib import Path
import sys

import numpy as np
import torch

from exo.config import Config
from exo.data.timing_augment import TimingAugmenter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from train_anticipation import model_name


def test_matched_no_gyro_configs_and_names():
    configs = sorted((ROOT/'configs/anticipation/no_gyro').glob('model_*.yaml'))
    assert len(configs) == 4
    names, horizons = set(), set()
    for path in configs:
        cfg = Config.load(path)
        original = Config.load(path.parent.parent / path.name.replace('_no_gyro', ''))
        features = cfg.data.feature_names()
        assert features == [f for f in original.data.feature_names() if '_Gyro_' not in f]
        assert len(features) == 8
        assert model_name(cfg) == path.stem != model_name(original)
        names.add(model_name(cfg)); horizons.add(cfg.data.prediction_horizon_ms)
        a, b = cfg.to_dict(), original.to_dict()
        for d in (a, b):
            d['data'].pop('feature_set'); d['data'].pop('feature_sets')
            d['train'].pop('wandb_run_name')
        assert a == b
        aug = TimingAugmenter(cfg.train.augment.timing, features, 100)
        x = torch.arange(300).float().repeat(8, 1)
        x[6] = (torch.arange(300) % 100 < 60).float()
        y = aug(x, np.random.default_rng(0))
        assert y.shape == (8, 300)
        assert torch.equal(y[6], x[6])
    assert len(names) == 4 and horizons == {0, 50, 100, 150}


def test_no_imu_current_model_uses_only_angle_and_stance():
    path = ROOT/'configs/anticipation/no_imu/model_obs_aug_subject_embedding_no_imu_current_action.yaml'
    cfg = Config.load(path)
    original = Config.load(ROOT/'configs/anticipation/model_obs_aug_subject_embedding_current_action.yaml')
    assert cfg.data.feature_names() == ['stance', 'gon_ankle_sagittal']
    assert cfg.data.prediction_horizon_ms == 0
    assert model_name(cfg) == path.stem
    a, b = cfg.to_dict(), original.to_dict()
    for d in (a, b):
        d['data'].pop('feature_set'); d['data'].pop('feature_sets')
        d['train'].pop('wandb_run_name')
    assert a == b
    aug = TimingAugmenter(cfg.train.augment.timing, cfg.data.feature_names(), 100)
    x = torch.randn(2, 300)
    y = aug(x, np.random.default_rng(0))
    assert y.shape == (2, 300)
    assert torch.equal(y[0], x[0])
