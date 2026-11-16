"""Timing corruption must preserve causality, sensor grouping and forecast labels."""
from pathlib import Path

import numpy as np
import pytest
import torch

from exo.config import AugmentConfig, Config, TimingAugmentConfig, replace
from exo.data.timing_augment import TimingAugmenter, acquisition_indices
from tests.test_forecasting import toy


FEATURES = [f'imu_{segment}_{kind}_{axis}'
            for segment in ('foot', 'shank') for kind in ('Accel', 'Gyro')
            for axis in ('X', 'Y', 'Z')] + ['stance', 'gon_ankle_sagittal']


def test_nominal_acquisition_is_identity():
    cfg = TimingAugmentConfig(enabled=True, rate_min_hz=100, rate_max_hz=100,
                             interval_jitter_fraction=0, delay_max_ms=0,
                             packet_drop_probability=0, burst_probability=0)
    result = acquisition_indices(300, 100, cfg, np.random.default_rng(0))
    np.testing.assert_array_equal(result, np.arange(300))


def test_half_rate_holds_previous_sample():
    cfg = TimingAugmentConfig(enabled=True, rate_min_hz=50, rate_max_hz=50,
                             interval_jitter_fraction=0, delay_max_ms=0,
                             packet_drop_probability=0, burst_probability=0)
    result = acquisition_indices(300, 100, cfg, np.random.default_rng(0))
    np.testing.assert_array_equal(result, np.arange(300) // 2 * 2)


def test_missing_all_packets_holds_initial_history():
    cfg = TimingAugmentConfig(enabled=True, packet_drop_probability=1)
    assert not acquisition_indices(300, 100, cfg, np.random.default_rng(0)).any()


def test_timing_is_causal_and_monotonic():
    cfg = TimingAugmentConfig(enabled=True, burst_probability=1)
    for seed in range(30):
        result = acquisition_indices(300, 100, cfg, np.random.default_rng(seed))
        assert np.all(result <= np.arange(300))
        assert np.all(np.diff(result) >= 0)
        assert np.any(np.diff(result) == 0)


def test_groups_stance_reproducibility_and_no_future_leak():
    cfg = TimingAugmentConfig(enabled=True, probability=1)
    aug = TimingAugmenter(cfg, FEATURES, 100)
    x = torch.arange(300).float().repeat(14, 1)
    x[12] = (torch.arange(300) % 100 < 60).float()
    y = aug(x, np.random.default_rng(8))
    assert torch.equal(y[12], x[12])
    for i in (0, 3, 6, 9):
        assert torch.equal(y[i], y[i+1]) and torch.equal(y[i], y[i+2])
    assert not torch.equal(y[0], y[3])  # distinct sensor-report schedules
    changed = x.clone()
    changed[:, 150:] += 1000
    after = aug(changed, np.random.default_rng(8))
    assert torch.equal(y[:, :150], after[:, :150])
    assert torch.equal(y, aug(x, np.random.default_rng(8)))


def test_probability_zero_preserves_clean_windows():
    aug = TimingAugmenter(TimingAugmentConfig(enabled=True, probability=0), FEATURES, 100)
    x = torch.randn(14, 300)
    assert torch.equal(aug(x), x)


@pytest.mark.parametrize('h', [0, 5, 10, 15])
def test_dataset_keeps_physical_target_time(cfg, tmp_path, h):
    ds, _ = toy(cfg, tmp_path, h)
    # Use a recognized report channel in the synthetic dataset.
    ds.feature_names = ['gon_ankle_sagittal']
    ds._augment_cfg = AugmentConfig(noise_std=0, imu_gain_jitter=0,
                                    timing=TimingAugmentConfig(enabled=True, probability=1,
                                                              packet_drop_probability=1))
    ds._augment = True
    _, target, _, _ = ds[0]
    x, actual, _, _ = ds[0]
    np.testing.assert_array_equal(actual[0], np.arange(h, h+4))
    assert torch.equal(actual, target)
    assert torch.equal(x[0], torch.zeros(4, dtype=x.dtype))


@pytest.mark.parametrize('field,value', [('probability', 1.1), ('rate_min_hz', 0),
    ('rate_max_hz', 101), ('interval_jitter_fraction', 1), ('delay_max_ms', -1),
    ('burst_min_ms', 100), ('packet_drop_probability', float('nan'))])
def test_invalid_settings_rejected(cfg, field, value):
    with pytest.raises(ValueError, match='timing'):
        replace(cfg, **{'train.augment.timing.enabled': True,
                        f'train.augment.timing.{field}': value})


def test_four_configs_only_differ_in_horizon_and_name(tmp_path):
    root = Path(__file__).resolve().parents[1] / 'configs/anticipation'
    paths = sorted(root.glob('model_obs_aug_*.yaml'))
    assert len(paths) == 4
    compared, horizons = [], set()
    for path in paths:
        cfg = Config.load(path)
        horizons.add(cfg.data.prediction_horizon_ms)
        aug = cfg.train.augment
        assert aug.timing.enabled
        assert not any((aug.noise_std, aug.imu_gain_jitter, aug.imu_rotation_deg,
                        aug.encoder_offset_rad, aug.stance_jitter_samples,
                        aug.time_warp, aug.latency_samples, aug.channel_dropout, aug.assist_perturb))
        cfg.save(tmp_path)
        assert Config.load(tmp_path / 'config.yaml') == cfg
        d = cfg.to_dict()
        d['data'].pop('prediction_horizon_ms')
        d['train'].pop('wandb_run_name')
        compared.append(d)
    assert horizons == {0, 50, 100, 150}
    assert all(d == compared[0] for d in compared)
