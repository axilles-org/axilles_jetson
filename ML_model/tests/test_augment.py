"""Augmentations, incl. the deployment domain-randomisation knobs."""
import numpy as np
import torch

from exo.config import AugmentConfig
from exo.data.augment import Augmenter, _small_rotation

_FEAT = [
    "imu_foot_Accel_X", "imu_foot_Accel_Y", "imu_foot_Accel_Z",
    "imu_foot_Gyro_X", "imu_foot_Gyro_Y", "imu_foot_Gyro_Z",
    "imu_shank_Accel_X", "imu_shank_Accel_Y", "imu_shank_Accel_Z",
    "imu_shank_Gyro_X", "imu_shank_Gyro_Y", "imu_shank_Gyro_Z",
    "stance", "gon_ankle_sagittal",
]


def _aug(**kw):
    kw.setdefault("noise_std", 0.0)
    kw.setdefault("imu_gain_jitter", 0.0)
    cfg = AugmentConfig(enabled=True, **kw)
    a = Augmenter(cfg, _FEAT)
    # identity scaler -> z-space == physical space for the checks
    a.set_scaler(np.zeros(len(_FEAT), np.float32), np.ones(len(_FEAT), np.float32))
    return a


def _window(seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(len(_FEAT), 300, generator=g)
    # stance channel: 60 % contact per 100-sample cycle, at z-levels 0/1
    x[12] = (torch.arange(300) % 100 < 60).float()
    return x


def test_disabled_is_identity():
    cfg = AugmentConfig(enabled=False)
    a = Augmenter(cfg, _FEAT)
    x = _window()
    assert torch.equal(a(x), x)


def test_small_rotation_is_orthonormal():
    for _ in range(20):
        R = _small_rotation(10.0)
        assert torch.allclose(R @ R.T, torch.eye(3), atol=1e-5)
        assert abs(float(torch.det(R)) - 1.0) < 1e-5


def test_imu_rotation_preserves_triad_norm():
    a = _aug(imu_rotation_deg=12.0)
    x = torch.zeros(len(_FEAT), 300)
    x[0], x[1], x[2] = 1.0, 2.0, 2.0            # foot accel triad, |.| = 3
    y = a(x)
    n_in = x[[0, 1, 2]].pow(2).sum(0).sqrt()
    n_out = y[[0, 1, 2]].pow(2).sum(0).sqrt()
    assert torch.allclose(n_in, n_out, atol=1e-4)


def test_stance_stays_two_level_under_all_dr():
    a = _aug(noise_std=0.02, imu_gain_jitter=0.05, imu_rotation_deg=8.0,
             encoder_offset_rad=0.05, stance_jitter_samples=3, assist_perturb=0.3,
             time_warp=0.05)
    x = _window()
    for _ in range(50):
        y = a(x)
        assert torch.isfinite(y).all()
        levels = set(round(v, 4) for v in y[12].unique().tolist())
        assert levels <= {0.0, 1.0}


def test_stance_jitter_shifts_edges():
    a = _aug(stance_jitter_samples=4)
    x = _window()
    shifted = 0
    for _ in range(30):
        y = a(x)
        if not torch.equal(y[12], x[12]):
            shifted += 1
    assert shifted > 0                          # at least sometimes moves


def test_encoder_offset_biases_ankle_channel():
    a = _aug(encoder_offset_rad=0.1)
    x = torch.zeros(len(_FEAT), 300)
    diffs = [float((a(x)[13] - x[13]).mean()) for _ in range(40)]
    assert max(abs(d) for d in diffs) > 1e-3    # some nonzero bias
    assert abs(np.mean(diffs)) < 0.05           # zero-mean over draws


def test_assist_perturb_reduces_late_stance_foot_gyro():
    a = _aug(assist_perturb=0.5)
    x = torch.ones(len(_FEAT), 300)             # foot gyro = 1 everywhere
    ys = torch.stack([a(x) for _ in range(50)]).mean(0)
    early = ys[3:6, :100].mean()                # foot gyro, early window
    late = ys[3:6, -50:].mean()                 # foot gyro, push-off region
    assert late < early                         # push-off spin reduced


def test_all_dr_preserves_shape_and_finiteness():
    a = _aug(latency_samples=5, imu_rotation_deg=8.0, encoder_offset_rad=0.05,
             stance_jitter_samples=3, assist_perturb=0.3, noise_std=0.01,
             imu_gain_jitter=0.05)
    x = _window()
    for s in range(20):
        y = a(_window(seed=s))
        assert y.shape == x.shape
        assert torch.isfinite(y).all()
