"""TCN forward pass, embedding path, and regularisation."""
import pytest
import torch

from exo.config import replace
from exo.models import TCN


def test_forward_with_embedding(cfg):
    model = TCN.from_config(cfg.model, in_channels=cfg.data.num_features,
                            out_channels=1, num_subjects=13)
    x = torch.randn(4, cfg.data.num_features, cfg.data.window_length)
    out = model(x, torch.randint(0, 13, (4,)))
    assert out.shape == (4, 1, cfg.data.window_length)


def test_embedding_requires_subject_idx(cfg):
    model = TCN.from_config(cfg.model, in_channels=cfg.data.num_features,
                            out_channels=1, num_subjects=13)
    with pytest.raises(ValueError):
        model(torch.randn(2, cfg.data.num_features, 64))


def test_forward_without_embedding(cfg):
    no_emb = replace(cfg, **{"model.use_subject_embedding": False})
    model = TCN.from_config(no_emb.model, in_channels=no_emb.data.num_features, out_channels=1)
    out = model(torch.randn(2, no_emb.data.num_features, 128))
    assert out.shape == (2, 1, 128)


def test_regularisation_loss_scales_with_reg(cfg):
    reg = replace(cfg, **{"model.l2_reg": 1e-3, "model.use_subject_embedding": False})
    model = TCN.from_config(reg.model, in_channels=reg.data.num_features, out_channels=1)
    assert model.regularization_loss().item() > 0.0


def test_receptive_field_matches_config(cfg):
    model = TCN.from_config(cfg.model, in_channels=cfg.data.num_features,
                            out_channels=1, num_subjects=13)
    assert model.receptive_field == cfg.model.receptive_field()
