"""Config load / save / override round-trips and validation."""
import pytest

from exo.config import Config, replace


def test_load_and_feature_resolution(cfg):
    assert cfg.data.feature_set == "mid"
    assert cfg.data.num_features == len(cfg.data.feature_names())
    assert "stance" in cfg.data.feature_names()
    assert "fp_Treadmill_R_vy" not in cfg.data.feature_names()


def test_save_roundtrip(cfg, tmp_path):
    cfg.save(tmp_path)
    reloaded = Config.load(tmp_path / "config.yaml")
    assert reloaded == cfg


def test_dotted_override(cfg):
    updated = replace(cfg, **{"train.epochs": 3, "model.kernel_size": 3})
    assert updated.train.epochs == 3
    assert updated.model.kernel_size == 3
    assert updated.model.receptive_field() < cfg.model.receptive_field()


def test_split_must_be_disjoint(cfg):
    with pytest.raises(ValueError):
        replace(cfg, **{"data.split": {
            "train": ["AB06"], "val": ["AB06"], "test": ["AB10"],
        }})


def test_loss_last_n_bounds(cfg):
    with pytest.raises(ValueError):
        replace(cfg, **{"train.loss_last_n": cfg.data.window_length + 1})
