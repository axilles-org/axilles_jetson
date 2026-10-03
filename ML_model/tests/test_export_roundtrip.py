"""Exported TorchScript module must match the eager pipeline, including scaler_x."""
import json

import pytest
import torch

from exo.config import Config, replace
from exo.data.scalers import ScalerBundle
from exo.deploy.export import build_deploy_module
from exo.models import TCN


@pytest.fixture(scope="module")
def trained_run(tmp_path_factory, processed_dir):
    """A minimally initialised run directory sufficient for export."""
    run_dir = tmp_path_factory.mktemp("run")
    cfg = replace(Config.load("configs/default.yaml"), **{"model.num_channels": [16, 16]})
    cfg.save(run_dir)
    ScalerBundle.load(processed_dir).save(run_dir)

    idx_map = {s: i for i, s in enumerate(sorted(cfg.data.split.train))}
    (run_dir / "model_meta.json").write_text(json.dumps({
        "num_training_subjects": len(idx_map),
        "subject_index": idx_map,
        "feature_names": cfg.data.feature_names(),
    }))
    model = TCN.from_config(cfg.model, in_channels=cfg.data.num_features,
                            out_channels=1, num_subjects=len(idx_map))
    torch.save(model.state_dict(), run_dir / "best.pt")
    return run_dir


def test_torchscript_matches_eager(trained_run):
    deploy, info = build_deploy_module(trained_run, subject_mass_kg=72.0,
                                       subject_demo=(1.75, 72.0, "M"))
    dummy = torch.randn(1, info["num_input_channels"], info["window_length"])
    with torch.no_grad():
        eager = deploy(dummy)
        traced = torch.jit.trace(deploy, dummy)
        scripted = traced(dummy)
    assert torch.allclose(eager, scripted, atol=1e-5)


def test_input_scaler_is_baked_in(trained_run):
    deploy, info = build_deploy_module(trained_run, subject_mass_kg=72.0,
                                       subject_demo=(1.75, 72.0, "M"))
    x_raw = torch.randn(1, info["num_input_channels"], info["window_length"])
    with torch.no_grad():
        baked = deploy(x_raw)
        x_z = (x_raw - deploy.x_mean) / deploy.x_scale
        sidx = deploy.subject_idx if deploy.use_subject_embedding else None
        manual = deploy.model(x_z, sidx)[0, :, -1]
        manual = (manual * deploy.y_scale + deploy.y_mean) / deploy.subject_mass
    assert torch.allclose(baked, manual, atol=1e-6)
