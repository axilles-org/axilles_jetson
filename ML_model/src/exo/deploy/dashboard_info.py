"""
Builds the dashboard's ModelInfo from a checkpoint directory. Shared by both
deploy scripts (jetson_deploy.py, jetson_mock_deploy.py) so a run started
from either one carries identical model/architecture metadata.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# repo root (four levels up from src/exo/deploy/) so dashboard/ is importable,
# same pattern used by TBE_controller/main.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from dashboard.backend.run_logger import ModelInfo  # noqa: E402

from ..config import Config


def build_model_info(run_dir: Path, cfg: Config) -> ModelInfo:
    """
    Assemble ModelInfo from whatever this checkpoint directory actually has.
    Older checkpoints (like tcn_mid_stance_lastN_20260831_212705) predate
    wandb_run.json, so that part is best-effort — everything else
    (architecture, feature set, offline test metrics) always exists because
    ExoController itself depends on deploy_metadata.json being present.
    """
    def _load_json(name: str) -> dict:
        p = run_dir / name
        if p.exists():
            return json.loads(p.read_text())
        return {}

    model_meta = _load_json("model_meta.json")
    deploy_meta = _load_json("deploy_metadata.json")
    test_metrics = _load_json("test_metrics.json")
    wandb_run = _load_json("wandb_run.json")

    architecture = {
        **model_meta.get("model", {}),
        "feature_names": model_meta.get("feature_names"),
        "num_features": model_meta.get("num_features"),
        "window_length": model_meta.get("window_length"),
        "num_training_subjects": model_meta.get("num_training_subjects"),
        "backend": cfg.deploy.backend,
        "control_rate_hz": deploy_meta.get("control_rate_hz"),
        "assistance_scale": cfg.deploy.assistance_scale,
        "test_rmse_nm_per_kg": test_metrics.get("rmse_nm_per_kg"),
        "test_mae_nm_per_kg": test_metrics.get("mae_nm_per_kg"),
    }

    return ModelInfo(
        name=run_dir.name,
        wandb_run_url=wandb_run.get("run_url"),
        wandb_run_id=wandb_run.get("run_id"),
        wandb_project=wandb_run.get("project"),
        architecture=architecture,
        checkpoint_ref=str(run_dir),
    )
