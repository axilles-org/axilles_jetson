import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from exo.config import Config


@pytest.fixture(scope="session")
def cfg() -> Config:
    return Config.load("configs/default.yaml")


@pytest.fixture(scope="session")
def processed_dir(cfg) -> Path:
    path = cfg.paths.processed()
    if not (path / "scaler_bundle.pkl").exists():
        pytest.skip("processed cache not available; run scripts/prepare.py")
    return path
