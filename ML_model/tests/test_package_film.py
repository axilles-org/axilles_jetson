"""FiLM packaging must retain its original sensor coordinate conversion."""
from pathlib import Path
import sys

import pytest

from exo.config import Config, replace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from package_film import deployment_config


def test_preserves_training_adapter_and_model():
    source = Config.load(ROOT / 'runs/terrain/20261029_011205/film/config.yaml')
    template = Config.load(ROOT / 'runs/deploy/config.yaml')
    actual = deployment_config(source, template)
    assert source.deploy.sensor_adapter != template.deploy.sensor_adapter
    assert actual.deploy.sensor_adapter == source.deploy.sensor_adapter
    assert actual.model == source.model
    assert actual.data == source.data
    assert actual.train == source.train
    assert actual.deploy.assistance_scale == 0.02
    assert actual.deploy.torque_limit_nm == 2.0
    assert actual.deploy.backend == 'onnx'


def test_rejects_other_architectures():
    source = Config.load(ROOT / 'runs/terrain/20261029_011205/film/config.yaml')
    for override in ({'model.arch': 'tcn'}, {'model.film_source': 'context'},
                     {'data.prediction_horizon_ms': 50}):
        with pytest.raises(ValueError, match='subject-conditioned FiLM'):
            deployment_config(replace(source, **override), source)
