"""Wearer export, engine invalidation, launch selection; no GPU or motor access."""
import json
from pathlib import Path
import sys

import pytest

from exo.config import replace
from exo.deploy import prepare_backend as preparation
from exo.deploy import tensorrt as trt_backend

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dashboard.backend import process_manager, server


@pytest.mark.parametrize('backend,onnx', [('jit', False), ('onnx', True), ('trt', True)])
def test_wearer_export_precedes_engine_preparation(cfg, tmp_path, monkeypatch, backend, onnx):
    events = []
    def export(path, mass, **kwargs):
        assert kwargs['subject_demo'] == (1.8, 74, 'M')
        assert kwargs['onnx'] == onnx
        events.append('export')
    monkeypatch.setattr(preparation, 'export', export)
    monkeypatch.setattr(trt_backend, 'ensure_engine', lambda _: events.append('engine'))
    preparation.prepare_backend(tmp_path, replace(cfg, **{'deploy.backend': backend}), 74, 1.8, 'M')
    assert events == (['export', 'engine'] if backend == 'trt' else ['export'])


def test_engine_error_propagates_before_controller_setup(cfg, tmp_path, monkeypatch):
    def fail(_):
        raise RuntimeError('GPU unavailable')
    monkeypatch.setattr(trt_backend, 'ensure_engine', fail)
    with pytest.raises(RuntimeError, match='GPU unavailable'):
        preparation.prepare_backend(tmp_path, replace(cfg, **{'deploy.backend': 'trt'}), 72)


def test_engine_cache_tracks_export_and_device(tmp_path, monkeypatch):
    (tmp_path / 'best.onnx').write_bytes(b'wearer-one')
    device = {'gpu': 'test-gpu', 'tensorrt': 'test-version'}
    builds = []
    monkeypatch.setattr(trt_backend, '_device_fingerprint', lambda: dict(device))
    def build(onnx, engine, fp16):
        assert fp16 is False
        builds.append(onnx.read_bytes())
        engine.write_bytes(b'engine:' + onnx.read_bytes())
    monkeypatch.setattr(trt_backend, 'build_engine', build)
    monkeypatch.setattr(trt_backend, '_verify_engine', lambda *_: 0.0)
    trt_backend.ensure_engine(tmp_path)
    trt_backend.ensure_engine(tmp_path)
    assert len(builds) == 1
    (tmp_path / 'best.onnx').write_bytes(b'wearer-two')
    trt_backend.ensure_engine(tmp_path)
    assert len(builds) == 2
    device['gpu'] = 'another-gpu'
    trt_backend.ensure_engine(tmp_path)
    assert len(builds) == 3
    (tmp_path / 'best.engine').write_bytes(b'corrupt')
    trt_backend.ensure_engine(tmp_path)
    assert len(builds) == 4


def test_bad_engine_never_replaces_previous(tmp_path, monkeypatch):
    (tmp_path / 'best.onnx').write_bytes(b'new-export')
    (tmp_path / 'best.engine').write_bytes(b'previous-engine')
    monkeypatch.setattr(trt_backend, '_device_fingerprint', lambda: {})
    monkeypatch.setattr(trt_backend, 'build_engine', lambda onnx, out, **kw: out.write_bytes(b'bad-engine'))
    def fail(*_):
        raise RuntimeError('does not match ONNX')
    monkeypatch.setattr(trt_backend, '_verify_engine', fail)
    with pytest.raises(RuntimeError, match='does not match ONNX'):
        trt_backend.ensure_engine(tmp_path)
    assert (tmp_path / 'best.engine').read_bytes() == b'previous-engine'
    assert not (tmp_path / 'best.engine.json').exists()


@pytest.mark.parametrize('mode', ['ml-live', 'ml-dry-run', 'ml-mock'])
@pytest.mark.parametrize('backend', ['onnx', 'trt'])
def test_dashboard_launch_forwards_backend_and_wearer(mode, backend):
    argv, _ = process_manager.build_argv(mode, 'runs/model_obs_aug_subject_embedding_current_action',
                                        72, None, 10, backend, 0.02, 1.75, 'M', 2)
    assert argv[argv.index('--backend')+1] == backend
    assert argv[argv.index('--height')+1] == '1.75'
    assert argv[argv.index('--gender')+1] == 'M'
    assert ('--arm' in argv) == (mode == 'ml-live')
    if mode == 'ml-live':
        assert '--start-passive' in argv


def test_model_selector_advertises_buildable_trt(tmp_path, monkeypatch):
    run = tmp_path / 'model_obs_aug_subject_embedding_future_action_50'
    run.mkdir()
    (run/'best.onnx').write_bytes(b'onnx')
    (run/'deploy_metadata.json').write_text(json.dumps({'prediction_horizon_ms': 50}))
    (run/'model_meta.json').write_text(json.dumps({'observation_augmentation': True}))
    # Use the actual expected directory structure in the temporary repo.
    repo = tmp_path / 'repo'
    runs = repo / 'ML_model/runs'
    runs.mkdir(parents=True)
    run.rename(runs / run.name)
    monkeypatch.setattr(server, 'ML_MODEL_RUNS', runs)
    monkeypatch.setattr(server, 'REPO_ROOT', repo)
    monkeypatch.setattr(server.importlib.util, 'find_spec', lambda _: object())
    model = server.list_models()[1]
    assert model['prediction_horizon_ms'] == 50
    assert model['observation_augmentation']
    assert 'trt' in model['backends']  # ONNX can be compiled; no engine required yet
    monkeypatch.setattr(server.importlib.util, 'find_spec', lambda _: None)
    model = server.list_models()[1]
    assert 'trt' not in model['backends']
    assert not next(b for b in model['backend_options'] if b['id'] == 'trt')['available']
