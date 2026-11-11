"""The hardware model menu must not grow when unrelated exports are added."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dashboard.backend import server
from dashboard.backend.model_catalog import MODEL_NAMES


def test_catalog_includes_only_selected_subject_embedding_models(tmp_path, monkeypatch):
    runs = tmp_path / 'ML_model/runs'
    excluded = [
        'model_film_context_current_action_terrain',
        'model_moe_current_action_terrain',
        'model_tcn_current_action_terrain',
        'model_subject_embedding_current_action_terrain',
        'model_subject_embedding_current_action_ablation_no_imu',
        'tcn_mid_stance_lastN_20260831_212705',
    ]
    for name in [*MODEL_NAMES, *excluded]:
        run = runs / name
        run.mkdir(parents=True)
        (run/'deploy_metadata.json').write_text('{}')
        (run/'best.onnx').write_bytes(b'export')
        (run/'display_name.txt').write_text('Old ambiguous name')
        arch = 'film' if name.startswith('model_film_') else 'tcn'
        (run/'model_meta.json').write_text(json.dumps({'model': {'arch': arch},
                                                     'deployment_notes': 'Sensor frame note'}))
    monkeypatch.setattr(server, 'ML_MODEL_RUNS', runs)
    monkeypatch.setattr(server, 'REPO_ROOT', tmp_path)
    monkeypatch.setattr(server.importlib.util, 'find_spec', lambda _: None)
    models = server.list_models()
    assert models[0]['controller'] == 'tbe'
    ml = models[1:]
    assert {m['id'] for m in ml} == set(MODEL_NAMES)
    assert {m['name'] for m in ml} == set(MODEL_NAMES.values())
    assert all(m['hardware_test_status'] == 'Awaiting hardware testing'
               for m in ml if m['id'] != 'deploy')
    assert next(m for m in ml if m['id'] == 'deploy')['hardware_test_status'] == 'Previously deployed baseline'
    film = next(m for m in ml if m['id'] == 'model_film_subject_embedding_current_action_terrain')
    assert film['kind'] == 'film'
    assert film['controller'] == 'ml'
    assert film['deployment_notes'] == 'Sensor frame note'
    assert 'onnx' in film['backends']


def test_missing_selected_export_is_not_listed(tmp_path, monkeypatch):
    (tmp_path/'deploy').mkdir()
    monkeypatch.setattr(server, 'ML_MODEL_RUNS', tmp_path)
    assert len(server.list_models()) == 1  # TBE only
