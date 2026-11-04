"""Calibration packaging and preflight must work before hardware is opened."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from exo.config import replace
from exo.deploy.ankle_reference import gatech_ankle_median_deg, load_ankle_reference

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import jetson_deploy
import package_anticipation


def training_config(cfg, tmp_path):
    for subject, degrees in [('TRAIN', [10., 20., 30.]), ('TEST', [90., 90., 90.])]:
        np.savez(tmp_path / f'{subject}_walking.npz',
                 feature_columns=['gon_ankle_sagittal'],
                 features=np.radians(degrees).reshape(-1, 1))
    return replace(cfg, **{'paths.processed_dir': str(tmp_path),
                          'data.split.train': ['TRAIN']})


def test_reference_uses_only_training_subjects_and_degrees(cfg, tmp_path):
    cfg = training_config(cfg, tmp_path)
    assert gatech_ankle_median_deg(cfg) == pytest.approx(20.)


def test_repair_adds_reference_to_manifest_without_changing_weights(cfg, tmp_path, monkeypatch):
    cfg = training_config(cfg, tmp_path)
    target = tmp_path / 'package'
    target.mkdir()
    (target / 'best.pt').write_bytes(b'unchanged checkpoint')
    (target / 'package_manifest.json').write_text(json.dumps({'files_sha256': {}}))
    monkeypatch.setattr(package_anticipation, 'ROOT', tmp_path)
    package_anticipation.ensure_ankle_reference(target, cfg, tmp_path / 'source')
    assert load_ankle_reference(target) == pytest.approx(20.)
    assert (target / 'best.pt').read_bytes() == b'unchanged checkpoint'
    manifest = json.loads((target / 'package_manifest.json').read_text())
    assert manifest['files_sha256']['ankle_reference.json'] == hashlib.sha256(
        (target / 'ankle_reference.json').read_bytes()).hexdigest()
    # An existing valid package does not require the training dataset on Jetson.
    monkeypatch.setattr(package_anticipation, 'gatech_ankle_median_deg',
                        lambda _: pytest.fail('must not regenerate existing reference'))
    package_anticipation.ensure_ankle_reference(target, cfg, tmp_path / 'source')


@pytest.mark.parametrize('content', [None, '{}', '{broken',
                                   '{"gatech_walking_median_deg": NaN}',
                                   '{"gatech_walking_median_deg": true}'])
def test_bad_reference_fails_before_backend_or_hardware(cfg, tmp_path, monkeypatch, content):
    if content is not None:
        (tmp_path / 'ankle_reference.json').write_text(content)
    monkeypatch.setattr(sys, 'argv', ['jetson_deploy', '--run', str(tmp_path),
                                    '--mass', '72', '--ankle-zero', 'auto', '--arm'])
    monkeypatch.setattr(jetson_deploy.Config, 'load', lambda _: cfg)
    monkeypatch.setattr(jetson_deploy, 'prepare_backend',
                        lambda *args: pytest.fail('must fail before backend/hardware setup'))
    with pytest.raises(ValueError, match='Automatic ankle calibration requires'):
        jetson_deploy.main()
