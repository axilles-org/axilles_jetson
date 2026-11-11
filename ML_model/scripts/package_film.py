"""Package the original subject-conditioned terrain FiLM for dashboard selection.

Unlike the newer exo-frame TCNs, this checkpoint requires the source run's
exo-to-GaTech sensor adapter. Preserve it when adopting current command defaults.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from exo.config import Config
from exo.deploy.export import export
from package_anticipation import digest, ensure_ankle_reference

NAME = 'model_film_subject_embedding_current_action_terrain'


def deployment_config(training: Config, template: Config) -> Config:
    if (training.model.arch != 'film' or training.model.film_source != 'subject'
            or not training.model.use_subject_embedding
            or training.data.prediction_horizon_ms != 0):
        raise ValueError('Expected current-time subject-conditioned FiLM')
    values = training.to_dict()
    adapter = values['deploy']['sensor_adapter']
    values['deploy'] = template.to_dict()['deploy']
    values['deploy']['sensor_adapter'] = adapter
    values['deploy']['assistance_scale'] = 0.02
    values['deploy']['torque_limit_nm'] = 2.0
    values['deploy']['backend'] = 'onnx'
    values['deploy']['device'] = 'cpu'
    return Config.from_dict(values)


def package(source: Path, destination: Path, template: Config) -> Path:
    source = source.resolve()
    target = destination.resolve() / NAME
    if target.exists():
        raise FileExistsError(f'Refusing to overwrite {target}')
    cfg = Config.load(source / 'config.yaml')
    packaged = deployment_config(cfg, template)
    scores = json.loads((source.parent / 'summary.json').read_text())['film']['nearest']['overall']
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / '.staging'
    staging.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=staging) as temporary:
        run = Path(temporary) / NAME
        run.mkdir()
        for filename in ('best.pt', 'scaler_bundle.pkl', 'model_meta.json', 'test_metrics.json'):
            shutil.copy2(source / filename, run / filename)
        shutil.copy2(source / 'config.yaml', run / 'training_config.yaml')
        packaged.save(run)
        ensure_ankle_reference(run, cfg, source)
        demographics = pd.read_parquet(cfg.paths.demographics())
        demographics[demographics.subject.isin(cfg.data.split.train)].to_parquet(
            run / 'demographics.parquet', index=False)
        meta = json.loads((run / 'model_meta.json').read_text())
        meta.update(
            model_name=NAME, observation_augmentation=cfg.train.augment.timing.enabled,
            training_augmentation=cfg.to_dict()['train']['augment'],
            deployment_notes=('Experimental subject FiLM, 5 epochs. Trained in GaTech '
                              'coordinates; source sensor adapter retained. '
                              'No observation timing augmentation; offline preprocessing was noncausal.'),
        )
        (run / 'model_meta.json').write_text(json.dumps(meta, indent=2) + '\n')
        (run / 'display_name.txt').write_text(NAME + '\n')
        metrics = {
            'source': str((source.parent / 'summary.json').relative_to(ROOT)),
            'evaluation': 'Pooled held-out last-timestep predictions, nearest demographic embedding',
            'prediction_horizon_ms': 0,
            'metrics': {'overall_streaming': {
                'n': scores['n'], 'r2': scores['r2'],
                'rmse_nm_per_kg': scores['rmse'], 'mae_nm_per_kg': scores['mae'],
            }},
        }
        (run / 'forecast_metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
        export(run, 72.0, subject_demo=(1.75, 72.0, 'M'), onnx=True)
        manifest = {
            'model_name': NAME,
            'source_run': str(source.relative_to(ROOT)),
            'checkpoint_sha256': digest(source / 'best.pt'),
            'prediction_horizon_ms': 0,
            'sensor_adapter_source': 'training_config.yaml: deploy.sensor_adapter (unchanged)',
            'hardware_test_status': 'Awaiting hardware testing',
            'target_semantics': 'biological ankle moment, not a learned motor action',
            'export_default_subject': {'height_m': 1.75, 'mass_kg': 72.0, 'gender': 'M'},
            'files_sha256': {p.name: digest(p) for p in sorted(run.iterdir()) if p.is_file()},
        }
        (run / 'package_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        run.rename(target)
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'runs/terrain/20261029_011205/film')
    parser.add_argument('--destination', type=Path, default=ROOT / 'runs')
    parser.add_argument('--deploy-template', type=Path, default=ROOT / 'runs/deploy/config.yaml')
    args = parser.parse_args()
    print(package(args.source, args.destination, Config.load(args.deploy_template)))
