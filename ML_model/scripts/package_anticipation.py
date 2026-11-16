"""Publish completed forecast experiments as portable dashboard model folders.

Keeps original experiments untouched. The supplied deployment template provides
the hardware adapter and command settings; training settings/weights are retained.
"""
from __future__ import annotations

import argparse
import hashlib
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
from exo.deploy.ankle_reference import gatech_ankle_median_deg, load_ankle_reference
from train_anticipation import model_name


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ensure_ankle_reference(target: Path, cfg: Config, source: Path) -> None:
    """Add the missing calibration artifact without replacing model weights."""
    path = target / 'ankle_reference.json'
    if path.exists():
        load_ankle_reference(target)
    else:
        reference = {
            'gatech_walking_median_deg': gatech_ankle_median_deg(cfg),
            'source_run': str(source.relative_to(ROOT.resolve())),
            'training_subjects': cfg.data.split.train,
            'method': 'median finite processed training ankle angles, radians converted to degrees',
        }
        path.write_text(json.dumps(reference, indent=2) + '\n')
    manifest_path = target / 'package_manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        manifest['files_sha256'][path.name] = digest(path)
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')


def package(source: Path, destination: Path, template: Config) -> Path:
    source, destination = source.resolve(), destination.resolve()
    cfg = Config.load(source / 'config.yaml')
    name = model_name(cfg)
    target = destination / name
    status = json.loads((source / 'status.json').read_text())
    if status['state'] != 'completed':
        raise ValueError(f'Experiment is not complete: {source}')
    training_hash = digest(source / 'best.pt')
    # Source remains the reproducible training record; package config uses the
    # current transformed-data hardware adapter instead of legacy default.yaml.
    deployment = cfg.to_dict()
    deployment['deploy'] = template.to_dict()['deploy']
    packaged_cfg = Config.from_dict(deployment)
    if target.exists():
        if (digest(target / 'best.pt') == training_hash
                and Config.load(target / 'config.yaml') == packaged_cfg
                and (target / 'package_manifest.json').exists()):
            ensure_ankle_reference(target, cfg, source)
            return target
        raise FileExistsError(f'Refusing to replace a different model package: {target}')
    destination.mkdir(parents=True, exist_ok=True)
    staging = destination / '.staging'
    staging.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=staging) as temporary:
        run = Path(temporary) / name
        run.mkdir()
        for filename in ('best.pt', 'scaler_bundle.pkl', 'model_meta.json',
                         'forecast_metrics.json', 'test_metrics.json', 'history.jsonl'):
            shutil.copy2(source / filename, run / filename)
        shutil.copy2(source / 'config.yaml', run / 'training_config.yaml')
        packaged_cfg.save(run)
        ensure_ankle_reference(run, cfg, source)
        # Only training-subject demographics are needed for unseen-user selection.
        demographics = pd.read_parquet(cfg.paths.demographics())
        demographics[demographics.subject.isin(cfg.data.split.train)].to_parquet(
            run / 'demographics.parquet', index=False)
        meta = json.loads((run / 'model_meta.json').read_text())
        meta.update(model_name=name, observation_augmentation=cfg.train.augment.timing.enabled,
                    training_augmentation=cfg.to_dict()['train']['augment'])
        (run / 'model_meta.json').write_text(json.dumps(meta, indent=2) + '\n')
        (run / 'display_name.txt').write_text(name + '\n')
        # Defaults support preview; live launcher re-exports for entered wearer.
        export(run, 72.0, subject_demo=(1.75, 72.0, 'M'), onnx=True)
        manifest = {
            'model_name': name,
            'source_run': str(source.resolve().relative_to(ROOT.resolve())),
            'checkpoint_sha256': training_hash,
            'prediction_horizon_ms': cfg.data.prediction_horizon_ms,
            'observation_augmentation': cfg.train.augment.timing.enabled,
            'target_semantics': 'biological ankle moment, not a learned motor action',
            'export_default_subject': {'height_m': 1.75, 'mass_kg': 72.0, 'gender': 'M'},
            'files_sha256': {p.name: digest(p) for p in sorted(run.iterdir()) if p.is_file()},
        }
        (run / 'package_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        run.rename(target)
    return target


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--experiment', required=True, type=Path)
    ap.add_argument('--destination', type=Path, default=ROOT / 'runs')
    ap.add_argument('--deploy-template', type=Path, default=ROOT / 'runs/deploy/config.yaml')
    args = ap.parse_args()
    template = Config.load(args.deploy_template)
    for source in sorted(args.experiment.iterdir()):
        if source.is_dir() and (source / 'best.pt').exists():
            print(package(source, args.destination, template), flush=True)


if __name__ == '__main__':
    main()
