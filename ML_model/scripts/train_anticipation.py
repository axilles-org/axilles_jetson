"""Train current/future-moment TCNs with resolved configs, provenance and status.

python3 scripts/train_anticipation.py
python3 scripts/train_anticipation.py --prepare-only
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout, redirect_stderr
import multiprocessing
import torch
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from exo.config import Config, replace
from exo.data import ingest
from exo.training.evaluate import Evaluator
from exo.training.trainer import Trainer

DEFAULT_CONFIGS = [ROOT / f'configs/anticipation/tcn_{h}ms.yaml' for h in (50, 100, 150)]


def model_name(cfg: Config) -> str:
    prefix = 'model_obs_aug' if cfg.train.augment.enabled and cfg.train.augment.timing.enabled else 'model'
    subject = '_subject_embedding' if cfg.model.use_subject_embedding else ''
    sensors = f'_{cfg.data.feature_set}' if cfg.data.feature_set in ('no_gyro', 'no_imu') else ''
    h = cfg.data.prediction_horizon_ms
    target = f'future_action_{h:g}' if h else 'current_action'
    return f'{prefix}{subject}{sensors}_{target}'


def save_json(path: Path, value) -> None:
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def prepare(cfg: Config) -> None:
    """Reuse only a cache with matching preprocessing AND scaler training split."""
    root = cfg.paths.processed()
    manifest = root / 'manifest.json'
    if manifest.exists():
        meta = json.loads(manifest.read_text())
        expected = ingest.cache_key(cfg, cfg.paths.hub.revision)
        train = {row['subject'] for row in meta['trials'] if row['split'] == 'train'}
        if meta['cache_key'] != expected or train != set(cfg.data.split.train):
            raise ValueError(f'Cache configuration/split mismatch at {root}; choose a new processed_dir')
        if not ingest.is_cached(root):
            raise ValueError(f'Cache checksum check failed at {root}; choose a new processed_dir')
        return
    if root.exists() and any(root.iterdir()):
        raise ValueError(f'Refusing to overwrite incomplete cache at {root}; choose a new processed_dir')
    if not cfg.paths.demographics().is_file():
        raise FileNotFoundError(f'Missing dataset: {cfg.paths.dataset()}')
    ingest.run(cfg, cfg.paths.dataset(), root, cfg.paths.hub.revision)



def train_one(cfg: Config, output: Path, parallel: bool = False) -> dict:
    """One independent process owns this model, checkpoint and log."""
    h = cfg.data.prediction_horizon_ms
    name = model_name(cfg)
    run_dir = output / name
    run_dir.mkdir(exist_ok=False)
    if parallel:
        torch.set_num_threads(2)
    state = dict(state='training', model_name=name, horizon_ms=h, checkpoint=None)
    save_json(run_dir / 'status.json', state)
    with (run_dir / 'training.log').open('w', buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
        try:
            print(f'{name}: {cfg.train.epochs} epochs, {cfg.data.prediction_horizon_samples} future samples', flush=True)
            trainer = Trainer(cfg, run_dir, use_wandb=False)
            state['device'] = str(trainer.device)
            save_json(run_dir / 'status.json', state)
            best = trainer.fit()
            del trainer
            state.update(state='evaluating', checkpoint=str(best))
            save_json(run_dir / 'status.json', state)
            evaluator = Evaluator(cfg, str(best))
            report, _ = evaluator.run(split='test')
            result = dict(model_name=name, horizon_ms=h, horizon_samples=cfg.data.prediction_horizon_samples,
                          checkpoint=str(best), metrics=asdict(report))
            save_json(run_dir / 'forecast_metrics.json', result)
            state.update(state='completed', finished_at_utc=datetime.now(timezone.utc).isoformat())
            save_json(run_dir / 'status.json', state)
            return result
        except BaseException as exc:
            state.update(state='failed', error=repr(exc), traceback=traceback.format_exc())
            save_json(run_dir / 'status.json', state)
            raise


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--configs', nargs='+', type=Path, default=DEFAULT_CONFIGS)
    ap.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'])
    ap.add_argument('--epochs', type=int)
    ap.add_argument('--output', type=Path, help='new, empty experiment directory')
    ap.add_argument('--prepare-only', action='store_true')
    ap.add_argument('--parallel', action='store_true', help='one independent training process per config')
    args = ap.parse_args()
    configs = [Config.load(p) for p in args.configs]
    if args.epochs is not None and args.epochs <= 0:
        ap.error('--epochs must be positive')
    overrides = {}
    if args.device:
        overrides['train.device'] = args.device
    if args.epochs is not None:
        overrides['train.epochs'] = args.epochs
    configs = [replace(cfg, **overrides) for cfg in configs]
    horizons = [cfg.data.prediction_horizon_ms for cfg in configs]
    if len(horizons) != len(set(horizons)) or any(h < 0 for h in horizons):
        ap.error('configs must have distinct nonnegative prediction horizons')
    reference = configs[0].to_dict()
    reference['data'].pop('prediction_horizon_ms')
    reference['train'].pop('wandb_run_name')
    for cfg in configs:
        if not cfg.data.ingest.causal_inputs:
            ap.error('anticipation experiments require causal_inputs: true')
        comparable = cfg.to_dict()
        comparable['data'].pop('prediction_horizon_ms')
        comparable['train'].pop('wandb_run_name')
        if comparable != reference:
            ap.error('configs must differ only in prediction_horizon_ms and run name')
    if args.prepare_only:
        prepare(configs[0])
        return
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_UTC')
    output = args.output or configs[0].paths.runs() / stamp
    output.mkdir(parents=True, exist_ok=False)
    sources = sorted({*ROOT.joinpath('src/exo').rglob('*.py'), Path(__file__).resolve()})
    save_json(output / 'source_manifest.json', {
        'sources_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in sources},
        'resolved_configs': {model_name(cfg): cfg.to_dict() for cfg in configs},
        'target_semantics': 'biological ankle moment; action in the name is shorthand, not motor torque',
    })
    status = dict(started_at_utc=datetime.now(timezone.utc).isoformat(), state='preparing',
                  horizons_ms=horizons, completed=[], current=None,
                  target='future biological ankle moment (not a learned motor action)',
                  input_preprocessing='causal IIR decimation; offline target filtering unchanged')
    save_json(output / 'status.json', status)
    results = {}
    try:
        prepare(configs[0])
        manifest = configs[0].paths.processed() / 'manifest.json'
        status['processed_manifest_sha256'] = hashlib.sha256(manifest.read_bytes()).hexdigest()
        status.update(state='training', execution='parallel' if args.parallel else 'sequential')
        save_json(output / 'status.json', status)
        def record(cfg, result):
            name = model_name(cfg)
            results[name] = result
            save_json(output / 'summary.json', results)
            status['completed'].append(name)
            save_json(output / 'status.json', status)
            print(f'Completed {name}: {result["metrics"]["overall_streaming"]}', flush=True)
        if args.parallel:
            with ProcessPoolExecutor(max_workers=len(configs),
                                     mp_context=multiprocessing.get_context('spawn')) as pool:
                jobs = {pool.submit(train_one, cfg, output, True): cfg for cfg in configs}
                for future in as_completed(jobs):
                    record(jobs[future], future.result())
        else:
            for cfg in configs:
                record(cfg, train_one(cfg, output))
        status.update(state='completed', current=None, finished_at_utc=datetime.now(timezone.utc).isoformat())
        save_json(output / 'status.json', status)
    except BaseException as exc:
        status.update(state='failed', error=repr(exc), traceback=traceback.format_exc())
        save_json(output / 'status.json', status)
        raise


if __name__ == '__main__':
    main()
