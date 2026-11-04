"""Paired clean/corrupted held-out window check for named model packages.

The same seeded timing corruption is used for a given trial/input endpoint across
models. This is a sampled window stress test, not a continuous hardware replay.
No motor or deployment controller is constructed.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from exo.config import Config
from exo.data.dataset import WindowDataset
from exo.data.timing_augment import TimingAugmenter
from exo.training.evaluate import Evaluator, _metrics


@torch.inference_mode()
def evaluate(run: Path, profile: Config, device: str, every: int, seed: int) -> dict:
    cfg = Config.load(run / 'config.yaml')
    evaluator = Evaluator(cfg, str(run / 'best.pt'), device=device)
    dataset = WindowDataset(cfg.data, cfg.paths.processed(), evaluator.scalers, split='test')
    timing = replace(profile.train.augment.timing, enabled=True, probability=1.0)
    aug = TimingAugmenter(timing, dataset.feature_names, cfg.data.ingest.target_rate_hz)
    indices = list(range(0, len(dataset), every))
    truth, clean, corrupted = [], [], []
    for start in range(0, len(indices), 128):
        items = indices[start:start+128]
        xs, bad, ys, subjects, masses = [], [], [], [], []
        for i in items:
            x, y, trial, subject = dataset[i]
            endpoint = int(dataset.index.entries[i][2])
            key = f'{seed}:{trial}:{endpoint}'.encode()
            rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8], 'little'))
            xs.append(x)
            bad.append(aug(x, rng))
            ys.append(float(y[0, -1]))
            subjects.append(evaluator.subject_index[subject])
            masses.append(evaluator.subject_mass[subject])
        subject_ids = torch.tensor(subjects, dtype=torch.long, device=evaluator.device)
        def predict(windows):
            pred = evaluator.model(torch.stack(windows).to(evaluator.device), subject_ids)
            return pred[:, 0, -1].cpu().numpy()
        def physical(z):
            return (np.asarray(z) * evaluator._y_scale + evaluator._y_mean) / np.asarray(masses)
        truth.extend(physical(ys))
        clean.extend(physical(predict(xs)))
        corrupted.extend(physical(predict(bad)))
    reference = np.asarray(truth)
    return dict(model=run.name, prediction_horizon_ms=cfg.data.prediction_horizon_ms,
                windows=len(indices), split='test', every_nth_window=every, seed=seed,
                timing_profile=asdict(timing),
                clean=_metrics(np.asarray(clean), reference),
                timing_corrupted=_metrics(np.asarray(corrupted), reference))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--profile', type=Path, default=ROOT / 'configs/anticipation/model_obs_aug_subject_embedding_current_action.yaml')
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
    parser.add_argument('--every', type=int, default=10)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.every < 1:
        parser.error('--every must be positive')
    if args.output.exists():
        parser.error('output already exists; choose a new result path')
    torch.set_num_threads(2)
    profile = Config.load(args.profile.resolve())
    results = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for run in args.runs:
        result = evaluate(run.resolve(), profile, args.device, args.every, args.seed)
        results[run.name] = result
        args.output.write_text(json.dumps(results, indent=2) + '\n')
        print(run.name, 'clean:', result['clean'], 'corrupted:', result['timing_corrupted'], flush=True)


if __name__ == '__main__':
    main()
