"""Turn the Parquet dataset into a processed, z-score-ready cache.

Output (per cache key, under ``paths.processed_dir``)::

    <subject>_<trial>.npz     features (physical units) + target + column names
    scaler_bundle.pkl         fit on training subjects only
    manifest.json             per-file hashes + provenance

The cache key is derived from the ingest config + dataset revision + code version,
so a changed setting transparently produces a new cache instead of a stale one.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..config import Config
from .raw_ingest import RawTrialReader
from .scalers import ScalerBundle

_CODE_VERSION = "1"


def cache_key(cfg: Config, dataset_revision: str) -> str:
    payload = json.dumps({
        "ingest": asdict(cfg.data.ingest),
        "target_rate": cfg.data.ingest.target_rate_hz,
        "dataset_revision": dataset_revision,
        "code_version": _CODE_VERSION,
    }, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def _hash_file(path: Path) -> str:
    h = hashlib.sha1()
    h.update(path.read_bytes())
    return h.hexdigest()


def is_cached(processed_dir: Path) -> bool:
    manifest = processed_dir / "manifest.json"
    if not manifest.exists() or not (processed_dir / "scaler_bundle.pkl").exists():
        return False
    data = json.loads(manifest.read_text())
    for entry in data["trials"]:
        f = processed_dir / f"{entry['name']}.npz"
        if not f.exists() or _hash_file(f) != entry["sha1"]:
            return False
    return True


def run(cfg: Config, dataset_root: Path, out_dir: Path, dataset_revision: str,
        subjects: list[str] | None = None, verbose: bool = True) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    reader = RawTrialReader(dataset_root, cfg.data.ingest)

    wanted = set(subjects or cfg.data.split.all_subjects())
    known = set(cfg.data.split.all_subjects())
    subject_list = [s for s in reader.subjects() if s in wanted and s in known]
    train_subjects = set(cfg.data.split.train)

    train_trials, manifest = [], []
    t0 = time.time()

    for subject in subject_list:
        for trial in reader.trials(subject):
            name = f"{subject}_{trial}"
            tr = reader.read(subject, trial)
            if tr is None or tr.n_samples < cfg.data.min_segment_length:
                if verbose:
                    print(f"  skip {name}")
                continue

            npz = out_dir / f"{name}.npz"
            np.savez(
                npz,
                features=tr.features.to_numpy(np.float32),
                target=tr.target.to_numpy(np.float32),
                feature_columns=np.array(list(tr.features.columns)),
                target_column=np.array([tr.target.columns[0]]),
            )
            manifest.append({
                "name": name, "subject": subject, "n_samples": tr.n_samples,
                "split": _split_of(subject, cfg), "sha1": _hash_file(npz),
            })
            if subject in train_subjects:
                train_trials.append(tr)
            if verbose:
                print(f"  {name:<34} {tr.n_samples:>7d} samples")

    if not train_trials:
        raise RuntimeError("no training trials ingested; cannot fit scalers")

    bundle = ScalerBundle.fit(train_trials, reader.root / "metadata.parquet", cfg.data.split.train)
    bundle.save(out_dir)

    (out_dir / "manifest.json").write_text(json.dumps({
        "cache_key": cache_key(cfg, dataset_revision),
        "dataset_revision": dataset_revision,
        "code_version": _CODE_VERSION,
        "ingest_config": asdict(cfg.data.ingest),
        "feature_columns": bundle.feature_columns,
        "n_trials": len(manifest),
        "trials": manifest,
    }, indent=2))

    if verbose:
        print(f"\n{len(manifest)} trials, scalers on {len(train_trials)} train trials, "
              f"{time.time() - t0:.1f}s -> {out_dir}")
    return out_dir


def _split_of(subject: str, cfg: Config) -> str:
    if subject in cfg.data.split.train:
        return "train"
    if subject in cfg.data.split.val:
        return "val"
    if subject in cfg.data.split.test:
        return "test"
    return "unknown"
