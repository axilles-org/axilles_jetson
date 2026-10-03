"""Parquet ingest, cache keying, and demographics from metadata."""
import numpy as np

from exo.config import replace
from exo.data import ingest
from exo.data.demographics import subject_mass
from exo.data.raw_ingest import RawTrialReader


def test_reader_lists_subjects_and_trials(cfg):
    dataset = cfg.paths.dataset()
    if not (dataset / "metadata.parquet").exists():
        import pytest
        pytest.skip("local Parquet dataset not available")
    reader = RawTrialReader(dataset, cfg.data.ingest)
    subjects = reader.subjects()
    assert set(cfg.data.split.train).issubset(set(subjects))
    trials = reader.trials(subjects[0])
    assert trials and all("__" not in t for t in trials)


def test_trial_alignment_and_stance(cfg):
    dataset = cfg.paths.dataset()
    if not (dataset / "metadata.parquet").exists():
        import pytest
        pytest.skip("local Parquet dataset not available")
    reader = RawTrialReader(dataset, cfg.data.ingest)
    subj = cfg.data.split.train[0]
    trial = reader.trials(subj)[0]
    t = reader.read(subj, trial)
    assert len(t.features) == len(t.target)
    stance = t.features["stance"].to_numpy()
    assert set(np.unique(stance)).issubset({0.0, 1.0})
    assert 0.4 < stance.mean() < 0.8  # plausible walking stance fraction


def test_cache_key_changes_with_ingest_config(cfg):
    k1 = ingest.cache_key(cfg, "rev-a")
    k2 = ingest.cache_key(replace(cfg, **{"data.ingest.target_rate_hz": 200}), "rev-a")
    k3 = ingest.cache_key(cfg, "rev-b")
    assert k1 != k2 and k1 != k3


def test_subject_mass_from_metadata(cfg):
    meta = cfg.paths.demographics()
    if not meta.exists():
        import pytest
        pytest.skip("metadata not available")
    masses = subject_mass(meta)
    assert all(40 < m < 120 for m in masses.values())
    assert set(cfg.data.split.all_subjects()).issubset(masses)
