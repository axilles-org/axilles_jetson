"""Subject-index mapping for the model's per-subject embedding.

Training subjects each get their own embedding row. An unseen subject (every real
exo user) is mapped to the nearest training subject by cosine similarity over the
normalised demographic vector ``[height, weight, gender_male, gender_female]``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data.scalers import DemographicStats


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


@dataclass
class SubjectIndex:
    """Maps subject id -> embedding row index."""

    mapping: dict[str, int]
    num_training_subjects: int

    def __getitem__(self, subject_id: str) -> int:
        return self.mapping[subject_id]

    def indices(self, subject_ids: list[str]) -> list[int]:
        return [self.mapping[s] for s in subject_ids]

    @classmethod
    def build(
        cls,
        train_subjects: list[str],
        metadata_path: str,
        demo_stats: DemographicStats,
        unseen_subjects: list[str] | None = None,) -> SubjectIndex:
        train_sorted = sorted(set(train_subjects))
        mapping = {sid: i for i, sid in enumerate(train_sorted)}

        if unseen_subjects:
            vecs = _demographic_vectors(metadata_path, demo_stats)
            for sid in unseen_subjects:
                if sid in mapping:
                    continue
                nearest = max(train_sorted, key=lambda t: _cosine(vecs[sid], vecs[t]))
                mapping[sid] = mapping[nearest]

        return cls(mapping=mapping, num_training_subjects=len(train_sorted))


def _demographic_vectors(metadata_path: str, stats: DemographicStats) -> dict[str, np.ndarray]:
    df = pd.read_parquet(metadata_path).drop_duplicates("subject")
    return {
        row["subject"]: stats.vector(row["height_m"], row["weight_kg"], row["gender"])
        for _, row in df.iterrows()
    }
