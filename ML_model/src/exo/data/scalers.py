"""Fitted normalisation, bundled into one artifact.

``ScalerBundle`` holds:
  * ``scaler_x`` — StandardScaler over the superset of ingested feature columns,
  * ``scaler_y`` — StandardScaler over the target (ankle moment, N·m),
  * demographic mean/std for height and weight, for the subject-embedding cold-start.

All three are fit on training subjects only, so there is no leakage. Everything
downstream (dataset, trainer, JIT export) loads this one file.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

_ARTIFACT = "scaler_bundle.pkl"


@dataclass
class DemographicStats:
    height_mean: float
    height_std: float
    weight_mean: float
    weight_std: float

    def vector(self, height_m: float, weight_kg: float, gender: str) -> np.ndarray:
        """Normalised [height, weight, gender_male, gender_female]."""
        g = gender.strip().upper()
        return np.array([
            (height_m - self.height_mean) / self.height_std,
            (weight_kg - self.weight_mean) / self.weight_std,
            1.0 if g == "M" else 0.0,
            1.0 if g == "F" else 0.0,
        ], dtype=np.float32)


@dataclass
class ScalerBundle:
    scaler_x: StandardScaler
    scaler_y: StandardScaler
    feature_columns: list[str]
    target_column: str
    demographics: DemographicStats

    # -- fitting ------------------------------------------------------
    @classmethod
    def fit(
        cls,
        trials: list,                       # list[raw_ingest.Trial], training subjects only
        metadata_path: str | Path,          # dataset metadata.parquet
        train_subjects: list[str],
    ) -> "ScalerBundle":
        if not trials:
            raise ValueError("no training trials given to ScalerBundle.fit")

        feat_cols = list(trials[0].features.columns)
        target_col = trials[0].target.columns[0]

        X = pd.concat([t.features[feat_cols] for t in trials], ignore_index=True)
        y = pd.concat([t.target for t in trials], ignore_index=True).dropna()

        sx = StandardScaler().fit(X.to_numpy(np.float64))
        sy = StandardScaler().fit(y.to_numpy(np.float64))

        meta = pd.read_parquet(metadata_path).drop_duplicates("subject")
        tr = meta[meta["subject"].isin(train_subjects)]
        demo = DemographicStats(
            height_mean=float(tr["height_m"].mean()), height_std=float(tr["height_m"].std()),
            weight_mean=float(tr["weight_kg"].mean()), weight_std=float(tr["weight_kg"].std()),
        )
        return cls(sx, sy, feat_cols, target_col, demo)

    # -- transforms -------------------------------------------------
    def transform_features(self, arr: np.ndarray) -> np.ndarray:
        return self.scaler_x.transform(arr).astype(np.float32)

    def transform_target(self, arr: np.ndarray) -> np.ndarray:
        return self.scaler_y.transform(arr).astype(np.float32)

    def subset_x(self, columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """(mean, scale) of ``scaler_x`` restricted to ``columns`` — for baking a
        feature-set-specific input scaler into the JIT."""
        idx = [self.feature_columns.index(c) for c in columns]
        return self.scaler_x.mean_[idx].astype(np.float32), self.scaler_x.scale_[idx].astype(np.float32)

    # -- io -----------------------------------------------------------
    def save(self, out_dir: str | Path) -> Path:
        """Persist as plain arrays so loading needs only numpy/joblib."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / _ARTIFACT
        joblib.dump({
            "x_mean": self.scaler_x.mean_, "x_scale": self.scaler_x.scale_,
            "y_mean": self.scaler_y.mean_, "y_scale": self.scaler_y.scale_,
            "feature_columns": self.feature_columns,
            "target_column": self.target_column,
            "demographics": self.demographics.__dict__,
        }, path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ScalerBundle":
        p = Path(path)
        if p.is_dir():
            p = p / _ARTIFACT
        d = joblib.load(p)
        sx = StandardScaler()
        sx.mean_, sx.scale_ = np.asarray(d["x_mean"]), np.asarray(d["x_scale"])
        sx.var_, sx.n_features_in_ = sx.scale_ ** 2, len(sx.mean_)
        sy = StandardScaler()
        sy.mean_, sy.scale_ = np.asarray(d["y_mean"]), np.asarray(d["y_scale"])
        sy.var_, sy.n_features_in_ = sy.scale_ ** 2, len(sy.mean_)
        return cls(sx, sy, list(d["feature_columns"]), d["target_column"],
                   DemographicStats(**d["demographics"]))
