"""The single feature transform used by BOTH training and deployment.

Sharing this one class is what removes train/serve skew: whatever column ordering
and per-channel handling training uses, deployment uses the exact same code path.

Contract:
  input  — a DataFrame (or 2-D array + column list) of ingested, physical-unit
           features that includes every name in ``feature_names`` (a superset is
           fine; extra columns are ignored).
  output — float32 ``(T, C)`` array, columns in ``feature_names`` order, ready to
           be z-scored by ``ScalerBundle`` (training) or fed to the JIT which
           z-scores internally (deployment).

The ``stance`` channel is expected to already be 0/1 (produced at ingest for
training data, produced by ``SensorAdapter`` from the exo FSRs at deployment).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class FeaturePipeline:
    def __init__(self, feature_names: list[str]):
        self.feature_names = list(feature_names)

    def transform(self, data, columns: list[str] | None = None) -> np.ndarray:
        if isinstance(data, pd.DataFrame):
            df = data
        else:
            if columns is None:
                raise ValueError("columns must be given when data is an array")
            df = pd.DataFrame(np.asarray(data), columns=list(columns))

        missing = [c for c in self.feature_names if c not in df.columns]
        if missing:
            raise ValueError(f"FeaturePipeline: input is missing columns {missing}")

        out = df[self.feature_names].to_numpy(np.float32, copy=True)

        # sanity: stance must be binary
        if "stance" in self.feature_names:
            s = out[:, self.feature_names.index("stance")]
            uniq = np.unique(s[~np.isnan(s)])
            if not np.all(np.isin(uniq, (0.0, 1.0))):
                raise ValueError(
                    f"FeaturePipeline: 'stance' channel is not 0/1 (values {uniq[:5]}...). "
                    f"Binarize it before this stage."
                )
        return out

    @property
    def num_features(self) -> int:
        return len(self.feature_names)
