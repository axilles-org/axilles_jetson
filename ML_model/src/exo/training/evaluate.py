"""Held-out evaluation: reconstruct full torque signals and score them.

Two views are produced per trial:

* **windowed** — overlapping window predictions are averaged, giving a smooth
  full-length signal (useful for plots and offline analysis);
* **streaming** — only the last timestep of each window is kept, exactly matching
  what the deployed controller consumes.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import torch

from ..config import Config
from ..data.dataset import WindowDataset
from ..data.demographics import subject_mass
from ..data.scalers import ScalerBundle
from ..models import TCN
from .metrics import r2_score
from .runtime_utils import select_device
from .subject_embedding import SubjectIndex


@dataclass
class TrialResult:
    name: str
    subject: str
    pred_windowed: np.ndarray      # (T,) N·m/kg
    pred_streaming: np.ndarray     # (T,) N·m/kg, NaN before the first full window
    truth: np.ndarray             # (T,) N·m/kg
    valid: np.ndarray             # bool (T,)


@dataclass
class EvalReport:
    per_trial: dict[str, dict[str, float]] = field(default_factory=dict)
    per_subject: dict[str, dict[str, float]] = field(default_factory=dict)
    overall_windowed: dict[str, float] = field(default_factory=dict)
    overall_streaming: dict[str, float] = field(default_factory=dict)


def _metrics(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    err = pred - truth
    p2p = max(truth.max() - truth.min(), 1e-6)
    return {
        "rmse_nm_per_kg": float(np.sqrt(np.mean(err ** 2))),
        "mae_nm_per_kg": float(np.mean(np.abs(err))),
        "normalized_mae": float(np.mean(np.abs(err)) / p2p),
        "r2": r2_score(pred, truth),
    }


class Evaluator:
    def __init__(self, cfg: Config, checkpoint: str, device: str | None = None):
        self.cfg = cfg
        self.device = select_device(device or cfg.train.device)
        self.scalers = ScalerBundle.load(cfg.paths.processed())
        self.subject_mass = subject_mass(cfg.paths.demographics())
        self.subject_index = SubjectIndex.build(
            cfg.data.split.train, str(cfg.paths.demographics()),
            self.scalers.demographics,
            unseen_subjects=cfg.data.split.val + cfg.data.split.test,
        )
        self.model = TCN.from_config(
            cfg.model, in_channels=cfg.data.num_features, out_channels=1,
            num_subjects=self.subject_index.num_training_subjects,
        ).to(self.device)
        self.model.load_state_dict(torch.load(checkpoint, map_location=self.device))
        self.model.eval()

        self._y_mean = float(self.scalers.scaler_y.mean_[0])
        self._y_scale = float(self.scalers.scaler_y.scale_[0])

    @torch.inference_mode()
    def run(self, split: str = "test") -> tuple[EvalReport, list[TrialResult]]:
        dataset = WindowDataset(self.cfg.data, self.cfg.paths.processed(), self.scalers, split=split)
        results = [self._reconstruct(dataset, tid)
                   for tid in range(len(dataset.index.trial_names))
                   if _has_windows(dataset, tid)]
        return _aggregate(results), results

    def _reconstruct(self, dataset: WindowDataset, tid: int) -> TrialResult:
        name = dataset.index.trial_names[tid]
        subject = name.split("_")[0]
        segs = [(i, int(s), int(e)) for i, (t, s, e) in enumerate(dataset.index.entries) if t == tid]

        feat_all, targ_all, _ = dataset._load(tid)
        length = len(targ_all)
        acc = np.zeros(length, dtype=np.float64)
        cnt = np.zeros(length, dtype=np.float64)
        stream = np.full(length, np.nan, dtype=np.float64)

        sidx = torch.tensor([self.subject_index[subject]], dtype=torch.long, device=self.device)
        batch = 128
        for start in range(0, len(segs), batch):
            chunk = segs[start:start + batch]
            xs = torch.stack([dataset[i][0] for i, _, _ in chunk]).to(self.device)
            out = self.model(xs, sidx.expand(len(chunk))).squeeze(1).cpu().numpy()  # (B, T)
            for row, (_, s, e) in enumerate(chunk):
                acc[s:e + 1] += out[row]
                cnt[s:e + 1] += 1.0
                stream[e] = out[row, -1]

        cnt[cnt == 0] = 1.0
        mass = self.subject_mass[subject]

        def unscale(z: np.ndarray) -> np.ndarray:
            return (z * self._y_scale + self._y_mean) / mass

        raw_target = np.asarray(targ_all[:, 0], dtype=np.float64)
        pred_windowed = unscale(acc / cnt)
        pred_streaming = unscale(stream)
        truth = raw_target / mass                  # npz targets are already physical N·m
        valid = ~np.isnan(raw_target)

        return TrialResult(name, subject, pred_windowed, pred_streaming, truth, valid)


def _has_windows(dataset: WindowDataset, tid: int) -> bool:
    return bool((dataset.index.entries[:, 0] == tid).any())


def _aggregate(results: list[TrialResult]) -> EvalReport:
    report = EvalReport()
    by_subject: dict[str, list[TrialResult]] = defaultdict(list)

    for r in results:
        v = r.valid
        report.per_trial[r.name] = {
            "windowed": _metrics(r.pred_windowed[v], r.truth[v]),
            "streaming": _metrics(*_streaming_pair(r)),
        }
        by_subject[r.subject].append(r)

    for subject, trials in by_subject.items():
        pw = np.concatenate([t.pred_windowed[t.valid] for t in trials])
        tw = np.concatenate([t.truth[t.valid] for t in trials])
        report.per_subject[subject] = _metrics(pw, tw)

    all_pw = np.concatenate([r.pred_windowed[r.valid] for r in results])
    all_tw = np.concatenate([r.truth[r.valid] for r in results])
    report.overall_windowed = _metrics(all_pw, all_tw)

    stream_pairs = [_streaming_pair(r) for r in results]
    report.overall_streaming = _metrics(
        np.concatenate([p for p, _ in stream_pairs]),
        np.concatenate([t for _, t in stream_pairs]),
    )
    return report


def _streaming_pair(r: TrialResult) -> tuple[np.ndarray, np.ndarray]:
    m = r.valid & ~np.isnan(r.pred_streaming)
    return r.pred_streaming[m], r.truth[m]
