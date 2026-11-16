"""Simulate asynchronous acquisition on a fixed decision-time grid.

This changes observations available at each decision, never the target or clock.
All source indices are <= their decision index. The source is already sampled
at input_rate_hz, so sub-grid acquisition times are conservatively rounded down.
There is no interpolation or invented high-frequency information.
"""
from __future__ import annotations

import numpy as np
import torch

from ..config import TimingAugmentConfig


def acquisition_indices(length: int, rate_hz: float, cfg: TimingAugmentConfig,
                        rng: np.random.Generator) -> np.ndarray:
    """Latest received source sample per decision; sample zero is initial history.

    A burst suppresses arrivals for its duration; held-value age also includes
    acquisition intervals and transport delay. This is not a freshness limiter.
    """
    decisions = np.arange(length, dtype=np.float64) / rate_hz
    period = 1.0 / rng.uniform(cfg.rate_min_hz, cfg.rate_max_hz)
    jitter = cfg.interval_jitter_fraction
    count = int(np.ceil(decisions[-1] / (period * (1 - jitter)))) + 2
    intervals = period * rng.uniform(1 - jitter, 1 + jitter, count)
    acquisition = np.concatenate(([0.0], np.cumsum(intervals)))
    arrival = acquisition + rng.uniform(0, cfg.delay_max_ms) / 1000.0
    keep = rng.random(len(acquisition)) >= cfg.packet_drop_probability
    if rng.random() < cfg.burst_probability:
        start = rng.uniform(0, decisions[-1])
        duration = rng.uniform(cfg.burst_min_ms, cfg.burst_max_ms) / 1000.0
        keep &= ~((arrival >= start) & (arrival < start + duration))
    arrival = arrival[keep]
    # Epsilon only counters floating-point roundoff at exact grid boundaries.
    sources = np.floor(acquisition[keep] * rate_hz + 1e-9).astype(np.int64)
    received = np.searchsorted(arrival, decisions + 1e-12, side="right") - 1
    result = np.zeros(length, dtype=np.int64)
    valid = received >= 0
    result[valid] = sources[received[valid]]
    return np.minimum(result, np.arange(length, dtype=np.int64))


class TimingAugmenter:
    """Perturb accel/gyro report streams and encoder; leave stance untouched.

    Axes within each report share a schedule. Different reports can have
    different rates/delays, as in the Jetson acquisition path. No age channels
    are appended: the 14-channel checkpoint interface stays unchanged.
    """
    def __init__(self, cfg: TimingAugmentConfig, feature_names: list[str], rate_hz: float):
        cfg.validate(rate_hz)
        self.cfg, self.rate_hz = cfg, rate_hz
        self.groups = []
        for segment in ("foot", "shank"):
            for report in ("Accel", "Gyro"):
                prefix = f"imu_{segment}_{report}_"
                indices = [i for i, name in enumerate(feature_names) if name.startswith(prefix)]
                if indices:
                    self.groups.append(indices)
        if "gon_ankle_sagittal" in feature_names:
            self.groups.append([feature_names.index("gon_ankle_sagittal")])

    def __call__(self, x: torch.Tensor, rng: np.random.Generator | None = None) -> torch.Tensor:
        if not self.cfg.enabled:
            return x
        if rng is None:
            # DataLoader seeds torch independently in each worker; no shared RNG.
            rng = np.random.default_rng(int(torch.randint(0, 2**32, ()).item()))
        if rng.random() >= self.cfg.probability:
            return x
        out = x.clone()
        for group in self.groups:
            indices = acquisition_indices(x.shape[-1], self.rate_hz, self.cfg, rng)
            out[group] = x[group][:, torch.as_tensor(indices, device=x.device)]
        return out
