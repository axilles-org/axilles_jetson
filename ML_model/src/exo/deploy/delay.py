"""Timestamp-based delay of model outputs, independent of control-loop rate."""
from collections import deque
import math


class PredictionDelay:
    def __init__(self, delay_ms: float = 0.0):
        if not math.isfinite(delay_ms) or not 0 <= delay_ms <= 1000:
            raise ValueError("assistance_delay_ms must be finite and between 0 and 1000")
        self.delay_ms = float(delay_ms)
        self._samples = deque()
        self.reset()

    def reset(self) -> None:
        self._samples.clear()
        self.ready = False
        self.age_ms = 0.0

    def update(self, value: float, now: float) -> float:
        """Hold the latest sample at or before now - delay; zero during fill.

        ``now`` is a monotonic timestamp at prediction completion. No interpolation
        or frame-count assumption: irregular sampling may add up to one source
        sample interval beyond the requested delay. Memory retains only the delay
        window plus its most recent eligible sample.
        """
        if self._samples and now < self._samples[-1][0]:
            self.reset()
        self._samples.append((now, value))
        cutoff = now - self.delay_ms / 1000.0
        while len(self._samples) > 1 and self._samples[1][0] <= cutoff + 1e-9:
            self._samples.popleft()
        at, delayed = self._samples[0]
        self.ready = at <= cutoff + 1e-9
        self.age_ms = max(0.0, (now - at) * 1000.0) if self.ready else 0.0
        return delayed if self.ready else 0.0
