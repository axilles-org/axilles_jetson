"""Convert a predicted human ankle moment into a safe exo assist command.

The exo provides a fraction of the predicted biological moment, only during
stance, ramped on entry, zeroed immediately on exit, and bounded by torque and
slew-rate limits.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

from ..config import DeployConfig, SensorAdapterConfig
from .delay import PredictionDelay, validate_delay_ms


class StanceDetector:
    """Binary stance from the two foot FSRs."""

    def __init__(self, cfg: SensorAdapterConfig):
        self.heel_threshold = cfg.heel_fsr_threshold
        self.toe_threshold = cfg.toe_fsr_threshold

    def __call__(self, heel_fsr: float, toe_fsr: float) -> bool:
        return heel_fsr > self.heel_threshold or toe_fsr > self.toe_threshold


@dataclass
class AssistCommand:
    torque_nm: float
    stance: bool
    ramp: float           # 0..1 blend factor currently applied


class AssistanceController:
    def __init__(self, cfg: DeployConfig):
        self.delay = PredictionDelay(cfg.assistance_delay_ms)
        self.delayed_moment = 0.0
        if cfg.assistance_mode not in ("plantarflexion", "bidirectional"):
            raise ValueError("assistance_mode must be plantarflexion or bidirectional")
        for name in ("dorsiflexion_scale_ratio", "dorsiflexion_limit_nm"):
            value = getattr(cfg, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        self.mode = cfg.assistance_mode
        self.dorsi_ratio = cfg.dorsiflexion_scale_ratio
        self.dorsi_limit = cfg.dorsiflexion_limit_nm
        self.scale = cfg.assistance_scale
        self.torque_limit = cfg.torque_limit_nm
        self.rate_limit = cfg.rate_limit_nm_per_s
        self.ramp_in_s = max(cfg.ramp_in_s, 1e-3)
        self.ramp_out_s = max(cfg.ramp_out_s, 1e-3)
        self.plantarflexion_sign = cfg.plantarflexion_sign
        self.exo_command_sign = cfg.exo_command_sign
        self.stance_detector = StanceDetector(cfg.sensor_adapter)

        self.faulted = False
        self._ramp = 0.0
        self._last_command = 0.0
        # Delay requested at runtime, applied at the next swing sample.
        self.pending_delay_ms: float | None = None

    def request_delay(self, delay_ms: float) -> None:
        """Queue a new assistance delay. It takes effect at the next swing sample,
        so the stance in progress finishes on the old delay and every stance runs
        on a single delay value."""
        delay_ms = validate_delay_ms(delay_ms)
        self.pending_delay_ms = None if delay_ms == self.delay.delay_ms else delay_ms

    @property
    def target_delay_ms(self) -> float:
        """The delay the next stance will use (applied, or pending if queued)."""
        return self.delay.delay_ms if self.pending_delay_ms is None else self.pending_delay_ms

    def reset(self) -> None:
        self.delay.reset()
        self.delayed_moment = 0.0
        self.faulted = False
        self._ramp = 0.0
        self._last_command = 0.0

    def update(
        self,
        predicted_moment_nm_per_kg: float,
        subject_mass_kg: float,
        heel_fsr: float,
        toe_fsr: float,
        dt: float,
        *, stance: bool | None = None, enabled: bool = True,
        ready: bool = True, now: float | None = None,
    ) -> AssistCommand:
        stance = self.stance_detector(heel_fsr, toe_fsr) if stance is None else bool(stance)

        # NaN passes straight through the max/min/_clamp below, past the stance
        # gate and both limits, so stop here and stay off until reset().
        if self.faulted or not math.isfinite(
                predicted_moment_nm_per_kg * self.scale * subject_mass_kg):
            self.faulted = True
            self.delay.reset()
            self.delayed_moment = 0.0
            self._ramp = 0.0
            self._last_command = 0.0
            return AssistCommand(torque_nm=0.0, stance=stance, ramp=0.0)

        # A swing sample must not inherit torque from the previous stance.
        if not stance or not enabled or not ready:
            if not stance and self.pending_delay_ms is not None:
                # Stance is over; the next one starts on the new delay.
                self.delay.set_delay_ms(self.pending_delay_ms)
                self.pending_delay_ms = None
            self.delay.reset()
            self.delayed_moment = 0.0
            self._ramp = 0.0
            self._last_command = 0.0
            return AssistCommand(torque_nm=0.0, stance=stance, ramp=0.0)

        self.delayed_moment = self.delay.update(
            predicted_moment_nm_per_kg, time.perf_counter() if now is None else now)
        if not self.delay.ready:
            self._ramp = 0.0
            self._last_command = 0.0
            return AssistCommand(torque_nm=0.0, stance=stance, ramp=0.0)

        self._ramp = min(1.0, self._ramp + dt / self.ramp_in_s)
        # Positive here means plantarflexion, negative means dorsiflexion.
        signed_moment = self.delayed_moment * self.plantarflexion_sign
        if signed_moment >= 0.0:
            desired = min(self.scale * signed_moment * subject_mass_kg * self._ramp,
                          self.torque_limit)
        elif self.mode == "bidirectional":
            desired = -min(self.scale * self.dorsi_ratio * (-signed_moment)
                           * subject_mass_kg * self._ramp,
                           self.dorsi_limit, self.torque_limit)
        else:
            desired = 0.0
        desired *= self.exo_command_sign

        max_delta = self.rate_limit * dt
        command = _clamp(desired, self._last_command - max_delta, self._last_command + max_delta)
        self._last_command = command

        return AssistCommand(torque_nm=command, stance=stance, ramp=self._ramp)


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))
