"""Convert a predicted human ankle moment into a safe exo assist command.

The exo provides a fraction of the predicted biological moment, only during
stance, ramped at the stance transitions and bounded by hard torque and
slew-rate limits.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..config import DeployConfig, SensorAdapterConfig


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
        self.scale = cfg.assistance_scale
        self.torque_limit = cfg.torque_limit_nm
        self.rate_limit = cfg.rate_limit_nm_per_s
        self.ramp_in_s = max(cfg.ramp_in_s, 1e-3)
        self.ramp_out_s = max(cfg.ramp_out_s, 1e-3)
        self.plantarflexion_sign = cfg.plantarflexion_sign
        self.exo_command_sign = cfg.exo_command_sign
        self.stance_detector = StanceDetector(cfg.sensor_adapter)

        self._ramp = 0.0
        self._last_command = 0.0

    def reset(self) -> None:
        self._ramp = 0.0
        self._last_command = 0.0

    def update(
        self,
        predicted_moment_nm_per_kg: float,
        subject_mass_kg: float,
        heel_fsr: float,
        toe_fsr: float,
        dt: float,
    ) -> AssistCommand:
        stance = self.stance_detector(heel_fsr, toe_fsr)

        target_ramp = 1.0 if stance else 0.0
        step = dt / (self.ramp_in_s if stance else self.ramp_out_s)
        if target_ramp > self._ramp:
            self._ramp = min(target_ramp, self._ramp + step)
        else:
            self._ramp = max(target_ramp, self._ramp - step)

        # Magnitude of the plantarflexion moment the human is producing.
        plantarflexion = max(predicted_moment_nm_per_kg * self.plantarflexion_sign, 0.0)
        magnitude = min(
            self.scale * plantarflexion * subject_mass_kg * self._ramp,
            self.torque_limit,
        )
        desired = magnitude * self.exo_command_sign

        max_delta = self.rate_limit * dt
        command = _clamp(desired, self._last_command - max_delta, self._last_command + max_delta)
        self._last_command = command

        return AssistCommand(torque_nm=command, stance=stance, ramp=self._ramp)


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))
