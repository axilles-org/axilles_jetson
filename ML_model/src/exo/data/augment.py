"""Train-split augmentations on the z-scored ``(C, T)`` window.

Noise, IMU gain jitter, time-warp, channel dropout, plus deployment domain
randomisation (IMU rotation error, encoder offset, stance-edge jitter, an
"assistance felt" kinematics nudge). The latency DR lives in ``WindowDataset``
because it also shifts the target. See docs/DOMAIN_RANDOMIZATION.md.
"""
from __future__ import annotations

import math

import torch

from ..config import AugmentConfig

_ACCEL_AXES = ("Accel_X", "Accel_Y", "Accel_Z")
_GYRO_AXES = ("Gyro_X", "Gyro_Y", "Gyro_Z")


def _triad_indices(feature_names: list[str], seg: str, kind_axes) -> list[int] | None:
    """Indices of ``imu_<seg>_<axis>`` for the three axes, in X,Y,Z order."""
    idx = []
    for ax in kind_axes:
        name = f"imu_{seg}_{ax}"
        if name not in feature_names:
            return None
        idx.append(feature_names.index(name))
    return idx


def _small_rotation(max_deg: float) -> torch.Tensor:
    """Random 3x3 rotation with angle <= max_deg about a random axis."""
    axis = torch.randn(3)
    axis = axis / (axis.norm() + 1e-9)
    ang = math.radians(max_deg) * (torch.rand(1).item() * 2 - 1)
    c, s = math.cos(ang), math.sin(ang)
    x, y, z = axis
    K = torch.tensor([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return torch.eye(3) + s * K + (1 - c) * (K @ K)


class Augmenter:
    def __init__(self, cfg: AugmentConfig, feature_names: list[str]):
        self.cfg = cfg
        self.feature_names = feature_names
        self.imu_idx = torch.tensor(
            [i for i, n in enumerate(feature_names) if n.startswith("imu_")],
            dtype=torch.long,
        )
        self.stance_idx = feature_names.index("stance") if "stance" in feature_names else -1
        self.ankle_idx = (feature_names.index("gon_ankle_sagittal")
                          if "gon_ankle_sagittal" in feature_names else -1)
        # z-score params for de-normalising the channels that DR needs in
        # physical units (rotation, encoder offset). Set by the dataset.
        self.x_mean: torch.Tensor | None = None
        self.x_scale: torch.Tensor | None = None

        self._triads = []
        for seg in ("foot", "shank"):
            a = _triad_indices(feature_names, seg, _ACCEL_AXES)
            g = _triad_indices(feature_names, seg, _GYRO_AXES)
            if a:
                self._triads.append(torch.tensor(a))
            if g:
                self._triads.append(torch.tensor(g))

    def set_scaler(self, x_mean, x_scale) -> None:
        self.x_mean = torch.as_tensor(x_mean, dtype=torch.float32)
        self.x_scale = torch.as_tensor(x_scale, dtype=torch.float32)
        # the two z-scored levels the binary stance channel can take
        if self.stance_idx >= 0:
            m, s = float(self.x_mean[self.stance_idx]), float(self.x_scale[self.stance_idx])
            self._stance_lo = (0.0 - m) / s
            self._stance_hi = (1.0 - m) / s
            self._stance_mid = 0.5 * (self._stance_lo + self._stance_hi)

    def _snap_stance(self, row: torch.Tensor) -> torch.Tensor:
        """Snap a possibly-perturbed stance row back to its two z-scored levels."""
        if self.x_scale is None:
            return (row > 0.5).float()
        return torch.where(row >= self._stance_mid,
                           torch.full_like(row, self._stance_hi),
                           torch.full_like(row, self._stance_lo))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """x: (C, T) z-scored — returns a modified copy."""
        c = self.cfg
        if not c.enabled:
            return x
        out = x.clone()

        if c.imu_rotation_deg > 0 and self._triads:
            self._rotate_triads(out, c.imu_rotation_deg)

        if c.assist_perturb > 0:
            self._simulate_assistance(out, c.assist_perturb)

        if c.encoder_offset_rad > 0 and self.ankle_idx >= 0 and self.x_scale is not None:
            off = (torch.rand(1).item() * 2 - 1) * c.encoder_offset_rad
            out[self.ankle_idx] = out[self.ankle_idx] + off / self.x_scale[self.ankle_idx]

        if c.stance_jitter_samples > 0 and self.stance_idx >= 0:
            k = int(torch.randint(-c.stance_jitter_samples,
                                  c.stance_jitter_samples + 1, (1,)).item())
            if k:
                out[self.stance_idx] = torch.roll(out[self.stance_idx], k)

        if c.noise_std > 0:
            noise = torch.randn_like(out) * c.noise_std
            if self.stance_idx >= 0:
                noise[self.stance_idx] = 0.0             # keep stance exactly 0/1
            out = out + noise

        if c.imu_gain_jitter > 0 and len(self.imu_idx):
            gain = 1.0 + (torch.rand(len(self.imu_idx)) * 2 - 1) * c.imu_gain_jitter
            out[self.imu_idx] = out[self.imu_idx] * gain.unsqueeze(-1)

        if c.channel_dropout > 0:
            keep = (torch.rand(out.size(0)) >= c.channel_dropout).float().unsqueeze(-1)
            if self.stance_idx >= 0:
                keep[self.stance_idx] = 1.0
            out = out * keep

        if c.time_warp > 0:
            out = _time_warp(out, c.time_warp)

        if self.stance_idx >= 0:
            out[self.stance_idx] = self._snap_stance(out[self.stance_idx])

        return out

    def _rotate_triads(self, out: torch.Tensor, max_deg: float) -> None:
        """Apply one random small rotation per IMU accel/gyro triad, in physical
        units (de-z-score -> rotate -> re-z-score)."""
        if self.x_mean is None:
            return
        for tri in self._triads:
            m = self.x_mean[tri].unsqueeze(-1)
            s = self.x_scale[tri].unsqueeze(-1)
            phys = out[tri] * s + m                       # (3, T)
            R = _small_rotation(max_deg)
            out[tri] = ((R @ phys) - m) / s

    def _simulate_assistance(self, out: torch.Tensor, max_frac: float) -> None:
        """Nudge the window as if the exo were already assisting: in late stance,
        shrink the foot push-off gyro peak and bias the ankle toward dorsiflexion,
        scaled by a random assist level."""
        if self.x_scale is None:
            return
        a = torch.rand(1).item() * max_frac
        T = out.size(1)
        # late-stance mask ~ last 35 % of the window (push-off region)
        w = torch.zeros(T)
        w[int(0.55 * T):] = torch.linspace(0, 1, T - int(0.55 * T))

        foot_gyro = _triad_indices(self.feature_names, "foot", _GYRO_AXES)
        if foot_gyro:
            gi = torch.tensor(foot_gyro)
            out[gi] = out[gi] * (1.0 - a * w)             # reduce push-off spin
        if self.ankle_idx >= 0:
            # dorsiflexion is +; a fixed nudge scaled to z-units
            nudge = a * 0.08 / self.x_scale[self.ankle_idx]
            out[self.ankle_idx] = out[self.ankle_idx] + nudge * w


def _time_warp(x: torch.Tensor, max_frac: float) -> torch.Tensor:
    """Resample the window onto a slightly stretched/compressed time axis."""
    C, T = x.shape
    factor = 1.0 + (torch.rand(1).item() * 2 - 1) * max_frac
    src = torch.linspace(0, T - 1, steps=int(round(T * factor))).clamp(0, T - 1)
    lo = src.floor().long()
    hi = src.ceil().long()
    w = (src - lo).unsqueeze(0)
    warped = x[:, lo] * (1 - w) + x[:, hi] * w
    if warped.size(1) >= T:
        return warped[:, -T:].contiguous()
    pad = x[:, : T - warped.size(1)]
    return torch.cat([pad, warped], dim=1).contiguous()
