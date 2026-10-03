from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "ELU": nn.ELU,
    "GELU": nn.GELU,
    "ReLU": nn.ReLU,
    "Swish": nn.SiLU,
    "Tanh": nn.Tanh,
}


def make_activation(name: str) -> nn.Module:
    try:
        return _ACTIVATIONS[name]()
    except KeyError:
        raise ValueError(f"unknown activation {name!r}; choose from {sorted(_ACTIVATIONS)}") from None


def make_norm1d(kind: str, num_channels: int) -> nn.Module:
    """Norm layer for ``(B, C, T)`` tensors. ``WeightNorm`` returns Identity because
    it is applied on the convolution weights, not the activations."""
    if kind == "BatchNorm":
        return nn.BatchNorm1d(num_channels)
    
    if kind == "LayerNorm":
        return nn.GroupNorm(1, num_channels)
    
    if kind == "WeightNorm":
        return nn.Identity()
    
    raise ValueError(f"norm_type must be BatchNorm|LayerNorm|WeightNorm, got {kind!r}")


class SpatialDropout1d(nn.Module):
    """Dropout that zeroes whole channels of a ``(B, C, T)`` tensor."""

    def __init__(self, p: float = 0.15):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        mask = torch.ones(x.size(0), x.size(1), 1, device=x.device, dtype=x.dtype)
        return x * F.dropout(mask, p=self.p, training=True)


def make_dropout1d(kind: str, p: float) -> nn.Module:
    if kind == "Spatial":
        return SpatialDropout1d(p)
    if kind == "ElementWise":
        return nn.Dropout(p)
    raise ValueError(f"dropout_type must be Spatial|ElementWise, got {kind!r}")


class Chomp1d(nn.Module):
    """Trims the right-hand padding so a padded convolution stays causal."""

    def __init__(self, size: int):
        super().__init__()
        self.size = size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : -self.size].contiguous() if self.size > 0 else x


class TemporalBlock(nn.Module):
    """Two dilated causal convolutions with a residual connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        activation: str,
        norm_type: str,
        dropout_type: str,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        use_weight_norm = norm_type == "WeightNorm"

        def conv(cin: int, cout: int) -> nn.Module:
            layer = nn.Conv1d(cin, cout, kernel_size, padding=padding, dilation=dilation)
            return weight_norm(layer) if use_weight_norm else layer

        self.block1 = nn.Sequential(
            conv(in_channels, out_channels), Chomp1d(padding),
            make_norm1d(norm_type, out_channels), make_activation(activation),
            make_dropout1d(dropout_type, dropout),
        )
        self.block2 = nn.Sequential(
            conv(out_channels, out_channels), Chomp1d(padding),
            make_norm1d(norm_type, out_channels), make_activation(activation),
            make_dropout1d(dropout_type, dropout),
        )
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        )
        self.out_activation = make_activation(activation)
        self._init_weights(use_weight_norm)

    def _init_weights(self, use_weight_norm: bool) -> None:
        for module in (*self.block1, *self.block2):
            if isinstance(module, nn.Conv1d):
                weight = module.weight_v if use_weight_norm and hasattr(module, "weight_v") else module.weight
                nn.init.normal_(weight, 0.0, 0.01)
        if self.residual is not None:
            nn.init.normal_(self.residual.weight, 0.0, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block2(self.block1(x))
        skip = x if self.residual is None else self.residual(x)
        return self.out_activation(out + skip)


def dilated_tcn_receptive_field(num_blocks: int, kernel_size: int) -> int:
    """Receptive field in timesteps for a stack of ``TemporalBlock`` with dilation ``2**i``."""
    rf = 1
    for i in range(num_blocks):
        rf += 2 * (kernel_size - 1) * (2 ** i)
    return rf
