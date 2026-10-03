"""Causal Temporal Convolutional Network for ankle-torque prediction.

Input  : ``(batch, in_channels, T)``  z-scored sensor features.
Output : ``(batch, out_channels, T)`` z-scored torque per timestep; inference uses
         the last timestep only.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..config import ModelConfig
from .layers import TemporalBlock, dilated_tcn_receptive_field


class TCN(nn.Module):
    """Dilated causal TCN with an optional learnable per-subject embedding.

    With ``use_subject_embedding``, a ``(num_subjects, emb_dim)`` embedding is
    broadcast along time and concatenated to the input channels. For an unseen
    subject, pass the nearest training subject's index
    (see ``exo.training.subject_embedding.SubjectIndex``).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_channels: list[int],
        kernel_size: int = 5,
        activation: str = "ReLU",
        norm_type: str = "WeightNorm",
        dropout_type: str = "Spatial",
        dropout: float = 0.15,
        l1_reg: float = 0.0,
        l2_reg: float = 0.0,
        use_subject_embedding: bool = False,
        num_subjects: int = 0,
        emb_dim: int = 4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.use_subject_embedding = use_subject_embedding

        if use_subject_embedding:
            if num_subjects <= 0:
                raise ValueError("num_subjects must be > 0 when use_subject_embedding=True")
            self.subject_embedding = nn.Embedding(num_subjects, emb_dim)
            nn.init.normal_(self.subject_embedding.weight, 0.0, 0.01)
            tcn_in = in_channels + emb_dim
        else:
            tcn_in = in_channels

        blocks = []
        prev = tcn_in
        for i, width in enumerate(num_channels):
            blocks.append(TemporalBlock(
                prev, width, kernel_size, dilation=2 ** i, dropout=dropout,
                activation=activation, norm_type=norm_type, dropout_type=dropout_type,
            ))
            prev = width
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Conv1d(num_channels[-1], out_channels, kernel_size=1)

        self._kernel_size = kernel_size
        self._num_blocks = len(num_channels)

    @classmethod
    def from_config(
        cls, cfg: ModelConfig, in_channels: int, out_channels: int, num_subjects: int = 0
    ) -> TCN:
        return cls(
            in_channels=in_channels,
            out_channels=out_channels,
            num_channels=list(cfg.num_channels),
            kernel_size=cfg.kernel_size,
            activation=cfg.activation,
            norm_type=cfg.norm_type,
            dropout_type=cfg.dropout_type,
            dropout=cfg.dropout,
            l1_reg=cfg.l1_reg,
            l2_reg=cfg.l2_reg,
            use_subject_embedding=cfg.use_subject_embedding,
            num_subjects=num_subjects,
            emb_dim=cfg.emb_dim,
        )

    def forward(self, x: torch.Tensor, subject_idx: torch.Tensor | None = None) -> torch.Tensor:
        if self.use_subject_embedding:
            if subject_idx is None:
                raise ValueError("subject_idx is required when use_subject_embedding=True")
            emb = self.subject_embedding(subject_idx).unsqueeze(-1).expand(-1, -1, x.size(2))
            x = torch.cat([x, emb], dim=1)
        return self.head(self.tcn(x))

    def regularization_loss(self) -> torch.Tensor:
        device = next(self.parameters()).device
        if self.l1_reg == 0.0 and self.l2_reg == 0.0:
            return torch.zeros((), device=device)
        l1 = torch.zeros((), device=device)
        l2 = torch.zeros((), device=device)
        for name, param in self.named_parameters():
            if "weight" not in name:
                continue
            if self.l1_reg > 0:
                l1 = l1 + param.abs().sum()
            if self.l2_reg > 0:
                l2 = l2 + param.pow(2).sum()
        return self.l1_reg * l1 + self.l2_reg * l2

    @property
    def receptive_field(self) -> int:
        return dilated_tcn_receptive_field(self._num_blocks, self._kernel_size)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
