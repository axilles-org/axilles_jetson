"""Causal Temporal Convolutional Network for ankle-torque prediction.

Input  : ``(batch, in_channels, T)``  z-scored sensor features.
Output : ``(batch, out_channels, T)`` z-scored torque per timestep; inference uses
         the last timestep only.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..config import ModelConfig
from .layers import TemporalBlock, dilated_tcn_receptive_field, make_activation


class TCN(nn.Module):
    """Dilated causal TCN with an optional learnable per-subject embedding.

    With ``use_subject_embedding``, a ``(num_subjects, emb_dim)`` embedding is
    broadcast along time and concatenated to the input channels. For an unseen
    subject, pass the nearest training subject's index
    (see ``exo.training.subject_embedding.SubjectIndex``).

    ``arch`` picks how context enters the model, everything causal per timestep:
      * ``tcn``  — trunk + 1x1 head.
      * ``film`` — each trunk block output becomes ``h * (1 + gamma) + beta``.
        ``film_source="subject"``: a linear layer maps the per-subject embedding to
        (gamma, beta) and the embedding is not concatenated to the input.
        ``film_source="context"``: a small causal TCN (dilations 8, 16, 32, so it
        spans the whole 3 s window, longer than the trunk) reads the same inputs and
        emits per-timestep (gamma, beta).
      * ``moe``  — K expert heads on the trunk features, mixed by a softmax gate.
        A load-balancing penalty keeps every expert in use.
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
        arch: str = "tcn",
        film_source: str = "subject",
        film_channels: int = 32,
        moe_experts: int = 4,
        moe_hidden: int = 32,
        moe_balance: float = 0.01,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.use_subject_embedding = use_subject_embedding

        film_on_subject = arch == "film" and film_source == "subject"
        if film_on_subject and not use_subject_embedding:
            raise ValueError("film_source='subject' conditions on the subject embedding; "
                             "set use_subject_embedding=True")
        if use_subject_embedding:
            if num_subjects <= 0:
                raise ValueError("num_subjects must be > 0 when use_subject_embedding=True")
            self.subject_embedding = nn.Embedding(num_subjects, emb_dim)
            nn.init.normal_(self.subject_embedding.weight, 0.0, 0.01)
            tcn_in = in_channels if film_on_subject else in_channels + emb_dim
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

        if arch not in ("tcn", "film", "moe"):
            raise ValueError(f"unknown arch {arch!r}")
        self.arch = arch
        self.moe_balance = moe_balance
        self._aux = None
        width = num_channels[-1]
        self.film_source = film_source
        if arch == "film":
            if film_source == "subject":
                self.film = nn.Linear(emb_dim, 2 * sum(num_channels))
            elif film_source == "context":
                self.context = nn.Sequential(*[
                    TemporalBlock(tcn_in if i == 0 else film_channels, film_channels, kernel_size,
                                  dilation=2 ** (i + 3), dropout=dropout, activation=activation,
                                  norm_type=norm_type, dropout_type=dropout_type)
                    for i in range(3)])                              # dilations 8, 16, 32
                self.film = nn.Conv1d(film_channels, 2 * sum(num_channels), kernel_size=1)
            else:
                raise ValueError(f"unknown film_source {film_source!r}")
            nn.init.zeros_(self.film.weight)                         # start as the plain TCN
            nn.init.zeros_(self.film.bias)
        if arch == "moe":
            k = moe_experts
            self.gate = nn.Conv1d(width, k, kernel_size=1)
            self.experts = nn.Sequential(
                nn.Conv1d(width, k * moe_hidden, kernel_size=1), make_activation(activation),
                nn.Conv1d(k * moe_hidden, k * out_channels, kernel_size=1, groups=k),
            )
        else:
            self.head = nn.Conv1d(width, out_channels, kernel_size=1)

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
            arch=cfg.arch,
            film_source=cfg.film_source,
            film_channels=cfg.film_channels,
            moe_experts=cfg.moe_experts,
            moe_hidden=cfg.moe_hidden,
            moe_balance=cfg.moe_balance,
        )

    def forward(self, x: torch.Tensor, subject_idx: torch.Tensor | None = None) -> torch.Tensor:
        if self.use_subject_embedding:
            if subject_idx is None:
                raise ValueError("subject_idx is required when use_subject_embedding=True")
            if self.arch == "film" and self.film_source == "subject":
                return self._film(x, self.film(self.subject_embedding(subject_idx)).unsqueeze(-1))
            emb = self.subject_embedding(subject_idx).unsqueeze(-1).expand(-1, -1, x.size(2))
            x = torch.cat([x, emb], dim=1)
        if self.arch == "film":
            return self._film(x, self.film(self.context(x)))
        h = self.tcn(x)
        if self.arch == "moe":
            return self._mix(h)
        return self.head(h)

    def _film(self, x: torch.Tensor, gb: torch.Tensor) -> torch.Tensor:
        """``gb``: (B, 2*sum(C), 1) per subject or (B, 2*sum(C), T) per timestep."""
        h, offset = x, 0
        for block in self.tcn:
            h = block(h)
            w = h.size(1)
            gamma, beta = gb[:, offset:offset + w], gb[:, offset + w:offset + 2 * w]
            h = h * (1 + gamma) + beta
            offset += 2 * w
        return self.head(h)

    def _mix(self, h: torch.Tensor) -> torch.Tensor:
        gate = torch.softmax(self.gate(h), dim=1)                    # (B, K, T)
        k = gate.size(1)
        experts = self.experts(h).view(h.size(0), k, self.out_channels, h.size(2))
        if self.training:
            usage = gate.mean(dim=(0, 2))                            # 1/K each when balanced
            self._aux = k * (usage ** 2).sum() - 1.0
        return (gate.unsqueeze(2) * experts).sum(dim=1)

    def gate_weights(self, x: torch.Tensor, subject_idx: torch.Tensor | None = None) -> torch.Tensor:
        """(B, K, T) gate probabilities of the MoE model, for analysis."""
        if self.use_subject_embedding:
            emb = self.subject_embedding(subject_idx).unsqueeze(-1).expand(-1, -1, x.size(2))
            x = torch.cat([x, emb], dim=1)
        return torch.softmax(self.gate(self.tcn(x)), dim=1)

    def regularization_loss(self) -> torch.Tensor:
        device = next(self.parameters()).device
        aux = torch.zeros((), device=device)
        if self.arch == "moe" and self.training and self._aux is not None:
            aux = self.moe_balance * self._aux
        if self.l1_reg == 0.0 and self.l2_reg == 0.0:
            return aux
        l1 = torch.zeros((), device=device)
        l2 = torch.zeros((), device=device)
        for name, param in self.named_parameters():
            if "weight" not in name:
                continue
            if self.l1_reg > 0:
                l1 = l1 + param.abs().sum()
            if self.l2_reg > 0:
                l2 = l2 + param.pow(2).sum()
        return aux + self.l1_reg * l1 + self.l2_reg * l2

    @property
    def receptive_field(self) -> int:
        return dilated_tcn_receptive_field(self._num_blocks, self._kernel_size)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
