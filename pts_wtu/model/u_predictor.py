"""Temporal-query U predictor: CrossAttn(Q_U, C_t) → U[1:4] (plan §16–§17).

Future observations must NEVER enter this module.
"""
from __future__ import annotations

import torch
from torch import nn


class TemporalQueryUPredictor(nn.Module):
    def __init__(self, d_model: int = 512, K: int = 4, n_layers: int = 2, n_heads: int = 8):
        super().__init__()
        self.K = K
        self.d_model = d_model
        self.queries = nn.Parameter(torch.zeros(1, K, d_model))
        nn.init.trunc_normal_(self.queries, std=0.02)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """context (B, T_c, D) → U (B, K, D)."""
        raise NotImplementedError("TemporalQueryUPredictor.forward")
