"""History visual Temporal Transformer E_hist^V (plan §12)."""
from __future__ import annotations

import torch
from torch import nn


class HistoryVisualEncoder(nn.Module):
    def __init__(self, d_model: int = 512, n_layers: int = 4, n_heads: int = 8, n_views: int = 2):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_views = n_views

    def forward(self, f_hist: torch.Tensor) -> torch.Tensor:
        """f_hist: (B, N_h, V, T_tok, D) → H_V token sequence."""
        raise NotImplementedError("HistoryVisualEncoder.forward")
