"""State history encoder E_hist^S (plan §14)."""
from __future__ import annotations

import torch
from torch import nn


class HistoryStateEncoder(nn.Module):
    def __init__(self, state_dim: int = 16, d_model: int = 512, n_layers: int = 2):
        super().__init__()
        self.state_dim = state_dim
        self.d_model = d_model

    def forward(self, s_hist: torch.Tensor) -> torch.Tensor:
        """s_hist: (B, N_h, D_s) → H_S tokens."""
        raise NotImplementedError("HistoryStateEncoder.forward")
