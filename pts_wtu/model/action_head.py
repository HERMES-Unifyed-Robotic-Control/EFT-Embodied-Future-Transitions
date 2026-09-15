"""Action head D_A(U, S_t) → A (plan §27). Used in Stage B only."""
from __future__ import annotations

import torch
from torch import nn


class ActionHead(nn.Module):
    def __init__(self, d_u: int = 512, state_dim: int = 16, action_dim: int = 12, H: int = 30):
        super().__init__()
        self.action_dim = action_dim
        self.H = H

    def forward(self, u: torch.Tensor, s_t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("ActionHead.forward")
