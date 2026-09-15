"""Visual transition decoder D_V(F_{k-1}, u_k) → F_k (plan §19–§21)."""
from __future__ import annotations

import torch
from torch import nn


class VisualTransitionDecoder(nn.Module):
    def __init__(self, d_u: int = 512, d_vis: int = 256, n_layers: int = 4):
        super().__init__()
        self.d_u = d_u
        self.d_vis = d_vis

    def forward(self, f_prev: torch.Tensor, u_k: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("VisualTransitionDecoder.forward")
