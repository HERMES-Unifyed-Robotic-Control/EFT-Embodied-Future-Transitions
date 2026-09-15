"""Concatenate C_t = [H_V; H_S; H_L] (plan §16)."""
from __future__ import annotations

import torch
from torch import nn


class ContextFusion(nn.Module):
    def forward(self, h_v: torch.Tensor, h_s: torch.Tensor, h_l: torch.Tensor) -> torch.Tensor:
        return torch.cat([h_v, h_s, h_l], dim=1)
