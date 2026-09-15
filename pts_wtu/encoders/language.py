"""Frozen text encoder for instruction L (plan §15)."""
from __future__ import annotations

import torch
from torch import nn


class FrozenLanguageEncoder(nn.Module):
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", d_model: int = 512):
        super().__init__()
        self.model_name = model_name
        self.d_model = d_model

    def forward(self, text: list[str]) -> torch.Tensor:
        """→ H_L tokens (B, T_l, D)."""
        raise NotImplementedError("FrozenLanguageEncoder.forward")
