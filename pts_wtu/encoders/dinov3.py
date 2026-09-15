"""Frozen DINOv3 + AdaptivePool 4×4 / view (plan §10–§11)."""
from __future__ import annotations

import torch
from torch import nn


class FrozenDINOv3(nn.Module):
    """Offline or online frozen visual encoder. Outputs (B, V, 16, D) tokens."""

    def __init__(self, ckpt_path: str, image_size: int = 256, pool: int = 4):
        super().__init__()
        self.ckpt_path = ckpt_path
        self.image_size = image_size
        self.pool = pool
        raise NotImplementedError("FrozenDINOv3: wire LaWAM / transformers backbone")

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
