"""Visual future transition loss L_V (plan §20–§21)."""
from __future__ import annotations

import torch


def visual_transition_loss(pred, target, weight=None) -> torch.Tensor:
    raise NotImplementedError
