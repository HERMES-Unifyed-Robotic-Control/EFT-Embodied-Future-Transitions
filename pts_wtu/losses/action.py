"""Action grounding loss L_A (plan §28). Stage B only."""
from __future__ import annotations

import torch


def action_loss(pred, target, valid=None) -> torch.Tensor:
    raise NotImplementedError
