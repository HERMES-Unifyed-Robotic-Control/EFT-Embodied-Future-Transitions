"""Relative robot-state transition loss L_S (plan §25)."""
from __future__ import annotations

import torch


def state_transition_loss(pred, target, valid=None, lambdas: dict | None = None) -> torch.Tensor:
    raise NotImplementedError
