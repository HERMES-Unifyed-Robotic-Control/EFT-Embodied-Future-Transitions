"""Stage A objective: L_V + λ_S L_S (plan §29)."""
from __future__ import annotations

import torch

from pts_wtu.losses.state import state_transition_loss
from pts_wtu.losses.visual import visual_transition_loss


def stage_a_loss(outputs: dict, batch: dict, lambda_s: float = 1.0) -> dict[str, torch.Tensor]:
    raise NotImplementedError("stage_a_loss")
