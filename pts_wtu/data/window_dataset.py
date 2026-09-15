"""History + future window samples (plan §6–§8).

Input X_t = {I_{t-M:t}, S_{t-M:t}, L}; targets Y_t for supervision only.
Drop windows with t+H > T_episode. Prefer Moving:Static ≈ 4:1.
"""
from __future__ import annotations

from torch.utils.data import Dataset


class PredictiveWindowDataset(Dataset):
    """RoboCasa predictive window dataset."""

    def __init__(self, config_path: str, split: str = "train"):
        self.config_path = config_path
        self.split = split
        raise NotImplementedError("PredictiveWindowDataset: build after window index export")

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> dict:
        raise NotImplementedError
