"""Batch collation for PredictiveWindowDataset."""
from __future__ import annotations


def collate_windows(batch: list[dict]) -> dict:
    raise NotImplementedError("pts_wtu.data.collate.collate_windows")
