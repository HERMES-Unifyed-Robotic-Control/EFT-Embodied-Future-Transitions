"""State / action / relative robot-transition normalization (plan §13, §23)."""
from __future__ import annotations

from pathlib import Path


def load_stats(path: str | Path) -> dict:
    raise NotImplementedError("pts_wtu.data.normalize.load_stats")


def apply_state(x, stats: dict):
    raise NotImplementedError


def apply_action(x, stats: dict):
    raise NotImplementedError


def apply_relative_state(x, stats: dict):
    raise NotImplementedError
