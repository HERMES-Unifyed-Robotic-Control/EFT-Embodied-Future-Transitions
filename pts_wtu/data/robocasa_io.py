"""RoboCasa episode IO (plan §5).

Load multi-view RGB, proprio/state, action, and language for an episode.
Implementation TBD — mirror paths under dataset.index_root.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass
class EpisodeRef:
    task: str
    episode_index: int
    parquet: Path
    videos: dict[str, Path]


def iter_episodes(index_root: str | Path, tasks: list[str], split_paths: dict[str, str]) -> Iterator[EpisodeRef]:
    raise NotImplementedError("pts_wtu.data.robocasa_io.iter_episodes")


def load_states(parquet: Path):
    raise NotImplementedError("pts_wtu.data.robocasa_io.load_states")


def load_actions(parquet: Path):
    raise NotImplementedError("pts_wtu.data.robocasa_io.load_actions")


def load_prompt(parquet: Path) -> str:
    raise NotImplementedError("pts_wtu.data.robocasa_io.load_prompt")
