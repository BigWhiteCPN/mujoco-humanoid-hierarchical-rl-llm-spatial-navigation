"""Dataset loader for bridge training episodes.

Expected episode format:

    episodes/episode_000001.npz

with arrays:

    maps: [T, C, H, W] float32
    state: [T, state_dim] float32
    memory: [T, memory_dim] float32
    task: [T, 3] int64, columns are intent/target/constraint ids
    skill: [T] int64
    candidates: [T, K, candidate_dim] float32
    candidate_mask: [T, K] bool
    event: [T] int64
    replan: [T] int64
    success: [T] float32
    stuck: [T] float32
    target_found: [T] float32
    cost: [T] float32, normalized to roughly [0, 1]
    info_gain: [T] float32
    candidate_score_target: [T, K] float32
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .constants import EVENT_TO_ID, REPLAN_TO_ID
from .schema import REQUIRED_KEYS


FAILURE_EVENT_IDS = {
    EVENT_TO_ID["navigation_stuck"],
    EVENT_TO_ID["path_invalidated"],
    EVENT_TO_ID["low_information_gain"],
    EVENT_TO_ID["need_replan"],
}
FAILURE_REPLAN_IDS = {
    REPLAN_TO_ID["switch_subgoal"],
    REPLAN_TO_ID["ask_llm_replan"],
    REPLAN_TO_ID["declare_unreachable"],
}


@dataclass(frozen=True)
class EpisodeIndex:
    path: Path
    start: int


class BridgeEpisodeDataset(Dataset):
    """Windowed dataset over saved bridge episodes."""

    def __init__(self, data_dir: str | Path, sequence_len: int = 16, stride: int = 4, failure_horizon: int = 8):
        self.data_dir = Path(data_dir)
        self.sequence_len = int(sequence_len)
        self.stride = int(stride)
        self.failure_horizon = max(0, int(failure_horizon))
        if self.sequence_len <= 0:
            raise ValueError("sequence_len must be positive")
        if self.stride <= 0:
            raise ValueError("stride must be positive")

        episode_dir = self.data_dir / "episodes"
        search_root = episode_dir if episode_dir.exists() else self.data_dir
        self.episode_paths = sorted(search_root.glob("*.npz"))
        if not self.episode_paths:
            raise FileNotFoundError(f"No episode .npz files found under {search_root}")

        self.index: list[EpisodeIndex] = []
        self._lengths: dict[Path, int] = {}
        self.state_dim = 0
        for path in self.episode_paths:
            with np.load(path) as data:
                missing = REQUIRED_KEYS - set(data.files)
                if missing:
                    missing_text = ", ".join(sorted(missing))
                    raise ValueError(f"{path} is missing required arrays: {missing_text}")
                length = int(data["maps"].shape[0])
                self.state_dim = max(self.state_dim, int(data["state"].shape[-1]))
            self._lengths[path] = length
            if length < self.sequence_len:
                continue
            for start in range(0, length - self.sequence_len + 1, self.stride):
                self.index.append(EpisodeIndex(path=path, start=start))

        if not self.index:
            raise ValueError(
                f"No windows of length {self.sequence_len} could be built from {len(self.episode_paths)} episodes"
            )

        self._cache_path: Path | None = None
        self._cache: dict[str, np.ndarray] | None = None

    @staticmethod
    def _fit_last_dim(array: np.ndarray, target_dim: int) -> np.ndarray:
        current_dim = int(array.shape[-1])
        if current_dim == target_dim:
            return array
        if current_dim > target_dim:
            return array[..., :target_dim]
        padded_shape = (*array.shape[:-1], target_dim)
        padded = np.zeros(padded_shape, dtype=array.dtype)
        padded[..., :current_dim] = array
        return padded

    def _failure_risk(self, event: np.ndarray, replan: np.ndarray, start: int, end: int) -> np.ndarray:
        failure = np.isin(event, list(FAILURE_EVENT_IDS)) | np.isin(replan, list(FAILURE_REPLAN_IDS))
        risk = np.zeros((end - start,), dtype=np.float32)
        for out_idx, step in enumerate(range(start, end)):
            horizon_end = min(event.shape[0], step + self.failure_horizon + 1)
            risk[out_idx] = float(np.any(failure[step:horizon_end]))
        return risk

    def __len__(self) -> int:
        return len(self.index)

    def _load_episode(self, path: Path) -> dict[str, np.ndarray]:
        if self._cache_path == path and self._cache is not None:
            return self._cache
        with np.load(path) as data:
            episode = {key: data[key] for key in data.files}
        self._cache_path = path
        self._cache = episode
        return episode

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        idx = self.index[item]
        episode = self._load_episode(idx.path)
        s = idx.start
        e = s + self.sequence_len

        batch = {
            "maps": torch.from_numpy(episode["maps"][s:e]).float(),
            "state": torch.from_numpy(self._fit_last_dim(episode["state"][s:e], self.state_dim)).float(),
            "memory": torch.from_numpy(episode["memory"][s:e]).float(),
            "task": torch.from_numpy(episode["task"][s:e]).long(),
            "skill": torch.from_numpy(episode["skill"][s:e]).long(),
            "candidates": torch.from_numpy(episode["candidates"][s:e]).float(),
            "candidate_mask": torch.from_numpy(episode["candidate_mask"][s:e]).bool(),
            "event": torch.from_numpy(episode["event"][s:e]).long(),
            "replan": torch.from_numpy(episode["replan"][s:e]).long(),
            "success": torch.from_numpy(episode["success"][s:e]).float(),
            "stuck": torch.from_numpy(episode["stuck"][s:e]).float(),
            "target_found": torch.from_numpy(episode["target_found"][s:e]).float(),
            "cost": torch.from_numpy(episode["cost"][s:e]).float(),
            "info_gain": torch.from_numpy(episode["info_gain"][s:e]).float(),
            "candidate_score_target": torch.from_numpy(episode["candidate_score_target"][s:e]).float(),
            "failure_risk": torch.from_numpy(self._failure_risk(episode["event"], episode["replan"], s, e)).float(),
        }
        return batch


def infer_shapes(dataset: BridgeEpisodeDataset) -> dict[str, int]:
    """Infer model dimensions from the first dataset sample."""
    sample = dataset[0]
    return {
        "map_channels": int(sample["maps"].shape[1]),
        "map_size": int(sample["maps"].shape[-1]),
        "state_dim": int(sample["state"].shape[-1]),
        "memory_dim": int(sample["memory"].shape[-1]),
        "candidate_dim": int(sample["candidates"].shape[-1]),
        "max_candidates": int(sample["candidates"].shape[-2]),
    }
