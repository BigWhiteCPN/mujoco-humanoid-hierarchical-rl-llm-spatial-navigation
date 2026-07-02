"""Dataset splitting helpers shared by training and evaluation."""

from __future__ import annotations

import random
from pathlib import Path

from torch.utils.data import Subset, random_split


def split_dataset(dataset, val_fraction: float, seed: int, split_by: str = "episode"):
    """Return train/validation subsets using either episode or window splits."""
    if split_by == "episode":
        rng = random.Random(seed)
        episode_paths = sorted({item.path for item in dataset.index})
        rng.shuffle(episode_paths)
        val_episode_count = max(1, int(len(episode_paths) * val_fraction))
        if val_episode_count >= len(episode_paths):
            val_episode_count = max(1, len(episode_paths) - 1)
        val_paths = set(episode_paths[:val_episode_count])
        train_indices = [idx for idx, item in enumerate(dataset.index) if item.path not in val_paths]
        val_indices = [idx for idx, item in enumerate(dataset.index) if item.path in val_paths]
        if not train_indices or not val_indices:
            raise ValueError("Episode split produced an empty train or validation set")
        return Subset(dataset, train_indices), Subset(dataset, val_indices), {
            "split_by": split_by,
            "val_episode_paths": [str(Path(path)) for path in sorted(val_paths)],
            "train_windows": len(train_indices),
            "val_windows": len(val_indices),
        }

    if split_by != "window":
        raise ValueError(f"Unknown split_by={split_by!r}")
    val_len = max(1, int(len(dataset) * val_fraction))
    train_len = len(dataset) - val_len
    train_set, val_set = random_split(
        dataset,
        [train_len, val_len],
        generator=__import__("torch").Generator().manual_seed(seed),
    )
    return train_set, val_set, {
        "split_by": split_by,
        "train_windows": train_len,
        "val_windows": val_len,
    }

