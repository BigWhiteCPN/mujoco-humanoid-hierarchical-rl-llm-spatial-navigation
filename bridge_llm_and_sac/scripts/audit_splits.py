#!/usr/bin/env python3
"""Audit train/validation class coverage for bridge datasets."""

from __future__ import annotations

try:
    from scripts._bootstrap import ensure_project_root
except ModuleNotFoundError:
    from _bootstrap import ensure_project_root

ensure_project_root()

import argparse

import torch

from bridge.constants import EVENT_TYPES, REPLAN_ACTIONS
from bridge.dataset import BridgeEpisodeDataset
from bridge.splits import split_dataset


def class_counts(dataset: BridgeEpisodeDataset, subset, key: str, num_classes: int) -> list[int]:
    counts = torch.zeros(num_classes, dtype=torch.long)
    for idx in subset.indices:
        sample = dataset[idx]
        counts += torch.bincount(sample[key].reshape(-1), minlength=num_classes)
    return [int(item) for item in counts.tolist()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit bridge split class coverage.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--sequence-len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--split-by", choices=["episode", "window"], default="episode")
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--seed-end", type=int, default=50)
    args = parser.parse_args()

    dataset = BridgeEpisodeDataset(args.data_dir, sequence_len=args.sequence_len, stride=args.stride)
    for seed in range(args.seed_start, args.seed_end + 1):
        train_set, val_set, _ = split_dataset(dataset, args.val_fraction, seed, args.split_by)
        train_events = class_counts(dataset, train_set, "event", len(EVENT_TYPES))
        val_events = class_counts(dataset, val_set, "event", len(EVENT_TYPES))
        train_replans = class_counts(dataset, train_set, "replan", len(REPLAN_ACTIONS))
        rare_ok = train_events[EVENT_TYPES.index("target_candidate_found")] > 0 and train_events[
            EVENT_TYPES.index("path_invalidated")
        ] > 0
        print(
            f"seed={seed} rare_ok={rare_ok} "
            f"train_target={train_events[EVENT_TYPES.index('target_candidate_found')]} "
            f"val_target={val_events[EVENT_TYPES.index('target_candidate_found')]} "
            f"train_path={train_events[EVENT_TYPES.index('path_invalidated')]} "
            f"val_path={val_events[EVENT_TYPES.index('path_invalidated')]} "
            f"train_switch={train_replans[REPLAN_ACTIONS.index('switch_subgoal')]} "
            f"val_switch={class_counts(dataset, val_set, 'replan', len(REPLAN_ACTIONS))[REPLAN_ACTIONS.index('switch_subgoal')]}"
        )


if __name__ == "__main__":
    main()
