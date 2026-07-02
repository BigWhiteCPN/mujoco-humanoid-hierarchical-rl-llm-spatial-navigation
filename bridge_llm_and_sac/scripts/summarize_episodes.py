#!/usr/bin/env python3
"""Summarize bridge episode label and shape distributions."""

from __future__ import annotations

try:
    from scripts._bootstrap import ensure_project_root
except ModuleNotFoundError:
    from _bootstrap import ensure_project_root

ensure_project_root()

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from bridge.constants import EVENT_TYPES, REPLAN_ACTIONS


def episode_files(path: Path) -> list[Path]:
    root = path / "episodes" if (path / "episodes").exists() else path
    return sorted(root.glob("*.npz"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize bridge episode files.")
    parser.add_argument("path")
    args = parser.parse_args()

    files = episode_files(Path(args.path))
    if not files:
        raise SystemExit(f"No .npz episodes found under {args.path}")

    event_counts: Counter[int] = Counter()
    replan_counts: Counter[int] = Counter()
    state_dims: Counter[int] = Counter()
    steps = 0
    guard_triggers = 0
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            steps += int(data["event"].shape[0])
            state_dims[int(data["state"].shape[-1])] += 1
            event_counts.update(data["event"].astype(int).tolist())
            replan_counts.update(data["replan"].astype(int).tolist())
        meta_path = path.with_suffix(".json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            guard_triggers += len(meta.get("progress_guard_triggers", []))

    print(f"episodes={len(files)} steps={steps} state_dims={dict(sorted(state_dims.items()))}")
    print(f"progress_guard_triggers={guard_triggers}")
    print("events")
    for idx, count in sorted(event_counts.items()):
        print(f"  {EVENT_TYPES[int(idx)]}: {int(count)}")
    print("replans")
    for idx, count in sorted(replan_counts.items()):
        print(f"  {REPLAN_ACTIONS[int(idx)]}: {int(count)}")


if __name__ == "__main__":
    main()
