#!/usr/bin/env python3
"""Validate bridge episode files before training."""

from __future__ import annotations

try:
    from scripts._bootstrap import ensure_project_root
except ModuleNotFoundError:
    from _bootstrap import ensure_project_root

ensure_project_root()

import argparse
from pathlib import Path

import numpy as np

from bridge.schema import REQUIRED_KEYS


def find_episode_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    episode_dir = path / "episodes"
    root = episode_dir if episode_dir.exists() else path
    return sorted(root.glob("*.npz"))


def validate_file(path: Path) -> list[str]:
    errors: list[str] = []
    with np.load(path) as data:
        missing = REQUIRED_KEYS - set(data.files)
        if missing:
            errors.append(f"missing arrays: {', '.join(sorted(missing))}")
            return errors

        t = data["maps"].shape[0]
        for key in REQUIRED_KEYS:
            if data[key].shape[0] != t:
                errors.append(f"{key} first dimension {data[key].shape[0]} != maps first dimension {t}")
        if data["maps"].ndim != 4:
            errors.append(f"maps must be [T, C, H, W], got {data['maps'].shape}")
        if data["state"].ndim != 2:
            errors.append(f"state must be [T, state_dim], got {data['state'].shape}")
        if data["memory"].ndim != 2:
            errors.append(f"memory must be [T, memory_dim], got {data['memory'].shape}")
        if data["task"].shape[-1] != 3:
            errors.append(f"task last dimension must be 3, got {data['task'].shape}")
        if data["candidates"].ndim != 3:
            errors.append(f"candidates must be [T, K, candidate_dim], got {data['candidates'].shape}")
        if data["candidate_mask"].shape != data["candidate_score_target"].shape:
            errors.append(
                "candidate_mask and candidate_score_target shape mismatch: "
                f"{data['candidate_mask'].shape} vs {data['candidate_score_target'].shape}"
            )
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate bridge episode npz files.")
    parser.add_argument("path")
    args = parser.parse_args()

    files = find_episode_files(Path(args.path))
    if not files:
        raise SystemExit(f"No .npz episode files found under {args.path}")

    failed = 0
    for path in files:
        errors = validate_file(path)
        if errors:
            failed += 1
            print(f"[FAIL] {path}")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"[OK] {path}")

    if failed:
        raise SystemExit(f"{failed}/{len(files)} episode files failed validation")
    print(f"Validated {len(files)} episode files")


if __name__ == "__main__":
    main()
