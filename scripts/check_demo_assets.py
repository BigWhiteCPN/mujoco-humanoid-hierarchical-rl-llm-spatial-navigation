#!/usr/bin/env python3
"""Check that the files needed by the interactive demo are present."""

from __future__ import annotations

import argparse
from pathlib import Path


REQUIRED_FILES = {
    "robot_mjcf": "resources/mjcf/Linnxil_fifteen_angle_bs_copy_20260302_copy.xml",
    "low_level_policy": "models/policy_20251026.pt",
    "sac_navigation_policy": "models/sac_lidar_interrupted_good3_0.91.zip",
    "random_map_env": "visual_train/robot_visual_env_random_map.py",
    "demo_preview": "assets/demo-preview.png",
}

REQUIRED_DIRS = {
    "robot_meshes": "resources/meshes",
}


def check_assets(root: Path) -> tuple[list[str], list[str]]:
    missing: list[str] = []
    empty: list[str] = []

    for label, relative_path in REQUIRED_FILES.items():
        path = root / relative_path
        if not path.is_file():
            missing.append(f"{label}: {relative_path}")
            continue
        if path.stat().st_size == 0:
            empty.append(f"{label}: {relative_path}")

    for label, relative_path in REQUIRED_DIRS.items():
        path = root / relative_path
        if not path.is_dir():
            missing.append(f"{label}: {relative_path}")
            continue
        if not any(path.iterdir()):
            empty.append(f"{label}: {relative_path}")

    return missing, empty


def main() -> int:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Check bundled runtime assets for the MuJoCo navigation demo.")
    parser.add_argument("--root", type=Path, default=default_root, help="Project root to check.")
    args = parser.parse_args()

    root = args.root.resolve()
    missing, empty = check_assets(root)

    if not missing and not empty:
        print(f"OK: demo assets are present under {root}")
        return 0

    if missing:
        print("Missing files or directories:")
        for item in missing:
            print(f"  - {item}")

    if empty:
        print("Empty files or directories:")
        for item in empty:
            print(f"  - {item}")

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
