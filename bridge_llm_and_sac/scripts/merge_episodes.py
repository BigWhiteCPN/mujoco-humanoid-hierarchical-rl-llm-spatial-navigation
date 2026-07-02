#!/usr/bin/env python3
"""Merge multiple bridge episode directories into one training dataset."""

from __future__ import annotations

try:
    from scripts._bootstrap import ensure_project_root
except ModuleNotFoundError:
    from _bootstrap import ensure_project_root

ensure_project_root()

import argparse
import json
import shutil
from pathlib import Path


def source_episode_dir(path: Path) -> Path:
    episode_dir = path / "episodes"
    return episode_dir if episode_dir.exists() else path


def safe_prefix(path: Path) -> str:
    name = path.name.strip().replace(" ", "_")
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge bridge .npz episodes without filename collisions.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("sources", nargs="+", help="Dataset dirs or episode dirs to merge.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_episode_dir = output_dir / "episodes"
    if output_dir.exists() and any(output_episode_dir.glob("*.npz")) and not args.overwrite:
        raise SystemExit(f"{output_episode_dir} already contains episodes; pass --overwrite to replace")
    if output_episode_dir.exists() and args.overwrite:
        shutil.rmtree(output_episode_dir)
    output_episode_dir.mkdir(parents=True, exist_ok=True)

    manifest = {"sources": [], "episodes": []}
    used_names: set[str] = set()
    for source_text in args.sources:
        source = Path(source_text)
        episode_dir = source_episode_dir(source)
        paths = sorted(episode_dir.glob("*.npz"))
        if not paths:
            raise FileNotFoundError(f"No .npz episodes found under {episode_dir}")
        prefix = safe_prefix(source)
        manifest["sources"].append({"source": str(source), "episodes": len(paths)})
        for path in paths:
            name = f"{prefix}_{path.name}"
            if name in used_names:
                raise RuntimeError(f"Duplicate output episode name: {name}")
            used_names.add(name)
            target = output_episode_dir / name
            shutil.copy2(path, target)
            item = {"source": str(path), "target": str(target)}
            meta_path = path.with_suffix(".json")
            if meta_path.exists():
                meta_target = target.with_suffix(".json")
                shutil.copy2(meta_path, meta_target)
                item["metadata"] = str(meta_target)
            manifest["episodes"].append(item)

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"merged {len(manifest['episodes'])} episodes into {output_episode_dir}")


if __name__ == "__main__":
    main()
