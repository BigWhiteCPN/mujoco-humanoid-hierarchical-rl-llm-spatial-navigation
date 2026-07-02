#!/usr/bin/env python3
"""Smoke test the sidecar collector with fake runtime objects."""

from __future__ import annotations

try:
    from scripts._bootstrap import ensure_project_root
except ModuleNotFoundError:
    from _bootstrap import ensure_project_root

ensure_project_root()

from pathlib import Path

import numpy as np

from bridge.recorder import TaskSpec
from bridge.sidecar import BridgeSidecarCollector
from scripts.validate_episode import validate_file


class FakeGridMap:
    def __init__(self):
        self.world_size_m = 20.0
        self.resolution = 6
        self.num_cells_world = 120
        self.world_origin_offset_m = np.array([10.0, 10.0], dtype=np.float32)
        self.grid = np.zeros((120, 120), dtype=np.float32)
        self.visited_grid = np.zeros((120, 120), dtype=np.float32)


class FakeData:
    def __init__(self):
        self.xpos = np.zeros((1, 3), dtype=np.float32)
        self.xmat = np.tile(np.eye(3, dtype=np.float32).reshape(1, 9), (1, 1))


class FakeEnv:
    def __init__(self):
        self.grid_map = FakeGridMap()
        self.robot_base_body_id = 0
        self.data = FakeData()
        self.goal_pos = np.array([5.0, 0.0], dtype=np.float32)
        self.current_path = np.array([[0.0, 0.0], [5.0, 0.0]], dtype=np.float32)
        self.current_frontiers = np.array([[2.0, 1.0], [3.0, -1.0]], dtype=np.float32)
        self.current_step = 0
        self._path_fallback = False


def main() -> None:
    env = FakeEnv()
    memory = type("FakeMemory", (), {})()
    memory.memory_db = {
        "landmark_blue": {"x": 4.8, "y": 0.2, "confidence": 1.0, "observations": 3},
    }
    memory.feature_log = [{"feature_id": "landmark_blue"}]
    topo_map = type("FakeTopo", (), {})()
    topo_map.nodes = [
        {"pos": np.array([1.0, 0.0], dtype=np.float32), "visit_count": 2, "visual_scan_count": 1},
    ]
    topo_map.current_node_id = 0

    collector = BridgeSidecarCollector(
        output_dir="data/sidecar_smoke",
        episode_id="smoke",
        task_spec=TaskSpec(intent="search_place", target="meeting_room", constraint="stop_when_found"),
        min_interval_s=0.0,
    )
    collector.start_segment()
    for step in range(16):
        env.current_step = step
        env.data.xpos[0, :2] = np.array([step * 0.25, 0.0], dtype=np.float32)
        c = 60 + step
        env.grid_map.visited_grid[58:62, 58:c] = 1.0
        if step > 10:
            env.current_path = None
            env._path_fallback = True
        collector.observe(env, memory=memory, topo_map=topo_map, skill="explore_frontier", force=True)
    collector.finish_segment()

    path = collector.save()
    errors = validate_file(path)
    if errors:
        raise SystemExit("\n".join(errors))
    data = np.load(path)
    assert data["state"].shape[-1] == collector.config.state_dim
    assert data["state"][-1, 16] > data["state"][0, 16]
    print(f"sidecar smoke ok: {Path(path)} steps={collector.num_steps}")


if __name__ == "__main__":
    main()
