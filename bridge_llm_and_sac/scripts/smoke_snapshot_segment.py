#!/usr/bin/env python3
"""Smoke-test segment progress features and 16-D checkpoint compatibility."""

from __future__ import annotations

try:
    from scripts._bootstrap import ensure_project_root
except ModuleNotFoundError:
    from _bootstrap import ensure_project_root

ensure_project_root()

import numpy as np

from bridge.recorder import RecorderConfig
from bridge.snapshot import BridgeSnapshotBuilder


class FakeGridMap:
    def __init__(self):
        self.world_size_m = 20.0
        self.grid = np.zeros((80, 80), dtype=np.float32)
        self.visited_grid = np.zeros((80, 80), dtype=np.float32)


class FakeData:
    def __init__(self):
        self.xpos = np.zeros((1, 3), dtype=np.float32)
        self.xmat = np.tile(np.eye(3, dtype=np.float32).reshape(1, 9), (1, 1))


class FakeEnv:
    def __init__(self):
        self.grid_map = FakeGridMap()
        self.data = FakeData()
        self.robot_base_body_id = 0
        self.goal_pos = np.asarray([5.0, 0.0], dtype=np.float32)
        self.current_path = np.asarray([[0.0, 0.0], [5.0, 0.0]], dtype=np.float32)
        self._path_fallback = False


def main() -> None:
    env = FakeEnv()

    old_builder = BridgeSnapshotBuilder(RecorderConfig(state_dim=16))
    old_builder.start_segment()
    old_snapshot = old_builder.build(env)
    assert old_snapshot.state.shape == (16,)

    new_builder = BridgeSnapshotBuilder(RecorderConfig(state_dim=22))
    new_builder.start_segment()
    first = new_builder.build(env)
    env.data.xpos[0, 0] = 1.0
    second = new_builder.build(env, prev_pos=first.robot_pos, prev_distance=first.distance_to_subgoal)
    assert second.state.shape == (22,)
    assert second.state[16] > first.state[16]
    assert second.state[19] > first.state[19]
    print("snapshot_segment_smoke_ok")


if __name__ == "__main__":
    main()
