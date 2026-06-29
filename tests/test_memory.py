import json

import numpy as np

from memory import SpatialMemory, TopologicalMap


class DummyGridMap:
    def __init__(self, num_cells=20, resolution=1):
        self.num_cells_world = num_cells
        self.resolution = resolution
        self.world_origin_offset_m = np.array([num_cells / 2, num_cells / 2])
        self.grid = np.ones((num_cells, num_cells), dtype=float)
        self.visited_grid = np.zeros((num_cells, num_cells), dtype=float)


def test_spatial_memory_save_and_load(tmp_path):
    memory = SpatialMemory(save_dir=str(tmp_path))
    memory.add_memory("landmark_blue", 1.0, 2.0, confidence=0.8)
    memory.log_odometry(0.1, 0.2, 0.3)

    save_path = memory.save_to_file("unit")

    loaded = SpatialMemory(save_dir=str(tmp_path))
    loaded.load_from_file(save_path)

    assert loaded.memory_db["landmark_blue"]["x"] == 1.0
    assert loaded.memory_db["landmark_blue"]["y"] == 2.0

    odometry_file = tmp_path / "session_unit" / "odometry.json"
    assert json.loads(odometry_file.read_text(encoding="utf-8"))[0]["yaw"] == 0.3


def test_spatial_memory_saves_and_merges_visited_map(tmp_path):
    grid = DummyGridMap()
    grid.grid[2, 3] = 4.0
    grid.visited_grid[2, 3] = 1.0

    memory = SpatialMemory(save_dir=str(tmp_path))
    save_path = memory.save_visited_map(grid, "map")

    new_grid = DummyGridMap()
    assert memory.load_visited_map(new_grid, save_path)
    assert new_grid.visited_grid[2, 3] > 0


def test_topological_map_matches_revisited_place(tmp_path):
    grid = DummyGridMap()
    topo = TopologicalMap(fingerprint_radius_m=3.0, fingerprint_size=8, match_threshold=0.85)

    first_node, first_is_new = topo.visit(grid, np.array([0.0, 0.0]), landmarks=["landmark_blue"])
    second_node, second_is_new = topo.visit(grid, np.array([0.0, 0.0]), landmarks=["landmark_green"])

    assert first_is_new is True
    assert second_is_new is False
    assert first_node is second_node
    assert second_node["visit_count"] == 2
    assert "landmark_green" in second_node["landmarks_seen"]

    save_path = tmp_path / "session_topo"
    save_path.mkdir()
    topo.save_to_file(str(save_path))

    loaded = TopologicalMap()
    assert loaded.load_from_file(str(save_path))
    assert loaded.nodes[0]["visit_count"] == 2
