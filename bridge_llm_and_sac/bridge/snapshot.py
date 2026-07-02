"""Feature builders for non-invasive bridge runtime snapshots.

The functions here use duck typing and only read public-ish attributes from the
robot stack. They are safe to use from external wrappers because they do not
modify the environment or import the robot package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .recorder import RecorderConfig, TaskSpec


LANDMARK_TARGETS = {
    "landmark_blue": "meeting_room",
    "landmark_yellow": "pantry",
    "landmark_green": "door",
    "landmark_red": "office",
}


@dataclass
class BridgeSnapshot:
    maps: np.ndarray
    state: np.ndarray
    memory: np.ndarray
    candidates: np.ndarray
    distance_to_subgoal: float
    visited_ratio: float
    frontier_ratio: float
    target_found: bool
    path_valid: bool
    goal_pos: np.ndarray | None
    robot_pos: np.ndarray


@dataclass(frozen=True)
class SegmentProgressContext:
    elapsed_steps: int = 0
    start_distance: float | None = None
    best_distance: float | None = None
    recent_progress: float = 0.0
    distance_regret: float = 0.0


class SegmentProgressTracker:
    """Track progress inside the current navigation segment."""

    def __init__(self, recent_window: int = 4):
        self.recent_window = max(1, int(recent_window))
        self.distances: list[float] = []

    def reset(self) -> None:
        self.distances.clear()

    def update(self, distance_m: float | None) -> SegmentProgressContext:
        if distance_m is None:
            return SegmentProgressContext(elapsed_steps=len(self.distances))

        distance = float(distance_m)
        self.distances.append(distance)
        start = self.distances[0]
        best = min(self.distances)
        recent_start = self.distances[max(0, len(self.distances) - self.recent_window - 1)]
        recent_progress = recent_start - distance
        return SegmentProgressContext(
            elapsed_steps=len(self.distances),
            start_distance=start,
            best_distance=best,
            recent_progress=recent_progress,
            distance_regret=max(0.0, distance - best),
        )


def logodds_to_prob(grid: np.ndarray) -> np.ndarray:
    return (1.0 - 1.0 / (1.0 + np.exp(grid))).astype(np.float32)


def resize_nearest(image: np.ndarray, size: int) -> np.ndarray:
    src_h, src_w = image.shape
    rows = np.linspace(0, src_h - 1, size).round().astype(np.int64)
    cols = np.linspace(0, src_w - 1, size).round().astype(np.int64)
    return image[np.ix_(rows, cols)].astype(np.float32)


def frontier_from_prob(prob: np.ndarray) -> np.ndarray:
    free = prob < 0.4
    unknown = (prob >= 0.4) & (prob <= 0.6)
    free_neighbor = np.zeros_like(free, dtype=bool)
    free_neighbor[1:, :] |= free[:-1, :]
    free_neighbor[:-1, :] |= free[1:, :]
    free_neighbor[:, 1:] |= free[:, :-1]
    free_neighbor[:, :-1] |= free[:, 1:]
    return (unknown & free_neighbor).astype(np.float32)


def world_to_grid(pos: np.ndarray, num_cells: int, world_size_m: float) -> tuple[int, int]:
    resolution = num_cells / world_size_m
    offset = world_size_m / 2.0
    c = int((float(pos[0]) + offset) * resolution)
    r = num_cells - 1 - int((float(pos[1]) + offset) * resolution)
    return int(np.clip(r, 0, num_cells - 1)), int(np.clip(c, 0, num_cells - 1))


def gaussian_heatmap(pos: np.ndarray | None, num_cells: int, world_size_m: float, sigma_m: float) -> np.ndarray:
    if pos is None:
        return np.zeros((num_cells, num_cells), dtype=np.float32)
    r, c = world_to_grid(pos, num_cells, world_size_m)
    yy, xx = np.mgrid[0:num_cells, 0:num_cells].astype(np.float32)
    sigma_px = max(1.0, sigma_m * num_cells / max(world_size_m, 1e-6))
    return np.exp(-((yy - r) ** 2 + (xx - c) ** 2) / (2.0 * sigma_px**2)).astype(np.float32)


def get_robot_pos(env: Any) -> np.ndarray:
    try:
        return np.asarray(env.data.xpos[env.robot_base_body_id][:2], dtype=np.float32)
    except Exception:
        return np.zeros(2, dtype=np.float32)


def get_robot_yaw(env: Any) -> float:
    try:
        mat = np.asarray(env.data.xmat[env.robot_base_body_id], dtype=np.float32).reshape(3, 3)
        return float(np.arctan2(mat[1, 0], mat[0, 0]))
    except Exception:
        return 0.0


def get_goal_pos(env: Any) -> np.ndarray | None:
    for name in ("agent_assigned_target", "goal_pos", "current_target_waypoint"):
        value = getattr(env, name, None)
        if value is not None:
            arr = np.asarray(value, dtype=np.float32).reshape(-1)
            if arr.shape[0] >= 2:
                return arr[:2].copy()
    return None


def get_world_size(grid_map: Any, fallback: float = 20.0) -> float:
    return float(getattr(grid_map, "world_size_m", fallback))


def get_grid_arrays(env: Any) -> tuple[np.ndarray, np.ndarray, Any]:
    grid_map = getattr(env, "grid_map", None)
    if grid_map is None:
        raise ValueError("env must expose grid_map for bridge snapshots")
    grid = np.asarray(getattr(grid_map, "grid"), dtype=np.float32)
    visited = np.asarray(getattr(grid_map, "visited_grid", np.zeros_like(grid)), dtype=np.float32)
    return grid, visited, grid_map


def spatial_memory_dict(memory: Any) -> dict[str, Any]:
    if memory is None:
        return {}
    if isinstance(memory, dict):
        return memory
    return getattr(memory, "memory_db", {}) or {}


def topo_nodes_list(topo_map: Any) -> list[dict[str, Any]]:
    if topo_map is None:
        return []
    if isinstance(topo_map, list):
        return topo_map
    return getattr(topo_map, "nodes", []) or []


def landmark_positions(memory: Any) -> list[tuple[str, np.ndarray]]:
    result = []
    for landmark_id, entry in spatial_memory_dict(memory).items():
        if "x" in entry and "y" in entry:
            result.append((landmark_id, np.asarray([entry["x"], entry["y"]], dtype=np.float32)))
    return result


def topo_positions(topo_map: Any) -> list[np.ndarray]:
    positions = []
    for node in topo_nodes_list(topo_map):
        if "pos" in node:
            positions.append(np.asarray(node["pos"], dtype=np.float32).reshape(-1)[:2])
    return positions


def build_map_layers(env: Any, robot_pos: np.ndarray, goal_pos: np.ndarray | None, cfg: RecorderConfig) -> tuple[np.ndarray, float, float]:
    grid, visited, grid_map = get_grid_arrays(env)
    world_size_m = get_world_size(grid_map)
    prob = logodds_to_prob(grid)
    frontier = frontier_from_prob(prob)
    robot = gaussian_heatmap(robot_pos, prob.shape[0], world_size_m, sigma_m=0.55)
    goal = gaussian_heatmap(goal_pos, prob.shape[0], world_size_m, sigma_m=0.9)
    layers = [
        resize_nearest(prob, cfg.map_size),
        resize_nearest(np.clip(visited, 0.0, 1.0), cfg.map_size),
        resize_nearest(frontier, cfg.map_size),
        resize_nearest(robot, cfg.map_size),
        resize_nearest(goal, cfg.map_size),
    ]
    visited_ratio = float(np.mean(visited > 0.3))
    frontier_ratio = float(np.mean(frontier > 0.5))
    return np.stack(layers, axis=0).astype(np.float32), visited_ratio, frontier_ratio


def build_state_vector(
    env: Any,
    robot_pos: np.ndarray,
    prev_pos: np.ndarray | None,
    goal_pos: np.ndarray | None,
    visited_ratio: float,
    frontier_ratio: float,
    known_landmarks: int,
    topo_count: int,
    target_found: bool,
    progress_rate: float,
    stuck_score: float,
    cfg: RecorderConfig,
    segment_context: SegmentProgressContext | None = None,
) -> tuple[np.ndarray, float]:
    grid_map = getattr(env, "grid_map", None)
    world_size_m = get_world_size(grid_map)
    half = world_size_m / 2.0
    yaw = get_robot_yaw(env) / np.pi
    if goal_pos is None:
        delta = np.zeros(2, dtype=np.float32)
        dist = 0.0
    else:
        delta = goal_pos - robot_pos
        dist = float(np.linalg.norm(delta))
    movement = 0.0 if prev_pos is None else float(np.linalg.norm(robot_pos - prev_pos))
    segment_context = segment_context or SegmentProgressContext()
    segment_start_distance = dist if segment_context.start_distance is None else float(segment_context.start_distance)
    segment_best_distance = dist if segment_context.best_distance is None else float(segment_context.best_distance)
    segment_progress = segment_start_distance - dist
    segment_recent_progress = float(segment_context.recent_progress)
    segment_regret = float(segment_context.distance_regret)

    full_state = np.asarray(
        [
            np.clip(robot_pos[0] / half, -1.0, 1.0),
            np.clip(robot_pos[1] / half, -1.0, 1.0),
            np.clip(yaw, -1.0, 1.0),
            np.clip(delta[0] / world_size_m, -1.0, 1.0),
            np.clip(delta[1] / world_size_m, -1.0, 1.0),
            np.clip(dist / world_size_m, 0.0, 1.0),
            np.clip(progress_rate, -1.0, 1.0),
            np.clip(stuck_score, 0.0, 1.0),
            float(getattr(env, "current_step", 0)) / 8000.0,
            float(getattr(env, "_path_fallback", False)),
            known_landmarks / 8.0,
            float(target_found),
            visited_ratio,
            frontier_ratio,
            topo_count / 32.0,
            movement,
            np.clip(float(segment_context.elapsed_steps) / 64.0, 0.0, 1.0),
            np.clip(segment_start_distance / world_size_m, 0.0, 1.0),
            np.clip(segment_best_distance / world_size_m, 0.0, 1.0),
            np.clip(segment_progress / world_size_m, -1.0, 1.0),
            np.clip(segment_regret / world_size_m, 0.0, 1.0),
            np.clip(segment_recent_progress / world_size_m, -1.0, 1.0),
        ],
        dtype=np.float32,
    )
    if cfg.state_dim <= full_state.shape[0]:
        state = full_state[: cfg.state_dim]
    else:
        state = np.zeros((cfg.state_dim,), dtype=np.float32)
        state[: full_state.shape[0]] = full_state
    if state.shape[0] != cfg.state_dim:
        raise ValueError(f"state_dim mismatch: built {state.shape[0]}, expected {cfg.state_dim}")
    return state, dist


def build_memory_vector(memory: Any, topo_map: Any, robot_pos: np.ndarray, visited_ratio: float, frontier_ratio: float, cfg: RecorderConfig) -> np.ndarray:
    landmarks = landmark_positions(memory)
    nodes = topo_nodes_list(topo_map)
    visit_counts = [float(node.get("visit_count", 1)) for node in nodes]
    visual_counts = [float(node.get("visual_scan_count", 0)) for node in nodes]
    confidences = [float(entry.get("confidence", 0.0)) for entry in spatial_memory_dict(memory).values()]
    nearest_landmark = 1.0
    if landmarks:
        nearest_landmark = min(float(np.linalg.norm(pos - robot_pos)) for _, pos in landmarks) / 20.0
    vec = np.asarray(
        [
            len(landmarks) / 8.0,
            len(nodes) / 32.0,
            (np.mean(visit_counts) if visit_counts else 0.0) / 8.0,
            (np.max(visit_counts) if visit_counts else 0.0) / 8.0,
            (np.sum(visual_counts) if visual_counts else 0.0) / 64.0,
            float(getattr(topo_map, "current_node_id", -1) is not None),
            visited_ratio,
            frontier_ratio,
            len(getattr(memory, "feature_log", []) or []) / 256.0,
            np.mean(confidences) if confidences else 0.0,
            np.clip(nearest_landmark, 0.0, 1.0),
            1.0 if landmarks else 0.0,
        ],
        dtype=np.float32,
    )
    if vec.shape[0] != cfg.memory_dim:
        raise ValueError(f"memory_dim mismatch: built {vec.shape[0]}, expected {cfg.memory_dim}")
    return vec


def build_candidates(env: Any, memory: Any, topo_map: Any, task_spec: TaskSpec, robot_pos: np.ndarray, cfg: RecorderConfig) -> np.ndarray:
    grid_map = getattr(env, "grid_map", None)
    world_size_m = get_world_size(grid_map)
    items: list[tuple[float, np.ndarray, int, int]] = []

    goal_pos = get_goal_pos(env)
    if goal_pos is not None:
        items.append((0.75, goal_pos, 0, 1))

    frontiers = getattr(env, "current_frontiers", None)
    if frontiers is not None:
        for point in np.asarray(frontiers, dtype=np.float32).reshape(-1, 2)[: cfg.max_candidates * 2]:
            dist = float(np.linalg.norm(point - robot_pos))
            quality = float(np.clip(0.60 - dist / max(world_size_m, 1.0) * 0.20, 0.10, 0.60))
            items.append((quality, point, 0, 1))

    for landmark_id, pos in landmark_positions(memory):
        quality = 1.0 if LANDMARK_TARGETS.get(landmark_id) == task_spec.target else 0.65
        items.append((quality, pos, 1, 0))

    for pos in topo_positions(topo_map):
        dist = float(np.linalg.norm(pos - robot_pos))
        quality = float(np.clip(0.50 - dist / max(world_size_m, 1.0) * 0.20, 0.05, 0.50))
        items.append((quality, pos, 0, 0))

    items.sort(key=lambda item: (-item[0], float(np.linalg.norm(item[1] - robot_pos))))
    candidates = np.zeros((cfg.max_candidates, cfg.candidate_dim), dtype=np.float32)
    for idx, (quality, pos, is_landmark, is_frontier) in enumerate(items[: cfg.max_candidates]):
        dist = float(np.linalg.norm(pos - robot_pos))
        candidates[idx] = np.asarray(
            [
                quality,
                np.clip(dist / world_size_m, 0.0, 1.0),
                quality if is_landmark else 0.35,
                1.0 if dist < world_size_m * 0.5 else 0.5,
                np.clip(pos[0] / (world_size_m / 2.0), -1.0, 1.0),
                np.clip(pos[1] / (world_size_m / 2.0), -1.0, 1.0),
                float(is_landmark),
                float(is_frontier),
            ],
            dtype=np.float32,
        )
    return candidates


def target_found(memory: Any, task_spec: TaskSpec, robot_pos: np.ndarray, radius_m: float = 1.8) -> bool:
    for landmark_id, pos in landmark_positions(memory):
        if task_spec.target != "none" and LANDMARK_TARGETS.get(landmark_id) != task_spec.target:
            continue
        if float(np.linalg.norm(pos - robot_pos)) <= radius_m:
            return True
    return False


class BridgeSnapshotBuilder:
    """Build recorder-ready bridge snapshots from existing runtime objects."""

    def __init__(self, config: RecorderConfig | None = None, task_spec: TaskSpec | None = None):
        self.config = config or RecorderConfig()
        self.task_spec = task_spec or TaskSpec()
        self.segment_tracker = SegmentProgressTracker()
        self.segment_active = False

    def start_segment(self) -> None:
        self.segment_tracker.reset()
        self.segment_active = True

    def finish_segment(self) -> None:
        self.segment_active = False

    def build(
        self,
        env: Any,
        memory: Any = None,
        topo_map: Any = None,
        prev_pos: np.ndarray | None = None,
        prev_distance: float | None = None,
        prev_visited_ratio: float | None = None,
    ) -> BridgeSnapshot:
        robot_pos = get_robot_pos(env)
        goal_pos = get_goal_pos(env)
        maps, visited_ratio, frontier_ratio = build_map_layers(env, robot_pos, goal_pos, self.config)
        found = target_found(memory, self.task_spec, robot_pos)
        if prev_distance is not None and goal_pos is not None:
            distance = float(np.linalg.norm(goal_pos - robot_pos))
            progress = prev_distance - distance
        else:
            distance = float(np.linalg.norm(goal_pos - robot_pos)) if goal_pos is not None else 0.0
            progress = 0.0
        segment_context = (
            self.segment_tracker.update(distance)
            if self.segment_active
            else SegmentProgressContext()
        )
        movement = 0.0 if prev_pos is None else float(np.linalg.norm(robot_pos - prev_pos))
        stuck_score = float(np.clip(1.0 - movement * 4.0, 0.0, 1.0)) if prev_pos is not None else 0.0
        known_landmarks = len(landmark_positions(memory))
        topo_count = len(topo_nodes_list(topo_map))
        state, distance = build_state_vector(
            env,
            robot_pos,
            prev_pos,
            goal_pos,
            visited_ratio,
            frontier_ratio,
            known_landmarks,
            topo_count,
            found,
            progress,
            stuck_score,
            self.config,
            segment_context,
        )
        mem = build_memory_vector(memory, topo_map, robot_pos, visited_ratio, frontier_ratio, self.config)
        candidates = build_candidates(env, memory, topo_map, self.task_spec, robot_pos, self.config)
        path_valid = bool(getattr(env, "current_path", None) is not None and not getattr(env, "_path_fallback", False))
        return BridgeSnapshot(
            maps=maps,
            state=state,
            memory=mem,
            candidates=candidates,
            distance_to_subgoal=distance,
            visited_ratio=visited_ratio,
            frontier_ratio=frontier_ratio,
            target_found=found,
            path_valid=path_valid,
            goal_pos=goal_pos,
            robot_pos=robot_pos,
        )
