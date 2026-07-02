#!/usr/bin/env python3
"""Convert saved agent memory logs into weakly labeled bridge episodes.

This is an offline converter only. It does not import or modify the robot
runtime. The resulting data is useful for representation pretraining and smoke
tests before a runtime recorder is wired in.
"""

from __future__ import annotations

try:
    from scripts._bootstrap import ensure_project_root
except ModuleNotFoundError:
    from _bootstrap import ensure_project_root

ensure_project_root()

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from bridge.recorder import BridgeEpisodeRecorder, RecorderConfig, StepLabels, TaskSpec


LANDMARK_TARGETS = {
    "landmark_blue": "meeting_room",
    "landmark_yellow": "pantry",
    "landmark_green": "door",
    "landmark_red": "office",
}


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


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


def gaussian_heatmap(pos: np.ndarray | None, num_cells: int, world_size_m: float, sigma_m: float = 0.8) -> np.ndarray:
    if pos is None:
        return np.zeros((num_cells, num_cells), dtype=np.float32)
    r, c = world_to_grid(pos, num_cells, world_size_m)
    yy, xx = np.mgrid[0:num_cells, 0:num_cells].astype(np.float32)
    sigma_px = max(1.0, sigma_m * num_cells / world_size_m)
    return np.exp(-((yy - r) ** 2 + (xx - c) ** 2) / (2.0 * sigma_px**2)).astype(np.float32)


def build_map_layers(
    occupancy_grid: np.ndarray,
    visited_grid: np.ndarray,
    robot_pos: np.ndarray,
    goal_pos: np.ndarray | None,
    map_size: int,
    world_size_m: float,
) -> np.ndarray:
    prob = logodds_to_prob(occupancy_grid)
    frontier = frontier_from_prob(prob)
    robot = gaussian_heatmap(robot_pos, prob.shape[0], world_size_m, sigma_m=0.55)
    goal = gaussian_heatmap(goal_pos, prob.shape[0], world_size_m, sigma_m=0.9)
    layers = [
        resize_nearest(prob, map_size),
        resize_nearest(np.clip(visited_grid, 0.0, 1.0), map_size),
        resize_nearest(frontier, map_size),
        resize_nearest(robot, map_size),
        resize_nearest(goal, map_size),
    ]
    return np.stack(layers, axis=0).astype(np.float32)


def topo_positions(topo_nodes: list[dict[str, Any]]) -> list[np.ndarray]:
    positions = []
    for node in sorted(topo_nodes, key=lambda item: float(item.get("first_seen", 0.0))):
        if "pos" in node:
            positions.append(np.asarray(node["pos"], dtype=np.float32))
    return positions


def odom_positions(odom: list[dict[str, Any]]) -> list[np.ndarray]:
    positions = []
    for item in odom:
        if "x" in item and "y" in item:
            positions.append(np.asarray([item["x"], item["y"]], dtype=np.float32))
    return positions


def interpolate_positions(positions: list[np.ndarray], steps_per_segment: int) -> list[np.ndarray]:
    if not positions:
        return [np.zeros(2, dtype=np.float32)]
    if len(positions) == 1:
        return [positions[0].astype(np.float32) for _ in range(max(steps_per_segment, 1))]
    output: list[np.ndarray] = []
    for a, b in zip(positions[:-1], positions[1:]):
        for i in range(steps_per_segment):
            alpha = i / max(steps_per_segment, 1)
            output.append((a * (1.0 - alpha) + b * alpha).astype(np.float32))
    output.append(positions[-1].astype(np.float32))
    return output


def choose_task(spatial_memory: dict[str, Any]) -> TaskSpec:
    if "landmark_blue" in spatial_memory:
        return TaskSpec(intent="search_place", target="meeting_room", constraint="stop_when_found")
    if spatial_memory:
        first_key = sorted(spatial_memory.keys())[0]
        return TaskSpec(intent="search_place", target=LANDMARK_TARGETS.get(first_key, "unknown_landmark"), constraint="stop_when_found")
    return TaskSpec(intent="explore", target="none", constraint="prefer_safe")


def landmark_positions(spatial_memory: dict[str, Any]) -> list[tuple[str, np.ndarray]]:
    result = []
    for landmark_id, entry in spatial_memory.items():
        if "x" in entry and "y" in entry:
            result.append((landmark_id, np.asarray([entry["x"], entry["y"]], dtype=np.float32)))
    return result


def build_candidates(
    robot_pos: np.ndarray,
    landmarks: list[tuple[str, np.ndarray]],
    topo: list[np.ndarray],
    target: str,
    max_candidates: int,
    world_size_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    items: list[tuple[float, np.ndarray, int]] = []
    for landmark_id, pos in landmarks:
        quality = 1.0 if LANDMARK_TARGETS.get(landmark_id) == target else 0.65
        items.append((quality, pos, 1))
    for pos in topo:
        dist = float(np.linalg.norm(pos - robot_pos))
        quality = float(np.clip(0.55 - dist / max(world_size_m, 1.0) * 0.25, 0.05, 0.55))
        items.append((quality, pos, 0))
    items.sort(key=lambda item: (-item[0], float(np.linalg.norm(item[1] - robot_pos))))

    candidates = np.zeros((max_candidates, 8), dtype=np.float32)
    score_target = np.zeros((max_candidates,), dtype=np.float32)
    for idx, (quality, pos, is_landmark) in enumerate(items[:max_candidates]):
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
                1.0 - float(is_landmark),
            ],
            dtype=np.float32,
        )
        score_target[idx] = float(quality)
    return candidates, score_target


def build_state(
    robot_pos: np.ndarray,
    prev_pos: np.ndarray | None,
    goal_pos: np.ndarray | None,
    step_idx: int,
    total_steps: int,
    visited_ratio: float,
    frontier_ratio: float,
    known_landmarks: int,
    topo_count: int,
    target_found: bool,
    world_size_m: float,
) -> np.ndarray:
    state = np.zeros((22,), dtype=np.float32)
    half = world_size_m / 2.0
    if goal_pos is not None:
        delta = goal_pos - robot_pos
        dist = float(np.linalg.norm(delta))
    else:
        delta = np.zeros(2, dtype=np.float32)
        dist = 0.0
    progress = 0.0
    if prev_pos is not None and goal_pos is not None:
        prev_dist = float(np.linalg.norm(goal_pos - prev_pos))
        progress = prev_dist - dist
    movement = 0.0 if prev_pos is None else float(np.linalg.norm(robot_pos - prev_pos))
    stuck_score = float(np.clip(1.0 - movement * 4.0, 0.0, 1.0)) if step_idx > 1 else 0.0
    yaw = 0.0
    if prev_pos is not None and movement > 1e-4:
        vec = robot_pos - prev_pos
        yaw = float(np.arctan2(vec[1], vec[0]) / np.pi)

    state[:16] = np.asarray(
        [
            np.clip(robot_pos[0] / half, -1.0, 1.0),
            np.clip(robot_pos[1] / half, -1.0, 1.0),
            yaw,
            np.clip(delta[0] / world_size_m, -1.0, 1.0),
            np.clip(delta[1] / world_size_m, -1.0, 1.0),
            np.clip(dist / world_size_m, 0.0, 1.0),
            np.clip(progress, -1.0, 1.0),
            stuck_score,
            step_idx / max(total_steps - 1, 1),
            0.0,
            known_landmarks / 8.0,
            float(target_found),
            visited_ratio,
            frontier_ratio,
            topo_count / 32.0,
            movement,
        ],
        dtype=np.float32,
    )
    state[16:22] = np.asarray(
        [
            min((step_idx + 1) / 64.0, 1.0),
            np.clip(dist / world_size_m, 0.0, 1.0),
            np.clip(dist / world_size_m, 0.0, 1.0),
            0.0,
            0.0,
            np.clip(progress / world_size_m, -1.0, 1.0),
        ],
        dtype=np.float32,
    )
    return state


def build_memory(
    spatial_memory: dict[str, Any],
    topo_nodes: list[dict[str, Any]],
    feature_obs: list[dict[str, Any]],
    robot_pos: np.ndarray,
    visited_ratio: float,
    frontier_ratio: float,
) -> np.ndarray:
    landmarks = landmark_positions(spatial_memory)
    visit_counts = [float(node.get("visit_count", 1)) for node in topo_nodes]
    visual_counts = [float(node.get("visual_scan_count", 0)) for node in topo_nodes]
    confidences = [float(entry.get("confidence", 0.0)) for entry in spatial_memory.values()]
    nearest_landmark = 1.0
    if landmarks:
        nearest_landmark = min(float(np.linalg.norm(pos - robot_pos)) for _, pos in landmarks) / 20.0
    return np.asarray(
        [
            len(landmarks) / 8.0,
            len(topo_nodes) / 32.0,
            (np.mean(visit_counts) if visit_counts else 0.0) / 8.0,
            (np.max(visit_counts) if visit_counts else 0.0) / 8.0,
            (np.sum(visual_counts) if visual_counts else 0.0) / 64.0,
            0.0,
            visited_ratio,
            frontier_ratio,
            len(feature_obs) / 256.0,
            np.mean(confidences) if confidences else 0.0,
            np.clip(nearest_landmark, 0.0, 1.0),
            1.0 if landmarks else 0.0,
        ],
        dtype=np.float32,
    )


def convert_session(session_dir: Path, output_dir: Path, args: argparse.Namespace) -> Path | None:
    occ_path = session_dir / "occupancy_grid.npy"
    vis_path = session_dir / "visited_grid.npy"
    if not occ_path.exists() or not vis_path.exists():
        return None

    occupancy = np.load(occ_path)
    visited = np.load(vis_path)
    prob = logodds_to_prob(occupancy)
    frontier = frontier_from_prob(prob)
    visited_ratio = float(np.mean(visited > 0.3))
    frontier_ratio = float(np.mean(frontier > 0.5))

    spatial_memory = load_json(session_dir / "spatial_memory.json", {})
    topo_nodes = load_json(session_dir / "topological_map.json", [])
    feature_obs = load_json(session_dir / "feature_observations.json", [])
    odom = load_json(session_dir / "odometry.json", [])

    positions = odom_positions(odom)
    if not positions:
        positions = topo_positions(topo_nodes)
    positions = interpolate_positions(positions, args.steps_per_segment)
    if len(positions) < args.min_steps:
        positions = positions + [positions[-1].copy() for _ in range(args.min_steps - len(positions))]

    task = choose_task(spatial_memory)
    landmarks = landmark_positions(spatial_memory)
    target_landmarks = [pos for landmark_id, pos in landmarks if LANDMARK_TARGETS.get(landmark_id) == task.target]
    goal_pos = target_landmarks[0] if target_landmarks else (landmarks[0][1] if landmarks else None)
    topo = topo_positions(topo_nodes)

    recorder = BridgeEpisodeRecorder(
        output_dir,
        config=RecorderConfig(map_size=args.map_size, max_candidates=args.max_candidates),
        episode_id=session_dir.name.replace("session_", ""),
        task_spec=task,
    )

    target_found_once = False
    for step_idx, robot_pos in enumerate(positions):
        prev_pos = positions[step_idx - 1] if step_idx > 0 else None
        candidates, score_target = build_candidates(
            robot_pos, landmarks, topo, task.target, args.max_candidates, args.world_size_m
        )
        target_found = False
        if goal_pos is not None and np.linalg.norm(goal_pos - robot_pos) < args.target_found_radius:
            target_found = True
            target_found_once = True

        maps = build_map_layers(occupancy, visited, robot_pos, goal_pos, args.map_size, args.world_size_m)
        state = build_state(
            robot_pos,
            prev_pos,
            goal_pos,
            step_idx,
            len(positions),
            visited_ratio,
            frontier_ratio,
            len(landmarks),
            len(topo_nodes),
            target_found,
            args.world_size_m,
        )
        memory = build_memory(spatial_memory, topo_nodes, feature_obs, robot_pos, visited_ratio, frontier_ratio)

        if target_found:
            event = "target_candidate_found"
            replan = "go_to_target_candidate"
        elif step_idx == len(positions) - 1 and target_found_once:
            event = "subgoal_completed"
            replan = "interrupt_and_scan"
        elif frontier_ratio < 0.002 and visited_ratio > 0.02:
            event = "low_information_gain"
            replan = "switch_subgoal"
        elif step_idx % max(args.steps_per_segment, 1) == 0 and topo_nodes:
            event = "need_scan"
            replan = "interrupt_and_scan"
        else:
            event = "continue"
            replan = "continue_current"

        dist = float(np.linalg.norm(goal_pos - robot_pos)) if goal_pos is not None else 0.0
        labels = StepLabels(
            event=event,
            replan=replan,
            success=float(event == "subgoal_completed"),
            stuck=0.0,
            target_found=float(target_found),
            cost=float(np.clip(dist / args.world_size_m, 0.0, 1.0)),
            info_gain=float(np.clip(frontier_ratio * 20.0 + visited_ratio, 0.0, 1.0)),
            candidate_score_target=score_target,
        )
        recorder.record_step(
            maps=maps,
            state=state,
            memory=memory,
            skill="explore_frontier",
            candidates=candidates,
            labels=labels,
        )

    return recorder.save()


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert saved memory_logs sessions to bridge episodes.")
    parser.add_argument(
        "--memory-logs",
        default="../agent_system_complex_version/memory_logs",
        help="Directory containing session_* memory log folders.",
    )
    parser.add_argument("--output-dir", default="data/from_memory_logs")
    parser.add_argument("--map-size", type=int, default=64)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--world-size-m", type=float, default=20.0)
    parser.add_argument("--steps-per-segment", type=int, default=4)
    parser.add_argument("--min-steps", type=int, default=12)
    parser.add_argument("--target-found-radius", type=float, default=1.8)
    parser.add_argument("--max-sessions", type=int, default=0)
    args = parser.parse_args()

    root = Path(args.memory_logs)
    if not root.is_absolute():
        root = Path.cwd() / root
    sessions = sorted(path for path in root.glob("session_*") if path.is_dir())
    if args.max_sessions > 0:
        sessions = sessions[-args.max_sessions :]
    if not sessions:
        raise SystemExit(f"No session_* directories found under {root}")

    output_dir = Path(args.output_dir)
    converted = []
    skipped = []
    for session in sessions:
        result = convert_session(session, output_dir, args)
        if result is None:
            skipped.append(session.name)
        else:
            converted.append(result)
            print(f"[OK] {session.name} -> {result}")
    print(f"Converted {len(converted)} sessions; skipped {len(skipped)}")


if __name__ == "__main__":
    main()
