#!/usr/bin/env python3
"""Generate a small synthetic dataset for BridgeNet smoke tests."""

from __future__ import annotations

try:
    from scripts._bootstrap import ensure_project_root
except ModuleNotFoundError:
    from _bootstrap import ensure_project_root

ensure_project_root()

import argparse
from pathlib import Path

import numpy as np

from bridge.constants import EVENT_TYPES, REPLAN_ACTIONS, SKILL_TYPES


def gaussian_map(size: int, center_x: float, center_y: float, sigma: float = 0.18) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    xx = xx / max(size - 1, 1)
    yy = yy / max(size - 1, 1)
    return np.exp(-((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2 * sigma**2)).astype(np.float32)


def generate_episode(rng: np.random.Generator, steps: int, map_size: int, max_candidates: int) -> dict[str, np.ndarray]:
    map_channels = 5
    state_dim = 22
    memory_dim = 12
    candidate_dim = 8

    maps = np.zeros((steps, map_channels, map_size, map_size), dtype=np.float32)
    state = np.zeros((steps, state_dim), dtype=np.float32)
    memory = np.zeros((steps, memory_dim), dtype=np.float32)
    task = np.zeros((steps, 3), dtype=np.int64)
    skill = np.zeros((steps,), dtype=np.int64)
    candidates = np.zeros((steps, max_candidates, candidate_dim), dtype=np.float32)
    candidate_mask = np.ones((steps, max_candidates), dtype=bool)
    candidate_score_target = np.zeros((steps, max_candidates), dtype=np.float32)

    event = np.zeros((steps,), dtype=np.int64)
    replan = np.zeros((steps,), dtype=np.int64)
    success = np.zeros((steps,), dtype=np.float32)
    stuck = np.zeros((steps,), dtype=np.float32)
    target_found = np.zeros((steps,), dtype=np.float32)
    cost = np.zeros((steps,), dtype=np.float32)
    info_gain = np.zeros((steps,), dtype=np.float32)

    intent_id = int(rng.integers(0, 3))
    target_id = int(rng.integers(1, 5))
    constraint_id = int(rng.integers(0, 5))
    task[:, :] = np.array([intent_id, target_id, constraint_id], dtype=np.int64)

    distance = float(rng.uniform(3.0, 9.0))
    start_distance = distance
    best_distance = distance
    distance_history: list[float] = []
    known_landmarks = float(rng.integers(0, 4))
    failed_subgoals = float(rng.integers(0, 3))

    for t in range(steps):
        progress = np.clip(rng.normal(0.08, 0.08), -0.08, 0.25)
        if rng.random() < 0.13:
            progress *= 0.1
        distance = max(0.0, distance - max(progress, 0.0) + float(rng.normal(0.0, 0.03)))
        stuck_score = float(np.clip(1.0 - progress * 8.0 + rng.normal(0.0, 0.15), 0.0, 1.0))
        found = float(rng.random() < (0.02 + 0.03 * known_landmarks + 0.05 * max(progress, 0.0)))
        target_found[t] = found
        stuck[t] = float(stuck_score > 0.78)
        success[t] = float(distance < 0.75)
        info_gain[t] = float(np.clip(progress * 2.5 + rng.normal(0.08, 0.08), 0.0, 1.0))
        cost[t] = float(np.clip(distance / 10.0 + stuck_score * 0.15, 0.0, 1.0))
        distance_history.append(distance)
        best_distance = min(best_distance, distance)
        recent_start = distance_history[max(0, len(distance_history) - 5)]
        recent_progress = recent_start - distance

        skill[t] = int(rng.choice([1, 2, 3], p=[0.35, 0.55, 0.10]))
        state[t, :8] = np.array(
            [
                rng.uniform(-1, 1),
                rng.uniform(-1, 1),
                rng.uniform(-np.pi, np.pi) / np.pi,
                rng.normal(0, 0.2),
                rng.normal(0, 0.2),
                distance / 10.0,
                progress,
                stuck_score,
            ],
            dtype=np.float32,
        )
        state[t, 8:12] = np.array([t / steps, failed_subgoals / 5.0, known_landmarks / 5.0, found], dtype=np.float32)
        state[t, 16:22] = np.array(
            [
                min((t + 1) / 64.0, 1.0),
                np.clip(start_distance / 10.0, 0.0, 1.0),
                np.clip(best_distance / 10.0, 0.0, 1.0),
                np.clip((start_distance - distance) / 10.0, -1.0, 1.0),
                np.clip((distance - best_distance) / 10.0, 0.0, 1.0),
                np.clip(recent_progress / 10.0, -1.0, 1.0),
            ],
            dtype=np.float32,
        )

        memory[t, :4] = np.array([known_landmarks / 5.0, failed_subgoals / 5.0, t / steps, info_gain[t]], dtype=np.float32)
        memory[t, 4:] = rng.normal(0.0, 0.2, size=memory_dim - 4)

        center_x = float(np.clip(0.5 + rng.normal(0.0, 0.12), 0.0, 1.0))
        center_y = float(np.clip(0.5 + rng.normal(0.0, 0.12), 0.0, 1.0))
        maps[t, 0] = np.clip(rng.beta(1.5, 5.0, size=(map_size, map_size)), 0.0, 1.0)
        maps[t, 1] = gaussian_map(map_size, center_x, center_y, sigma=0.22)
        maps[t, 2] = (rng.random((map_size, map_size)) < (0.02 + 0.04 * info_gain[t])).astype(np.float32)
        maps[t, 3] = gaussian_map(map_size, rng.random(), rng.random(), sigma=0.12)
        maps[t, 4] = gaussian_map(map_size, 0.5, 0.5, sigma=0.28)

        cand_quality = rng.uniform(0.0, 1.0, size=max_candidates).astype(np.float32)
        if found:
            cand_quality[0] = 1.0
        candidates[t, :, 0] = cand_quality
        candidates[t, :, 1] = rng.uniform(0.0, 1.0, size=max_candidates)
        candidates[t, :, 2] = info_gain[t]
        candidates[t, :, 3] = 1.0 - stuck_score
        candidates[t, :, 4:] = rng.normal(0.0, 0.3, size=(max_candidates, candidate_dim - 4))
        candidate_score_target[t] = np.clip(0.45 * cand_quality + 0.35 * info_gain[t] + 0.20 * (1.0 - stuck_score), 0.0, 1.0)

        if success[t] > 0.5:
            event[t] = EVENT_TYPES.index("subgoal_completed")
            replan[t] = REPLAN_ACTIONS.index("interrupt_and_scan")
        elif found > 0.5:
            event[t] = EVENT_TYPES.index("target_candidate_found")
            replan[t] = REPLAN_ACTIONS.index("go_to_target_candidate")
            known_landmarks += 1.0
        elif stuck[t] > 0.5:
            event[t] = EVENT_TYPES.index("navigation_stuck")
            replan[t] = REPLAN_ACTIONS.index("switch_subgoal")
            failed_subgoals += 1.0
        elif info_gain[t] < 0.08 and t > steps * 0.25:
            event[t] = EVENT_TYPES.index("low_information_gain")
            replan[t] = REPLAN_ACTIONS.index("switch_subgoal")
        elif skill[t] == SKILL_TYPES.index("scan"):
            event[t] = EVENT_TYPES.index("need_scan")
            replan[t] = REPLAN_ACTIONS.index("interrupt_and_scan")
        else:
            event[t] = EVENT_TYPES.index("continue")
            replan[t] = REPLAN_ACTIONS.index("continue_current")

    return {
        "maps": maps,
        "state": state,
        "memory": memory,
        "task": task,
        "skill": skill,
        "candidates": candidates,
        "candidate_mask": candidate_mask,
        "event": event,
        "replan": replan,
        "success": success,
        "stuck": stuck,
        "target_found": target_found,
        "cost": cost,
        "info_gain": info_gain,
        "candidate_score_target": candidate_score_target,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic bridge episodes.")
    parser.add_argument("--output-dir", default="data/synthetic")
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--map-size", type=int, default=64)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    out_dir = Path(args.output_dir) / "episodes"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    for episode_idx in range(args.episodes):
        episode = generate_episode(rng, args.steps, args.map_size, args.max_candidates)
        np.savez_compressed(out_dir / f"episode_{episode_idx:06d}.npz", **episode)
    print(f"Wrote {args.episodes} synthetic episodes to {out_dir}")


if __name__ == "__main__":
    main()
