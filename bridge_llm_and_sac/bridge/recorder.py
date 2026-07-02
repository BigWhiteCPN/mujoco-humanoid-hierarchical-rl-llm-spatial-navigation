"""Non-invasive episode recorder for bridge training data.

This module intentionally has no dependency on MuJoCo or the existing
`agent_system_complex_version` package. It records arrays that callers provide
from wrappers, callbacks, or offline log converters.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .constants import (
    CONSTRAINT_TO_ID,
    EVENT_TO_ID,
    INTENT_TO_ID,
    REPLAN_TO_ID,
    SKILL_TO_ID,
    TARGET_TO_ID,
)


@dataclass
class RecorderConfig:
    map_channels: int = 5
    map_size: int = 64
    state_dim: int = 22
    memory_dim: int = 12
    candidate_dim: int = 8
    max_candidates: int = 8
    default_event: str = "continue"
    default_replan: str = "continue_current"
    stuck_score_threshold: float = 0.78
    low_info_gain_threshold: float = 0.08
    low_progress_threshold: float = 0.015
    success_dist_threshold: float = 0.75


@dataclass
class TaskSpec:
    intent: str = "explore"
    target: str = "none"
    constraint: str = "none"
    raw_text: str = ""

    def to_ids(self) -> np.ndarray:
        return np.asarray(
            [
                INTENT_TO_ID.get(self.intent, INTENT_TO_ID["explore"]),
                TARGET_TO_ID.get(self.target, TARGET_TO_ID["none"]),
                CONSTRAINT_TO_ID.get(self.constraint, CONSTRAINT_TO_ID["none"]),
            ],
            dtype=np.int64,
        )


@dataclass
class StepLabels:
    event: str = "continue"
    replan: str = "continue_current"
    success: float = 0.0
    stuck: float = 0.0
    target_found: float = 0.0
    cost: float = 0.0
    info_gain: float = 0.0
    candidate_score_target: np.ndarray | None = None


@dataclass
class StepSignals:
    distance_to_subgoal: float | None = None
    progress_rate: float | None = None
    stuck_score: float | None = None
    info_gain: float | None = None
    target_found: bool = False
    path_valid: bool = True
    need_scan: bool = False
    final_distance: float | None = None
    skill_done: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


def _as_float_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {arr.shape}")
    return arr


def _normalize_name_or_id(value: str | int, mapping: dict[str, int], name: str) -> int:
    if isinstance(value, str):
        if value not in mapping:
            known = ", ".join(mapping.keys())
            raise ValueError(f"Unknown {name} '{value}'. Known values: {known}")
        return mapping[value]
    return int(value)


class BridgeEpisodeRecorder:
    """Collect one bridge-training episode and write it as a compressed npz."""

    def __init__(
        self,
        output_dir: str | Path,
        config: RecorderConfig | None = None,
        episode_id: str | None = None,
        task_spec: TaskSpec | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.episode_dir = self.output_dir / "episodes"
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or RecorderConfig()
        self.episode_id = episode_id or time.strftime("%Y%m%d_%H%M%S")
        self.task_spec = task_spec or TaskSpec()
        self.steps: list[dict[str, np.ndarray | int | float | bool]] = []
        self.metadata: dict[str, Any] = {
            "episode_id": self.episode_id,
            "created_at": time.time(),
            "task_spec": asdict(self.task_spec),
            "config": asdict(self.config),
        }

    def infer_labels(self, signals: StepSignals | None, candidate_count: int) -> StepLabels:
        cfg = self.config
        if signals is None:
            return StepLabels(
                event=cfg.default_event,
                replan=cfg.default_replan,
                candidate_score_target=np.zeros((candidate_count,), dtype=np.float32),
            )

        final_dist = signals.final_distance
        if final_dist is None:
            final_dist = signals.distance_to_subgoal
        success = float(bool(signals.skill_done) or (final_dist is not None and final_dist < cfg.success_dist_threshold))
        stuck_score = float(signals.stuck_score or 0.0)
        progress_rate = float(signals.progress_rate or 0.0)
        low_progress = progress_rate < cfg.low_progress_threshold
        stuck = float(stuck_score >= cfg.stuck_score_threshold and low_progress)
        info_gain = float(np.clip(signals.info_gain if signals.info_gain is not None else 0.0, 0.0, 1.0))
        target_found = float(bool(signals.target_found))

        if success:
            event = "subgoal_completed"
            replan = "interrupt_and_scan"
        elif target_found:
            event = "target_candidate_found"
            replan = "go_to_target_candidate"
        elif not signals.path_valid:
            event = "path_invalidated"
            replan = "switch_subgoal"
        elif stuck:
            event = "navigation_stuck"
            replan = "switch_subgoal"
        elif signals.need_scan:
            event = "need_scan"
            replan = "interrupt_and_scan"
        elif info_gain < cfg.low_info_gain_threshold and low_progress:
            event = "low_information_gain"
            replan = "switch_subgoal"
        else:
            event = cfg.default_event
            replan = cfg.default_replan

        candidate_scores = np.zeros((candidate_count,), dtype=np.float32)
        return StepLabels(
            event=event,
            replan=replan,
            success=success,
            stuck=stuck,
            target_found=target_found,
            cost=float(np.clip((final_dist or 0.0) / 10.0 + stuck_score * 0.15, 0.0, 1.0)),
            info_gain=info_gain,
            candidate_score_target=candidate_scores,
        )

    def record_step(
        self,
        *,
        maps: np.ndarray,
        state: np.ndarray,
        memory: np.ndarray,
        skill: str | int,
        candidates: np.ndarray | None = None,
        labels: StepLabels | None = None,
        signals: StepSignals | None = None,
    ) -> None:
        cfg = self.config
        maps_arr = _as_float_array(maps, (cfg.map_channels, cfg.map_size, cfg.map_size), "maps")
        state_arr = _as_float_array(state, (cfg.state_dim,), "state")
        memory_arr = _as_float_array(memory, (cfg.memory_dim,), "memory")
        skill_id = _normalize_name_or_id(skill, SKILL_TO_ID, "skill")

        candidate_arr = np.zeros((cfg.max_candidates, cfg.candidate_dim), dtype=np.float32)
        candidate_mask = np.zeros((cfg.max_candidates,), dtype=bool)
        candidate_count = 0
        if candidates is not None:
            raw_candidates = np.asarray(candidates, dtype=np.float32)
            if raw_candidates.ndim != 2 or raw_candidates.shape[1] != cfg.candidate_dim:
                raise ValueError(
                    f"candidates must have shape [K, {cfg.candidate_dim}], got {raw_candidates.shape}"
                )
            candidate_count = min(raw_candidates.shape[0], cfg.max_candidates)
            candidate_arr[:candidate_count] = raw_candidates[:candidate_count]
            candidate_mask[:candidate_count] = True

        if labels is None:
            labels = self.infer_labels(signals, cfg.max_candidates)

        candidate_score_target = labels.candidate_score_target
        if candidate_score_target is None:
            candidate_score_target = np.zeros((cfg.max_candidates,), dtype=np.float32)
        else:
            candidate_score_target = np.asarray(candidate_score_target, dtype=np.float32)
            if candidate_score_target.shape[0] > cfg.max_candidates:
                candidate_score_target = candidate_score_target[: cfg.max_candidates]
            elif candidate_score_target.shape[0] < cfg.max_candidates:
                padded = np.zeros((cfg.max_candidates,), dtype=np.float32)
                padded[: candidate_score_target.shape[0]] = candidate_score_target
                candidate_score_target = padded

        self.steps.append(
            {
                "maps": maps_arr,
                "state": state_arr,
                "memory": memory_arr,
                "task": self.task_spec.to_ids(),
                "skill": skill_id,
                "candidates": candidate_arr,
                "candidate_mask": candidate_mask,
                "event": _normalize_name_or_id(labels.event, EVENT_TO_ID, "event"),
                "replan": _normalize_name_or_id(labels.replan, REPLAN_TO_ID, "replan"),
                "success": float(labels.success),
                "stuck": float(labels.stuck),
                "target_found": float(labels.target_found),
                "cost": float(labels.cost),
                "info_gain": float(labels.info_gain),
                "candidate_score_target": candidate_score_target,
            }
        )

    def to_episode_arrays(self) -> dict[str, np.ndarray]:
        if not self.steps:
            raise ValueError("Cannot save an empty bridge episode")
        keys = self.steps[0].keys()
        arrays: dict[str, np.ndarray] = {}
        for key in keys:
            arrays[key] = np.asarray([step[key] for step in self.steps])
        return arrays

    def relabel_steps(
        self,
        start: int,
        end: int | None = None,
        *,
        event: str | int,
        replan: str | int,
        success: float | None = None,
        stuck: float | None = None,
        target_found: float | None = None,
        cost: float | None = None,
        info_gain: float | None = None,
    ) -> int:
        """Patch labels for an already-recorded step range.

        This is useful for hindsight labels after a skill returns success/failure.
        """
        if end is None:
            end = len(self.steps)
        start = max(0, int(start))
        end = min(len(self.steps), int(end))
        if start >= end:
            return 0

        event_id = _normalize_name_or_id(event, EVENT_TO_ID, "event")
        replan_id = _normalize_name_or_id(replan, REPLAN_TO_ID, "replan")
        for step in self.steps[start:end]:
            step["event"] = event_id
            step["replan"] = replan_id
            if success is not None:
                step["success"] = float(success)
            if stuck is not None:
                step["stuck"] = float(stuck)
            if target_found is not None:
                step["target_found"] = float(target_found)
            if cost is not None:
                step["cost"] = float(cost)
            if info_gain is not None:
                step["info_gain"] = float(info_gain)
        return end - start

    def save(self) -> Path:
        arrays = self.to_episode_arrays()
        episode_path = self.episode_dir / f"episode_{self.episode_id}.npz"
        np.savez_compressed(episode_path, **arrays)

        meta_path = self.episode_dir / f"episode_{self.episode_id}.json"
        meta = dict(self.metadata)
        meta["num_steps"] = len(self.steps)
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return episode_path
