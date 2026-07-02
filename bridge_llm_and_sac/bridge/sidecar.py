"""Sidecar collector for runtime bridge episodes.

The collector is deliberately passive: callers decide when to invoke
`observe()`. It only reads state and writes bridge training data.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from .recorder import BridgeEpisodeRecorder, RecorderConfig, StepSignals, TaskSpec
from .snapshot import BridgeSnapshotBuilder


class BridgeSidecarCollector:
    """Passive runtime collector around `BridgeEpisodeRecorder`."""

    def __init__(
        self,
        output_dir: str | Path,
        task_spec: TaskSpec | None = None,
        config: RecorderConfig | None = None,
        episode_id: str | None = None,
        min_interval_s: float = 0.2,
    ):
        self.config = config or RecorderConfig()
        self.task_spec = task_spec or TaskSpec()
        self.recorder = BridgeEpisodeRecorder(
            output_dir=output_dir,
            config=self.config,
            episode_id=episode_id,
            task_spec=self.task_spec,
        )
        self.builder = BridgeSnapshotBuilder(self.config, self.task_spec)
        self.min_interval_s = float(min_interval_s)
        self._last_time = 0.0
        self._last_pos: np.ndarray | None = None
        self._last_distance: float | None = None
        self._last_visited_ratio: float | None = None
        self._last_step_count = 0

    @property
    def num_steps(self) -> int:
        return len(self.recorder.steps)

    def start_segment(self) -> None:
        self.builder.start_segment()
        self._last_pos = None
        self._last_distance = None
        self._last_visited_ratio = None

    def finish_segment(self) -> None:
        self.builder.finish_segment()

    def observe(
        self,
        env: Any,
        memory: Any = None,
        topo_map: Any = None,
        skill: str = "idle",
        force: bool = False,
    ) -> bool:
        now = time.time()
        if not force and now - self._last_time < self.min_interval_s:
            return False

        snapshot = self.builder.build(
            env=env,
            memory=memory,
            topo_map=topo_map,
            prev_pos=self._last_pos,
            prev_distance=self._last_distance,
            prev_visited_ratio=self._last_visited_ratio,
        )
        info_gain = 0.0
        if self._last_visited_ratio is not None:
            info_gain = max(0.0, snapshot.visited_ratio - self._last_visited_ratio) * 20.0
        movement = 0.0 if self._last_pos is None else float(np.linalg.norm(snapshot.robot_pos - self._last_pos))
        stuck_score = float(np.clip(1.0 - movement * 4.0, 0.0, 1.0)) if self._last_pos is not None else 0.0
        progress_rate = 0.0
        if self._last_distance is not None:
            progress_rate = self._last_distance - snapshot.distance_to_subgoal

        signals = StepSignals(
            distance_to_subgoal=snapshot.distance_to_subgoal,
            progress_rate=progress_rate,
            stuck_score=stuck_score,
            info_gain=info_gain,
            target_found=snapshot.target_found,
            path_valid=snapshot.path_valid,
            need_scan=False,
        )
        self.recorder.record_step(
            maps=snapshot.maps,
            state=snapshot.state,
            memory=snapshot.memory,
            skill=skill,
            candidates=snapshot.candidates,
            signals=signals,
        )

        self._last_time = now
        self._last_pos = snapshot.robot_pos.copy()
        self._last_distance = snapshot.distance_to_subgoal
        self._last_visited_ratio = snapshot.visited_ratio
        self._last_step_count = self.num_steps
        return True

    def save(self) -> Path:
        return self.recorder.save()

    def mark_progress_guard_triggered(
        self,
        start_step: int,
        *,
        reason: str,
        max_backfill_steps: int = 8,
    ) -> int:
        """Record a progress-guard intervention as trainable hindsight."""
        end_step = self.num_steps
        if end_step <= start_step:
            return 0
        triggers = self.recorder.metadata.setdefault("progress_guard_triggers", [])
        triggers.append(
            {
                "start_step": int(start_step),
                "end_step": int(end_step),
                "reason": str(reason),
            }
        )
        start = max(int(start_step), end_step - max(1, int(max_backfill_steps)))
        return self.recorder.relabel_steps(
            start,
            end_step,
            event="path_invalidated",
            replan="switch_subgoal",
            success=0.0,
            stuck=1.0,
            cost=0.95,
        )

    def mark_skill_outcome(
        self,
        start_step: int,
        *,
        success: bool,
        final_distance: float | None = None,
        max_backfill_steps: int = 8,
    ) -> int:
        """Add hindsight labels to the end of a just-finished skill segment."""
        end_step = self.num_steps
        if end_step <= start_step:
            return 0
        start = max(int(start_step), end_step - max(1, int(max_backfill_steps)))
        if success:
            return self.recorder.relabel_steps(
                max(end_step - 1, int(start_step)),
                end_step,
                event="subgoal_completed",
                replan="interrupt_and_scan",
                success=1.0,
                cost=0.0 if final_distance is None else min(float(final_distance) / 10.0, 1.0),
            )
        return self.recorder.relabel_steps(
            start,
            end_step,
            event="path_invalidated",
            replan="switch_subgoal",
            success=0.0,
            cost=0.85 if final_distance is None else min(0.5 + float(final_distance) / 10.0, 1.0),
        )
