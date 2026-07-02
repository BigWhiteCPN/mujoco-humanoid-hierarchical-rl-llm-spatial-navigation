#!/usr/bin/env python3
"""Smoke-test hindsight relabeling for skill outcomes."""

from __future__ import annotations

try:
    from scripts._bootstrap import ensure_project_root
except ModuleNotFoundError:
    from _bootstrap import ensure_project_root

ensure_project_root()

import numpy as np

from bridge.constants import EVENT_TO_ID, REPLAN_TO_ID
from bridge.sidecar import BridgeSidecarCollector


def record_dummy_steps(collector: BridgeSidecarCollector, count: int) -> None:
    cfg = collector.config
    for _ in range(count):
        collector.recorder.record_step(
            maps=np.zeros((cfg.map_channels, cfg.map_size, cfg.map_size), dtype=np.float32),
            state=np.zeros((cfg.state_dim,), dtype=np.float32),
            memory=np.zeros((cfg.memory_dim,), dtype=np.float32),
            skill="explore_frontier",
            candidates=np.zeros((cfg.max_candidates, cfg.candidate_dim), dtype=np.float32),
        )


def main() -> None:
    collector = BridgeSidecarCollector("/tmp/bridge_hindsight_smoke")
    record_dummy_steps(collector, 10)
    changed = collector.mark_skill_outcome(2, success=False, final_distance=4.0, max_backfill_steps=3)
    assert changed == 3, changed
    events = [step["event"] for step in collector.recorder.steps]
    replans = [step["replan"] for step in collector.recorder.steps]
    assert events[-3:] == [EVENT_TO_ID["path_invalidated"]] * 3, events
    assert replans[-3:] == [REPLAN_TO_ID["switch_subgoal"]] * 3, replans
    changed = collector.mark_progress_guard_triggered(0, reason="far_after_patience", max_backfill_steps=2)
    assert changed == 2, changed
    assert collector.recorder.steps[-1]["event"] == EVENT_TO_ID["path_invalidated"]
    assert collector.recorder.metadata["progress_guard_triggers"][-1]["reason"] == "far_after_patience"

    changed = collector.mark_skill_outcome(0, success=True, final_distance=0.4, max_backfill_steps=3)
    assert changed == 1, changed
    assert collector.recorder.steps[-1]["event"] == EVENT_TO_ID["subgoal_completed"]
    assert collector.recorder.steps[-1]["replan"] == REPLAN_TO_ID["interrupt_and_scan"]
    print("hindsight_smoke_ok")


if __name__ == "__main__":
    main()
