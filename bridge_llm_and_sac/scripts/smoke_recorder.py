#!/usr/bin/env python3
"""Create and validate one recorder episode without touching the robot stack."""

from __future__ import annotations

try:
    from scripts._bootstrap import ensure_project_root
except ModuleNotFoundError:
    from _bootstrap import ensure_project_root

ensure_project_root()

from pathlib import Path

import numpy as np

from bridge.recorder import BridgeEpisodeRecorder, StepSignals, TaskSpec
from scripts.validate_episode import validate_file


def main() -> None:
    out_dir = Path("data/recorder_smoke")
    recorder = BridgeEpisodeRecorder(
        out_dir,
        episode_id="smoke",
        task_spec=TaskSpec(intent="search_place", target="meeting_room", constraint="stop_when_found"),
    )
    rng = np.random.default_rng(3)
    for step in range(12):
        maps = rng.random((5, 64, 64), dtype=np.float32)
        state = rng.random((recorder.config.state_dim,), dtype=np.float32)
        memory = rng.random((12,), dtype=np.float32)
        candidates = rng.random((4, 8), dtype=np.float32)
        recorder.record_step(
            maps=maps,
            state=state,
            memory=memory,
            skill="explore_frontier",
            candidates=candidates,
            signals=StepSignals(
                distance_to_subgoal=float(max(0.0, 4.0 - step * 0.3)),
                stuck_score=0.2 if step < 8 else 0.86,
                info_gain=0.2 if step < 9 else 0.03,
                target_found=step == 6,
            ),
        )
    path = recorder.save()
    errors = validate_file(path)
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"recorder smoke ok: {path}")


if __name__ == "__main__":
    main()
