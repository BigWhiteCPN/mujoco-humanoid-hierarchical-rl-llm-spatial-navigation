#!/usr/bin/env python3
"""Smoke-test segment progress guard."""

from __future__ import annotations

try:
    from scripts._bootstrap import ensure_project_root
except ModuleNotFoundError:
    from _bootstrap import ensure_project_root

ensure_project_root()

from bridge.arbitration import SegmentProgressGuard, SegmentProgressGuardConfig


def main() -> None:
    guard = SegmentProgressGuard(
        SegmentProgressGuardConfig(
            enabled=True,
            warmup_steps=3,
            window_steps=2,
            min_progress_m=0.2,
            min_distance_m=1.0,
        )
    )
    assert guard.update(5.0) is None
    assert guard.update(4.9) is None
    assert guard.update(4.85) is not None
    guard.reset()
    assert guard.update(5.0) is None
    assert guard.update(4.5) is None
    assert guard.update(4.1) is None

    guard = SegmentProgressGuard(
        SegmentProgressGuardConfig(
            enabled=True,
            max_far_steps=3,
            far_distance_m=3.0,
        )
    )
    assert guard.update(5.0) is None
    assert guard.update(4.0) is None
    assert guard.update(3.5) is not None
    print("progress_guard_smoke_ok")


if __name__ == "__main__":
    main()
