#!/usr/bin/env python3
"""Smoke-test advisor intervention policy gates."""

from __future__ import annotations

try:
    from scripts._bootstrap import ensure_project_root
except ModuleNotFoundError:
    from _bootstrap import ensure_project_root

ensure_project_root()

from bridge.advisor import BridgeDecision
from bridge.arbitration import BridgeInterventionConfig, BridgeInterventionPolicy


def decision(event: str, replan: str, stuck: float = 0.0, target: float = 0.0, risk: float = 0.0) -> BridgeDecision:
    return BridgeDecision(
        event=event,
        replan=replan,
        event_confidence=0.95,
        replan_confidence=0.95,
        success_prob=0.0,
        stuck_prob=stuck,
        target_found_prob=target,
        cost=0.0,
        info_gain=0.0,
        candidate_index=0,
        candidate_score=0.0,
        candidate_xy_norm=(0.0, 0.0),
        event_top3=[(event, 0.95)],
        replan_top3=[(replan, 0.95)],
        failure_risk_prob=risk,
    )


def expect_stop(policy: BridgeInterventionPolicy, items: list[BridgeDecision], expected_success: bool | None) -> None:
    result = None
    for item in items:
        result = policy.update(item)
    if expected_success is None:
        assert result is None, result
    else:
        assert result is not None, "expected intervention"
        assert result.override_success is expected_success, result


def main() -> None:
    base = BridgeInterventionConfig(confidence_threshold=0.90, consecutive_steps=2, warmup_steps=2)

    off = BridgeInterventionPolicy(BridgeInterventionConfig(mode="off"))
    expect_stop(
        off,
        [decision("navigation_stuck", "switch_subgoal", stuck=0.95) for _ in range(4)],
        None,
    )

    safe = BridgeInterventionPolicy(BridgeInterventionConfig(**{**base.__dict__, "mode": "safe"}))
    expect_stop(
        safe,
        [decision("target_candidate_found", "go_to_target_candidate", target=0.95) for _ in range(2)],
        True,
    )
    safe.reset()
    expect_stop(
        safe,
        [decision("navigation_stuck", "switch_subgoal", stuck=0.95) for _ in range(4)],
        None,
    )

    replan = BridgeInterventionPolicy(BridgeInterventionConfig(**{**base.__dict__, "mode": "replan"}))
    expect_stop(
        replan,
        [decision("navigation_stuck", "switch_subgoal", stuck=0.95) for _ in range(2)],
        False,
    )

    risk = BridgeInterventionPolicy(BridgeInterventionConfig(**{**base.__dict__, "mode": "risk"}))
    expect_stop(
        risk,
        [decision("subgoal_completed", "interrupt_and_scan") for _ in range(2)],
        None,
    )
    risk.reset()
    expect_stop(
        risk,
        [decision("low_information_gain", "switch_subgoal") for _ in range(2)],
        False,
    )
    risk.reset()
    expect_stop(
        risk,
        [decision("continue", "continue_current", risk=0.9) for _ in range(2)],
        False,
    )
    print("arbitration_smoke_ok")


if __name__ == "__main__":
    main()
