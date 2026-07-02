#!/usr/bin/env python3
"""Smoke-test hook-level intervention result override without MuJoCo."""

from __future__ import annotations

try:
    from scripts._bootstrap import ensure_project_root
except ModuleNotFoundError:
    from _bootstrap import ensure_project_root

ensure_project_root()

from bridge.advisor import BridgeDecision
from bridge.arbitration import BridgeIntervention
from bridge.hooks import install_navigation_sidecar_hook, restore_navigation_hook


def fake_decision() -> BridgeDecision:
    return BridgeDecision(
        event="navigation_stuck",
        replan="switch_subgoal",
        event_confidence=0.99,
        replan_confidence=0.99,
        success_prob=0.0,
        stuck_prob=0.95,
        target_found_prob=0.0,
        cost=0.0,
        info_gain=0.0,
        candidate_index=None,
        candidate_score=None,
        candidate_xy_norm=None,
        event_top3=[("navigation_stuck", 0.99)],
        replan_top3=[("switch_subgoal", 0.99)],
    )


class FakeEnv:
    pass


class FakeNavSkill:
    def __init__(self):
        self.env = FakeEnv()

    def go_to(self, *args, **kwargs):
        callback = kwargs["step_callback"]
        if callback():
            return True, 4.2
        return True, 0.0


class FakeCollector:
    def observe(self, *args, **kwargs):
        return True


class FakeAdvisor:
    def observe_runtime(self, *args, **kwargs):
        return fake_decision()


class FakePolicy:
    def reset(self):
        pass

    def update(self, decision):
        return BridgeIntervention(
            should_stop=True,
            reason=f"{decision.event}/{decision.replan}",
            override_success=False,
            decision=decision,
        )


def main() -> None:
    nav = FakeNavSkill()
    original = install_navigation_sidecar_hook(
        nav,
        FakeCollector(),
        advisor=FakeAdvisor(),
        intervention_policy=FakePolicy(),
    )
    try:
        success, dist = nav.go_to(1.0, 2.0, step_callback=lambda: False)
        assert success is False, (success, dist)
        assert abs(dist - 4.2) < 1e-6, (success, dist)
    finally:
        restore_navigation_hook(nav, original)
    print("hook_arbitration_smoke_ok")


if __name__ == "__main__":
    main()
