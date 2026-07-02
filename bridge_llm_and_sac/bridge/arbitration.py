"""Decision gating for BridgeAdvisor outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .advisor import BridgeDecision


@dataclass(frozen=True)
class BridgeIntervention:
    should_stop: bool
    reason: str
    override_success: bool | None
    decision: "BridgeDecision"


@dataclass
class BridgeInterventionConfig:
    mode: str = "off"
    confidence_threshold: float = 0.85
    consecutive_steps: int = 3
    warmup_steps: int = 8
    stuck_prob_threshold: float = 0.65
    target_prob_threshold: float = 0.70
    failure_risk_threshold: float = 0.75


class BridgeInterventionPolicy:
    """Convert repeated advisor decisions into optional navigation stops.

    Modes:
        off: never stop, diagnostics only.
        safe: stop only for target/subgoal completion; keep success=True.
        risk: stop only repeated stuck/low-info/path-invalidated rollouts;
            override the wrapped go_to result to success=False.
        replan: also stop stuck/low-info/path-invalidated rollouts; override
            the wrapped go_to result to success=False.
    """

    SUCCESS_EVENTS = {"subgoal_completed", "target_candidate_found"}
    SUCCESS_REPLANS = {"interrupt_and_scan", "go_to_target_candidate"}
    FAILURE_EVENTS = {"navigation_stuck", "low_information_gain", "path_invalidated", "need_replan"}
    FAILURE_REPLANS = {"switch_subgoal", "ask_llm_replan", "declare_unreachable"}

    def __init__(self, config: BridgeInterventionConfig | None = None):
        self.config = config or BridgeInterventionConfig()
        self.step_count = 0
        self._last_key: tuple[str, str] | None = None
        self._run_length = 0

    def reset(self) -> None:
        self.step_count = 0
        self._last_key = None
        self._run_length = 0

    def update(self, decision: "BridgeDecision") -> BridgeIntervention | None:
        cfg = self.config
        self.step_count += 1
        risk_high = cfg.mode == "risk" and getattr(decision, "failure_risk_prob", 0.0) >= cfg.failure_risk_threshold
        key = ("failure_risk", "switch_subgoal") if risk_high else (decision.event, decision.replan)
        if key == self._last_key:
            self._run_length += 1
        else:
            self._last_key = key
            self._run_length = 1

        if cfg.mode == "off":
            return None
        if self.step_count < cfg.warmup_steps:
            return None
        if self._run_length < cfg.consecutive_steps:
            return None
        if risk_high:
            return BridgeIntervention(
                should_stop=True,
                reason=f"failure_risk/{getattr(decision, 'failure_risk_prob', 0.0):.2f}",
                override_success=False,
                decision=decision,
            )
        if decision.replan_confidence < cfg.confidence_threshold:
            return None

        if cfg.mode != "risk" and self._is_success_stop(decision):
            return BridgeIntervention(
                should_stop=True,
                reason=f"{decision.event}/{decision.replan}",
                override_success=True,
                decision=decision,
            )

        if cfg.mode in {"risk", "replan"} and self._is_failure_stop(decision):
            return BridgeIntervention(
                should_stop=True,
                reason=f"{decision.event}/{decision.replan}",
                override_success=False,
                decision=decision,
            )

        return None

    def _is_success_stop(self, decision: "BridgeDecision") -> bool:
        cfg = self.config
        if decision.event == "target_candidate_found" or decision.replan == "go_to_target_candidate":
            return decision.target_found_prob >= cfg.target_prob_threshold or decision.event_confidence >= cfg.confidence_threshold
        if decision.event in self.SUCCESS_EVENTS and decision.replan in self.SUCCESS_REPLANS:
            return decision.event_confidence >= cfg.confidence_threshold
        return False

    def _is_failure_stop(self, decision: "BridgeDecision") -> bool:
        cfg = self.config
        if decision.event == "navigation_stuck":
            return decision.stuck_prob >= cfg.stuck_prob_threshold
        return decision.event in self.FAILURE_EVENTS and decision.replan in self.FAILURE_REPLANS


@dataclass
class SegmentProgressGuardConfig:
    enabled: bool = False
    warmup_steps: int = 4
    window_steps: int = 4
    min_progress_m: float = 0.25
    min_distance_m: float = 1.5
    max_far_steps: int = 0
    far_distance_m: float = 3.0


class SegmentProgressGuard:
    """Conservative progress monitor for a single go_to segment."""

    def __init__(self, config: SegmentProgressGuardConfig | None = None):
        self.config = config or SegmentProgressGuardConfig()
        self.distances: list[float] = []

    def reset(self) -> None:
        self.distances.clear()

    def update(self, distance_m: float | None) -> str | None:
        cfg = self.config
        if not cfg.enabled or distance_m is None:
            return None
        distance = float(distance_m)
        self.distances.append(distance)
        if cfg.max_far_steps > 0 and len(self.distances) >= cfg.max_far_steps and distance > cfg.far_distance_m:
            return f"far_after_patience callbacks={len(self.distances)} dist={distance:.2f}m"
        if len(self.distances) < max(cfg.warmup_steps, cfg.window_steps + 1):
            return None
        if distance < cfg.min_distance_m:
            return None
        window = self.distances[-(cfg.window_steps + 1) :]
        progress = window[0] - min(window[1:])
        if progress < cfg.min_progress_m:
            return f"low_segment_progress progress={progress:.2f}m dist={distance:.2f}m"
        return None
