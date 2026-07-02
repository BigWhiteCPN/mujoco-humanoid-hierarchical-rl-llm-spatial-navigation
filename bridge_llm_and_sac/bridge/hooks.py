"""Runtime hooks that collect bridge data without editing the robot stack."""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable

import numpy as np


def _goal_distance(env: Any) -> float | None:
    goal = None
    for name in ("agent_assigned_target", "goal_pos", "current_target_waypoint"):
        value = getattr(env, name, None)
        if value is not None:
            arr = np.asarray(value, dtype=np.float32).reshape(-1)
            if arr.shape[0] >= 2:
                goal = arr[:2]
                break
    if goal is None:
        return None
    try:
        robot = np.asarray(env.data.xpos[env.robot_base_body_id][:2], dtype=np.float32)
    except Exception:
        return None
    return float(np.linalg.norm(robot - goal))


def install_navigation_sidecar_hook(
    nav_skill: Any,
    collector: Any,
    memory: Any = None,
    topo_map: Any = None,
    skill_name: str = "go_to",
    callback_freq: int | None = None,
    advisor: Any = None,
    advisor_task_spec: Any = None,
    advisor_log_every: int = 0,
    intervention_policy: Any = None,
    progress_guard: Any = None,
    hindsight_labels: bool = True,
    hindsight_backfill_steps: int = 8,
) -> Callable:
    """Wrap one NavigationSkill instance so every go_to rollout records snapshots.

    This mutates only the passed instance. It does not modify class definitions
    or source files. The original method is returned so callers can restore it.
    """

    original_go_to = nav_skill.go_to
    advisor_step = 0
    pending_intervention = None

    def observe_all(env: Any, force: bool = False, allow_intervention: bool = True):
        nonlocal advisor_step
        collector.observe(env, memory=memory, topo_map=topo_map, skill=skill_name, force=force)
        if advisor is None:
            return None
        decision = advisor.observe_runtime(
            env,
            memory=memory,
            topo_map=topo_map,
            task_spec=advisor_task_spec,
            skill=skill_name,
        )
        advisor_step += 1
        if advisor_log_every > 0 and (force or advisor_step % advisor_log_every == 0):
            print(
                "[advisor] "
                f"step={advisor_step} event={decision.event}({decision.event_confidence:.2f}) "
                f"replan={decision.replan}({decision.replan_confidence:.2f}) "
                f"stuck={decision.stuck_prob:.2f} target={decision.target_found_prob:.2f} "
                f"risk={getattr(decision, 'failure_risk_prob', 0.0):.2f} "
                f"candidate={decision.candidate_index}"
            )
        if intervention_policy is None or not allow_intervention:
            return None
        return intervention_policy.update(decision)

    @wraps(original_go_to)
    def wrapped_go_to(*args, **kwargs):
        nonlocal pending_intervention
        env = getattr(nav_skill, "env", None)
        user_callback = kwargs.get("step_callback")
        user_freq = int(kwargs.get("callback_freq", 50))
        if callback_freq is not None:
            kwargs["callback_freq"] = max(1, min(user_freq, int(callback_freq)))
        pending_intervention = None
        hindsight_outcome = None
        if intervention_policy is not None:
            intervention_policy.reset()
        if progress_guard is not None:
            progress_guard.reset()
        segment_start = int(getattr(collector, "num_steps", 0))
        progress_guard_reason = None
        for segment_owner in (collector, advisor):
            if hasattr(segment_owner, "start_segment"):
                try:
                    segment_owner.start_segment()
                except Exception:
                    pass

        def sidecar_callback():
            nonlocal pending_intervention, progress_guard_reason
            stop = False
            if user_callback is not None:
                stop = bool(user_callback())
            if env is not None:
                intervention = observe_all(env)
                if not stop and user_callback is not None and intervention is not None and intervention.should_stop:
                    pending_intervention = intervention
                    print(
                        "[advisor-control] "
                        f"stop reason={intervention.reason} override_success={intervention.override_success}"
                    )
                    stop = True
                if not stop and user_callback is not None and progress_guard is not None:
                    reason = progress_guard.update(_goal_distance(env))
                    if reason is not None:
                        progress_guard_reason = reason
                        pending_intervention = type(
                            "ProgressGuardIntervention",
                            (),
                            {"override_success": False, "reason": reason},
                        )()
                        print(f"[progress-guard] stop reason={reason} override_success=False")
                        stop = True
            return stop

        kwargs["step_callback"] = sidecar_callback
        if env is not None:
            observe_all(env, force=True, allow_intervention=False)
        try:
            result = original_go_to(*args, **kwargs)
            if pending_intervention is not None and pending_intervention.override_success is not None:
                try:
                    _, dist = result
                    result = bool(pending_intervention.override_success), dist
                except Exception:
                    pass
            if hindsight_labels and user_callback is not None and hasattr(collector, "mark_skill_outcome"):
                try:
                    success, dist = result
                    hindsight_outcome = (bool(success), float(dist))
                except Exception:
                    pass
            return result
        finally:
            if env is not None:
                observe_all(env, force=True, allow_intervention=False)
            if progress_guard_reason is not None and hasattr(collector, "mark_progress_guard_triggered"):
                try:
                    collector.mark_progress_guard_triggered(
                        segment_start,
                        reason=progress_guard_reason,
                        max_backfill_steps=hindsight_backfill_steps,
                    )
                except Exception:
                    pass
            if hindsight_outcome is not None:
                try:
                    success, dist = hindsight_outcome
                    collector.mark_skill_outcome(
                        segment_start,
                        success=success,
                        final_distance=dist,
                        max_backfill_steps=hindsight_backfill_steps,
                    )
                except Exception:
                    pass
            for segment_owner in (collector, advisor):
                if hasattr(segment_owner, "finish_segment"):
                    try:
                        segment_owner.finish_segment()
                    except Exception:
                        pass

    nav_skill.go_to = wrapped_go_to
    return original_go_to


def restore_navigation_hook(nav_skill: Any, original_go_to: Callable) -> None:
    nav_skill.go_to = original_go_to
