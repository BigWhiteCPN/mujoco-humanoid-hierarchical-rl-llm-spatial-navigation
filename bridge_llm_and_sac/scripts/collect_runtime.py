#!/usr/bin/env python3
"""Collect real runtime bridge episodes without modifying the robot package."""

from __future__ import annotations

try:
    from scripts._bootstrap import ensure_project_root
except ModuleNotFoundError:
    from _bootstrap import ensure_project_root

ensure_project_root()

import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

from bridge.hooks import install_navigation_sidecar_hook, restore_navigation_hook
from bridge.recorder import TaskSpec
from bridge.sidecar import BridgeSidecarCollector


DEFAULT_MODEL_XML = "resources/mjcf/Linnxil_fifteen_angle_bs_copy_20260302_copy.xml"
DEFAULT_LOW_LEVEL_POLICY = "models/policy_20251026.pt"
DEFAULT_SAC_MODEL = "models/sac_lidar_interrupted_good3_0.91.zip"


def resolve_path(root: Path, path: str) -> str:
    p = Path(path)
    return str(p if p.is_absolute() else root / p)


def configure_runtime(args) -> None:
    thread_count = str(max(1, int(args.cpu_threads)))
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[name] = thread_count
    os.environ["MUJOCO_GL"] = args.mujoco_gl
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


def task_from_args(args) -> TaskSpec:
    return TaskSpec(intent=args.intent, target=args.target, constraint=args.constraint, raw_text=args.raw_task)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect sidecar bridge data from the MuJoCo robot runtime.")
    parser.add_argument("--agent-root", default="../agent_system_complex_version")
    parser.add_argument("--output-dir", default="data/runtime")
    parser.add_argument("--episode-id", default=None)
    parser.add_argument("--robot-model-xml", default=DEFAULT_MODEL_XML)
    parser.add_argument("--low-level-policy", default=DEFAULT_LOW_LEVEL_POLICY)
    parser.add_argument("--sac-model", default=DEFAULT_SAC_MODEL)
    parser.add_argument("--render-mode", default="rgb_array", choices=["rgb_array", "human", "dashboard", "fast_dashboard"])
    parser.add_argument("--mujoco-gl", default="egl")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--render-decimation", type=int, default=50)
    parser.add_argument("--action-repeat", type=int, default=4)
    parser.add_argument("--history-length", type=int, default=15)
    parser.add_argument("--dynamic-obstacles", action="store_true")
    parser.add_argument("--max-rounds", type=int, default=2)
    parser.add_argument("--nav-steps-per-round", type=int, default=800)
    parser.add_argument("--target-place", default="")
    parser.add_argument("--direction-hint", default="")
    parser.add_argument("--observe-interval-s", type=float, default=0.2)
    parser.add_argument("--callback-freq", type=int, default=40)
    parser.add_argument("--intent", default="search_place")
    parser.add_argument("--target", default="meeting_room")
    parser.add_argument("--constraint", default="stop_when_found")
    parser.add_argument("--raw-task", default="runtime sidecar collection")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--show-landmark-debug", action="store_true")
    parser.add_argument("--advisor-checkpoint", default="", help="Optional BridgeNet checkpoint for online diagnostics.")
    parser.add_argument("--advisor-device", default="cpu")
    parser.add_argument("--advisor-log-every", type=int, default=20)
    parser.add_argument("--advisor-control", choices=["off", "safe", "risk", "replan"], default="off")
    parser.add_argument("--advisor-stop-confidence", type=float, default=0.88)
    parser.add_argument("--advisor-stop-consecutive", type=int, default=3)
    parser.add_argument("--advisor-warmup-steps", type=int, default=8)
    parser.add_argument("--advisor-stuck-prob", type=float, default=0.70)
    parser.add_argument("--advisor-target-prob", type=float, default=0.70)
    parser.add_argument("--advisor-risk-prob", type=float, default=0.75)
    parser.add_argument("--progress-guard", action="store_true")
    parser.add_argument("--progress-guard-warmup", type=int, default=4)
    parser.add_argument("--progress-guard-window", type=int, default=4)
    parser.add_argument("--progress-guard-min-progress", type=float, default=0.25)
    parser.add_argument("--progress-guard-min-distance", type=float, default=1.5)
    parser.add_argument("--progress-guard-max-far-steps", type=int, default=0)
    parser.add_argument("--progress-guard-far-distance", type=float, default=3.0)
    parser.add_argument("--disable-hindsight-labels", action="store_true")
    parser.add_argument("--hindsight-backfill-steps", type=int, default=8)
    args = parser.parse_args()

    configure_runtime(args)
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    agent_root = Path(args.agent_root)
    if not agent_root.is_absolute():
        agent_root = (Path.cwd() / agent_root).resolve()
    sys.path.insert(0, str(agent_root))

    from agent_env import AgentVisualEnv
    from memory import SpatialMemory, TopologicalMap
    from skills import FrontierExplorationSkill, NavigationSkill, PerceptionSkill

    model_xml = resolve_path(agent_root, args.robot_model_xml)
    low_level_policy = resolve_path(agent_root, args.low_level_policy)
    sac_model = resolve_path(agent_root, args.sac_model)
    for path in (model_xml, low_level_policy, sac_model):
        if not Path(path).exists():
            raise FileNotFoundError(path)

    env = None
    original_go_to = None
    task_spec = task_from_args(args)
    collector = BridgeSidecarCollector(
        output_dir=args.output_dir,
        episode_id=args.episode_id or time.strftime("%Y%m%d_%H%M%S"),
        task_spec=task_spec,
        min_interval_s=args.observe_interval_s,
    )
    advisor = None
    intervention_policy = None
    progress_guard = None
    if args.advisor_checkpoint:
        from bridge.advisor import BridgeAdvisor

        advisor = BridgeAdvisor(args.advisor_checkpoint, device=args.advisor_device)
        print(f"[advisor] loaded {args.advisor_checkpoint} device={args.advisor_device}")
        if args.advisor_control != "off":
            from bridge.arbitration import BridgeInterventionConfig, BridgeInterventionPolicy

            intervention_policy = BridgeInterventionPolicy(
                BridgeInterventionConfig(
                    mode=args.advisor_control,
                    confidence_threshold=args.advisor_stop_confidence,
                    consecutive_steps=args.advisor_stop_consecutive,
                    warmup_steps=args.advisor_warmup_steps,
                    stuck_prob_threshold=args.advisor_stuck_prob,
                    target_prob_threshold=args.advisor_target_prob,
                    failure_risk_threshold=args.advisor_risk_prob,
                )
            )
            print(
                "[advisor-control] "
                f"mode={args.advisor_control} conf={args.advisor_stop_confidence} "
                f"consecutive={args.advisor_stop_consecutive} warmup={args.advisor_warmup_steps}"
            )
    if args.progress_guard:
        from bridge.arbitration import SegmentProgressGuard, SegmentProgressGuardConfig

        progress_guard = SegmentProgressGuard(
            SegmentProgressGuardConfig(
                enabled=True,
                warmup_steps=args.progress_guard_warmup,
                window_steps=args.progress_guard_window,
                min_progress_m=args.progress_guard_min_progress,
                min_distance_m=args.progress_guard_min_distance,
                max_far_steps=args.progress_guard_max_far_steps,
                far_distance_m=args.progress_guard_far_distance,
            )
        )
        print(
            "[progress-guard] "
            f"enabled warmup={args.progress_guard_warmup} window={args.progress_guard_window} "
            f"min_progress={args.progress_guard_min_progress} min_dist={args.progress_guard_min_distance} "
            f"max_far_steps={args.progress_guard_max_far_steps} far_dist={args.progress_guard_far_distance}"
        )
    try:
        env = AgentVisualEnv(
            model_path=model_xml,
            low_level_policy_path=low_level_policy,
            render_mode=args.render_mode,
            render_decimation=args.render_decimation,
            action_repeat=args.action_repeat,
            history_length=args.history_length,
            enable_dynamic_obstacles=bool(args.dynamic_obstacles),
            enable_mujoco_viewer=False,
        )
        env.reset(seed=args.seed)
        if args.show_landmark_debug and hasattr(env, "get_landmark_positions_debug"):
            print(env.get_landmark_positions_debug())

        memory = SpatialMemory(save_dir=str(Path(args.output_dir) / "memory_logs"))
        topo_map = TopologicalMap(fingerprint_radius_m=3.0, fingerprint_size=8, match_threshold=0.85)
        nav_skill = NavigationSkill(env, sac_model)
        percept_skill = PerceptionSkill(env, memory)
        explore_skill = FrontierExplorationSkill(env, nav_skill, percept_skill, topo_map=topo_map)
        original_go_to = install_navigation_sidecar_hook(
            nav_skill,
            collector,
            memory=memory,
            topo_map=topo_map,
            skill_name="explore_frontier",
            callback_freq=args.callback_freq,
            advisor=advisor,
            advisor_task_spec=task_spec,
            advisor_log_every=args.advisor_log_every,
            intervention_policy=intervention_policy,
            progress_guard=progress_guard,
            hindsight_labels=not args.disable_hindsight_labels,
            hindsight_backfill_steps=args.hindsight_backfill_steps,
        )

        collector.observe(env, memory=memory, topo_map=topo_map, skill="idle", force=True)
        target_place = args.target_place.strip() or None
        direction_hint = args.direction_hint.strip() or None
        print(
            f"[collector] start max_rounds={args.max_rounds} nav_steps_per_round={args.nav_steps_per_round} "
            f"target_place={target_place}"
        )
        log = explore_skill.execute(
            max_rounds=args.max_rounds,
            nav_steps_per_round=args.nav_steps_per_round,
            target_place=target_place,
            direction_hint=direction_hint,
        )
        print("[collector] exploration log:")
        print(log)
        collector.observe(env, memory=memory, topo_map=topo_map, skill="idle", force=True)
        episode_path = collector.save()
        print(f"[collector] saved {episode_path} steps={collector.num_steps}")
    finally:
        if original_go_to is not None:
            try:
                restore_navigation_hook(nav_skill, original_go_to)
            except Exception:
                pass
        if env is not None and hasattr(env, "close"):
            env.close()


if __name__ == "__main__":
    main()
