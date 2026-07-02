#!/usr/bin/env python3
"""Run multiple non-invasive runtime collection episodes sequentially."""

from __future__ import annotations

try:
    from scripts._bootstrap import ensure_project_root
except ModuleNotFoundError:
    from _bootstrap import ensure_project_root

ensure_project_root()

import argparse
import subprocess
import sys
from pathlib import Path


TASKS = [
    {
        "name": "meeting_room",
        "intent": "search_place",
        "target": "meeting_room",
        "constraint": "stop_when_found",
        "target_place": "会议室",
    },
    {
        "name": "pantry",
        "intent": "search_place",
        "target": "pantry",
        "constraint": "stop_when_found",
        "target_place": "茶水间",
    },
    {
        "name": "explore",
        "intent": "explore",
        "target": "none",
        "constraint": "prefer_safe",
        "target_place": "",
    },
    {
        "name": "door",
        "intent": "search_place",
        "target": "door",
        "constraint": "stop_when_found",
        "target_place": "大门",
    },
    {
        "name": "office",
        "intent": "search_place",
        "target": "office",
        "constraint": "stop_when_found",
        "target_place": "老板办公室",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect multiple runtime bridge episodes.")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--output-dir", default="data/runtime_v3")
    parser.add_argument("--agent-root", default="../agent_system_complex_version")
    parser.add_argument("--max-rounds", type=int, default=1)
    parser.add_argument("--nav-steps-per-round", type=int, default=400)
    parser.add_argument("--render-mode", default="rgb_array")
    parser.add_argument("--mujoco-gl", default="egl")
    parser.add_argument("--callback-freq", type=int, default=20)
    parser.add_argument("--observe-interval-s", type=float, default=0.1)
    parser.add_argument("--dynamic-obstacles", action="store_true")
    parser.add_argument("--task-offset", type=int, default=0)
    parser.add_argument("--seed-base", type=int, default=None)
    parser.add_argument("--advisor-checkpoint", default="")
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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    failures = 0
    for offset in range(args.episodes):
        episode_num = args.start_index + offset
        task = TASKS[(offset + args.task_offset) % len(TASKS)]
        episode_id = f"batch_{episode_num:04d}_{task['name']}"
        cmd = [
            sys.executable,
            "-m",
            "scripts.collect_runtime",
            "--agent-root",
            args.agent_root,
            "--output-dir",
            str(output_dir),
            "--episode-id",
            episode_id,
            "--max-rounds",
            str(args.max_rounds),
            "--nav-steps-per-round",
            str(args.nav_steps_per_round),
            "--render-mode",
            args.render_mode,
            "--mujoco-gl",
            args.mujoco_gl,
            "--callback-freq",
            str(args.callback_freq),
            "--observe-interval-s",
            str(args.observe_interval_s),
            "--intent",
            task["intent"],
            "--target",
            task["target"],
            "--constraint",
            task["constraint"],
            "--raw-task",
            f"batch_{task['name']}",
        ]
        if args.seed_base is not None:
            cmd.extend(["--seed", str(args.seed_base + offset)])
        if task["target_place"]:
            cmd.extend(["--target-place", task["target_place"]])
        if args.dynamic_obstacles:
            cmd.append("--dynamic-obstacles")
        if args.disable_hindsight_labels:
            cmd.append("--disable-hindsight-labels")
        cmd.extend(["--hindsight-backfill-steps", str(args.hindsight_backfill_steps)])
        if args.progress_guard:
            cmd.extend(
                [
                    "--progress-guard",
                    "--progress-guard-warmup",
                    str(args.progress_guard_warmup),
                    "--progress-guard-window",
                    str(args.progress_guard_window),
                    "--progress-guard-min-progress",
                    str(args.progress_guard_min_progress),
                    "--progress-guard-min-distance",
                    str(args.progress_guard_min_distance),
                    "--progress-guard-max-far-steps",
                    str(args.progress_guard_max_far_steps),
                    "--progress-guard-far-distance",
                    str(args.progress_guard_far_distance),
                ]
            )
        if args.advisor_checkpoint:
            cmd.extend(
                [
                    "--advisor-checkpoint",
                    args.advisor_checkpoint,
                    "--advisor-device",
                    args.advisor_device,
                    "--advisor-log-every",
                    str(args.advisor_log_every),
                    "--advisor-control",
                    args.advisor_control,
                    "--advisor-stop-confidence",
                    str(args.advisor_stop_confidence),
                    "--advisor-stop-consecutive",
                    str(args.advisor_stop_consecutive),
                    "--advisor-warmup-steps",
                    str(args.advisor_warmup_steps),
                    "--advisor-stuck-prob",
                    str(args.advisor_stuck_prob),
                    "--advisor-target-prob",
                    str(args.advisor_target_prob),
                    "--advisor-risk-prob",
                    str(args.advisor_risk_prob),
                ]
            )

        print(f"\n[batch] episode {offset + 1}/{args.episodes}: {episode_id}", flush=True)
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            failures += 1
            print(f"[batch] failed {episode_id}: returncode={result.returncode}", flush=True)

    if failures:
        raise SystemExit(f"{failures}/{args.episodes} collection episodes failed")
    print(f"[batch] collected {args.episodes} episodes into {output_dir}", flush=True)


if __name__ == "__main__":
    main()
