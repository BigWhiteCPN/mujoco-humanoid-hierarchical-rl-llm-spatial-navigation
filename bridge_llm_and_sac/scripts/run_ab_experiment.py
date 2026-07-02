#!/usr/bin/env python3
"""Run paired off-vs-bridge-control runtime experiments."""

from __future__ import annotations

try:
    from scripts._bootstrap import ensure_project_root
except ModuleNotFoundError:
    from _bootstrap import ensure_project_root

ensure_project_root()

import argparse
import json
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from bridge.constants import EVENT_TYPES, REPLAN_ACTIONS
from scripts.collect_runtime_batch import TASKS


def build_collect_cmd(args, *, condition: str, task: dict, episode_id: str, seed: int, output_dir: Path) -> list[str]:
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
        f"ab_{condition}_{task['name']}",
        "--seed",
        str(seed),
    ]
    if task["target_place"]:
        cmd.extend(["--target-place", task["target_place"]])
    if args.dynamic_obstacles:
        cmd.append("--dynamic-obstacles")
    if args.disable_hindsight_labels:
        cmd.append("--disable-hindsight-labels")
    cmd.extend(["--hindsight-backfill-steps", str(args.hindsight_backfill_steps)])
    if condition == "replan":
        cmd.extend(
            [
                "--advisor-checkpoint",
                args.advisor_checkpoint,
                "--advisor-device",
                args.advisor_device,
                "--advisor-log-every",
                str(args.advisor_log_every),
                "--advisor-control",
                args.advisor_control_mode,
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
    return cmd


def stream_run(cmd: list[str], log_path: Path, prefix: str) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log_file.write(line)
            log_file.flush()
            print(f"{prefix} {line}", end="", flush=True)
        return int(process.wait())


def parse_log(log_text: str) -> dict[str, int | bool]:
    return {
        "advisor_control_stops": log_text.count("[advisor-control] stop"),
        "progress_guard_stops": log_text.count("[progress-guard] stop"),
        "callback_stops": log_text.count("step_callback 请求提前结束当前导航"),
        "nav_timeouts": log_text.count("导航超时"),
        "nav_arrivals": log_text.count("到达目标!"),
        "frontier_successes": log_text.count("已到达 frontier 目标点"),
        "frontier_blocks": log_text.count("导航受阻"),
        "target_complete": "目标集" in log_text and "已全部找齐" in log_text,
        "new_place_mentions": log_text.count("发现新地点"),
        "stuck_recoveries": log_text.count("检测到卡住"),
    }


def load_episode_metrics(output_dir: Path, episode_id: str) -> dict:
    episode_path = output_dir / "episodes" / f"episode_{episode_id}.npz"
    metrics: dict[str, object] = {"episode_path": str(episode_path), "episode_exists": episode_path.exists()}
    if not episode_path.exists():
        return metrics
    data = np.load(episode_path, allow_pickle=False)
    event_counts = Counter(data["event"].astype(int).tolist())
    replan_counts = Counter(data["replan"].astype(int).tolist())
    metrics.update(
        {
            "steps": int(data["event"].shape[0]),
            "event_counts": {EVENT_TYPES[k]: int(v) for k, v in sorted(event_counts.items())},
            "replan_counts": {REPLAN_ACTIONS[k]: int(v) for k, v in sorted(replan_counts.items())},
            "success_labels": int(np.asarray(data["success"]).sum()),
            "stuck_labels": int(np.asarray(data["stuck"]).sum()),
            "target_found_labels": int(np.asarray(data["target_found"]).sum()),
            "mean_cost": float(np.asarray(data["cost"], dtype=np.float32).mean()),
            "mean_info_gain": float(np.asarray(data["info_gain"], dtype=np.float32).mean()),
        }
    )
    return metrics


def summarize(results: list[dict]) -> dict:
    by_condition: dict[str, list[dict]] = {}
    for item in results:
        by_condition.setdefault(str(item["condition"]), []).append(item)

    summary = {"conditions": {}, "pairs": []}
    numeric_keys = [
        "steps",
        "advisor_control_stops",
        "progress_guard_stops",
        "callback_stops",
        "nav_timeouts",
        "nav_arrivals",
        "frontier_successes",
        "frontier_blocks",
        "new_place_mentions",
        "stuck_recoveries",
        "success_labels",
        "stuck_labels",
        "target_found_labels",
        "mean_cost",
        "mean_info_gain",
    ]
    for condition, items in by_condition.items():
        row = {"episodes": len(items), "returncode_failures": sum(int(i["returncode"] != 0) for i in items)}
        for key in numeric_keys:
            values = [float(i.get(key, 0.0)) for i in items if key in i]
            if values:
                row[f"{key}_mean"] = float(np.mean(values))
                row[f"{key}_sum"] = float(np.sum(values))
        row["target_complete_rate"] = float(np.mean([bool(i.get("target_complete", False)) for i in items]))
        summary["conditions"][condition] = row

    pair_ids = sorted({str(i["pair_id"]) for i in results})
    for pair_id in pair_ids:
        off = next((i for i in results if i["pair_id"] == pair_id and i["condition"] == "off"), None)
        replan = next((i for i in results if i["pair_id"] == pair_id and i["condition"] == "replan"), None)
        if not off or not replan:
            continue
        summary["pairs"].append(
            {
                "pair_id": pair_id,
                "task": off["task"],
                "seed": off["seed"],
                "steps_delta_replan_minus_off": int(replan.get("steps", 0)) - int(off.get("steps", 0)),
                "frontier_blocks_delta": int(replan.get("frontier_blocks", 0)) - int(off.get("frontier_blocks", 0)),
                "target_complete_off": bool(off.get("target_complete", False)),
                "target_complete_replan": bool(replan.get("target_complete", False)),
                "advisor_control_stops": int(replan.get("advisor_control_stops", 0)),
                "progress_guard_stops": int(replan.get("progress_guard_stops", 0)),
            }
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paired bridge-control A/B experiments.")
    parser.add_argument("--output-dir", default="data/ab_runtime_v1")
    parser.add_argument("--episodes", type=int, default=5, help="Number of paired tasks.")
    parser.add_argument("--seed-base", type=int, default=9100)
    parser.add_argument("--task-offset", type=int, default=0)
    parser.add_argument("--agent-root", default="../agent_system_complex_version")
    parser.add_argument("--max-rounds", type=int, default=1)
    parser.add_argument("--nav-steps-per-round", type=int, default=300)
    parser.add_argument("--render-mode", default="rgb_array")
    parser.add_argument("--mujoco-gl", default="egl")
    parser.add_argument("--callback-freq", type=int, default=20)
    parser.add_argument("--observe-interval-s", type=float, default=0.1)
    parser.add_argument("--dynamic-obstacles", action="store_true")
    parser.add_argument("--disable-hindsight-labels", action="store_true")
    parser.add_argument("--hindsight-backfill-steps", type=int, default=8)
    parser.add_argument("--advisor-checkpoint", default="runs/runtime_v2_v3_v4_finetune/best.pt")
    parser.add_argument("--advisor-control-mode", choices=["safe", "risk", "replan"], default="replan")
    parser.add_argument("--advisor-device", default="cpu")
    parser.add_argument("--advisor-log-every", type=int, default=0)
    parser.add_argument("--advisor-stop-confidence", type=float, default=0.90)
    parser.add_argument("--advisor-stop-consecutive", type=int, default=2)
    parser.add_argument("--advisor-warmup-steps", type=int, default=6)
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
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    log_dir = output_dir / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    start_time = time.time()

    for pair_idx in range(args.episodes):
        task = TASKS[(pair_idx + args.task_offset) % len(TASKS)]
        seed = args.seed_base + pair_idx
        pair_id = f"pair_{pair_idx + 1:03d}_{task['name']}"
        for condition in ("off", "replan"):
            condition_dir = output_dir / condition
            episode_id = f"{pair_id}_{condition}"
            cmd = build_collect_cmd(
                args,
                condition=condition,
                task=task,
                episode_id=episode_id,
                seed=seed,
                output_dir=condition_dir,
            )
            print(f"\n[ab] pair={pair_idx + 1}/{args.episodes} condition={condition} task={task['name']} seed={seed}", flush=True)
            log_path = log_dir / f"{episode_id}.log"
            returncode = stream_run(cmd, log_path, prefix=f"[{condition}]")
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
            metrics = {
                "pair_id": pair_id,
                "condition": condition,
                "task": task["name"],
                "seed": seed,
                "returncode": returncode,
                "log_path": str(log_path),
                "elapsed_s": time.time() - start_time,
            }
            metrics.update(parse_log(log_text))
            metrics.update(load_episode_metrics(condition_dir, episode_id))
            results.append(metrics)
            (output_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
            (output_dir / "summary.json").write_text(
                json.dumps(summarize(results), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            if returncode != 0:
                raise SystemExit(f"{condition} run failed for {pair_id}: returncode={returncode}")

    summary = summarize(results)
    print("\n[ab] summary")
    print(json.dumps(summary["conditions"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
