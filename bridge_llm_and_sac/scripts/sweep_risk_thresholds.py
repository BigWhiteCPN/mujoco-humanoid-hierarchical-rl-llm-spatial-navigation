#!/usr/bin/env python3
"""Sweep advisor risk thresholds on recorded bridge episodes."""

from __future__ import annotations

try:
    from scripts._bootstrap import ensure_project_root
except ModuleNotFoundError:
    from _bootstrap import ensure_project_root

ensure_project_root()

import argparse
from pathlib import Path

import numpy as np

from bridge.advisor import BridgeAdvisor
from bridge.arbitration import BridgeInterventionConfig, BridgeInterventionPolicy
from bridge.dataset import FAILURE_EVENT_IDS, FAILURE_REPLAN_IDS


def episode_files(path: Path) -> list[Path]:
    root = path / "episodes" if (path / "episodes").exists() else path
    return sorted(root.glob("*.npz"))


def future_failure(event: np.ndarray, replan: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    failure = np.isin(event, list(FAILURE_EVENT_IDS)) | np.isin(replan, list(FAILURE_REPLAN_IDS))
    labels = np.zeros_like(failure, dtype=bool)
    lead = np.full_like(event, fill_value=-1, dtype=np.int64)
    for step in range(event.shape[0]):
        end = min(event.shape[0], step + horizon + 1)
        hits = np.flatnonzero(failure[step:end])
        if hits.size:
            labels[step] = True
            lead[step] = int(hits[0])
    return labels, lead


def replay_decisions(advisor: BridgeAdvisor, episode_path: Path) -> tuple[list, np.ndarray, np.ndarray]:
    advisor.reset()
    data = np.load(episode_path, allow_pickle=False)
    decisions = []
    for step in range(int(data["event"].shape[0])):
        advisor.record_frame(
            maps=data["maps"][step],
            state=data["state"][step],
            memory=data["memory"][step],
            task=data["task"][step],
            skill=int(data["skill"][step]),
            candidates=data["candidates"][step],
            candidate_mask=data["candidate_mask"][step],
        )
        decisions.append(advisor.predict())
    return decisions, data["event"].astype(int), data["replan"].astype(int)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep failure-risk thresholds offline.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--thresholds", default="0.70,0.80,0.85,0.90,0.95")
    parser.add_argument("--confidence", type=float, default=0.65)
    parser.add_argument("--consecutive", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--stuck-prob", type=float, default=0.50)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--max-episodes", type=int, default=0)
    args = parser.parse_args()

    thresholds = [float(item) for item in args.thresholds.split(",") if item.strip()]
    paths = episode_files(Path(args.data_dir))
    if args.max_episodes > 0:
        paths = paths[: args.max_episodes]
    if not paths:
        raise SystemExit(f"No .npz episodes found under {args.data_dir}")

    advisor = BridgeAdvisor(args.checkpoint, device=args.device)
    cached = []
    for path in paths:
        decisions, event, replan = replay_decisions(advisor, path)
        labels, lead = future_failure(event, replan, args.horizon)
        cached.append((path, decisions, labels, lead))

    print(f"episodes={len(cached)} checkpoint={args.checkpoint}")
    print("threshold stops tp fp mean_first_stop mean_lead risk_stop event_stop")
    for threshold in thresholds:
        stops = 0
        tp = 0
        fp = 0
        first_steps = []
        leads = []
        risk_stops = 0
        event_stops = 0
        for _, decisions, labels, lead in cached:
            policy = BridgeInterventionPolicy(
                BridgeInterventionConfig(
                    mode="risk",
                    confidence_threshold=args.confidence,
                    consecutive_steps=args.consecutive,
                    warmup_steps=args.warmup,
                    stuck_prob_threshold=args.stuck_prob,
                    failure_risk_threshold=threshold,
                )
            )
            first = None
            reason = ""
            for step, decision in enumerate(decisions):
                intervention = policy.update(decision)
                if intervention is not None and intervention.should_stop:
                    first = step
                    reason = intervention.reason
                    break
            if first is None:
                continue
            stops += 1
            first_steps.append(first)
            if labels[first]:
                tp += 1
                leads.append(int(lead[first]))
            else:
                fp += 1
            if reason.startswith("failure_risk"):
                risk_stops += 1
            else:
                event_stops += 1
        mean_first = float(np.mean(first_steps)) if first_steps else -1.0
        mean_lead = float(np.mean(leads)) if leads else -1.0
        print(
            f"{threshold:.2f} {stops} {tp} {fp} {mean_first:.1f} {mean_lead:.1f} "
            f"{risk_stops} {event_stops}"
        )


if __name__ == "__main__":
    main()
