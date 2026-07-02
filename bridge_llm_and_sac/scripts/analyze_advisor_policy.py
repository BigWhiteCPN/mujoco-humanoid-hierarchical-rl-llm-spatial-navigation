#!/usr/bin/env python3
"""Replay a recorded episode through BridgeAdvisor and intervention policy."""

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
from bridge.constants import EVENT_TYPES, REPLAN_ACTIONS


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect advisor decisions and policy stops on one episode.")
    parser.add_argument("episode")
    parser.add_argument("--checkpoint", default="runs/runtime_v2_v3_v4_finetune/best.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mode", choices=["off", "safe", "risk", "replan"], default="replan")
    parser.add_argument("--confidence", type=float, default=0.90)
    parser.add_argument("--consecutive", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=6)
    parser.add_argument("--stuck-prob", type=float, default=0.70)
    parser.add_argument("--target-prob", type=float, default=0.70)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    advisor = BridgeAdvisor(args.checkpoint, device=args.device)
    policy = BridgeInterventionPolicy(
        BridgeInterventionConfig(
            mode=args.mode,
            confidence_threshold=args.confidence,
            consecutive_steps=args.consecutive,
            warmup_steps=args.warmup,
            stuck_prob_threshold=args.stuck_prob,
            target_prob_threshold=args.target_prob,
        )
    )
    data = np.load(Path(args.episode), allow_pickle=False)
    rows = []
    stops = []
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
        decision = advisor.predict()
        intervention = policy.update(decision)
        row = {
            "step": step,
            "pred_event": decision.event,
            "pred_event_conf": decision.event_confidence,
            "pred_replan": decision.replan,
            "pred_replan_conf": decision.replan_confidence,
            "stuck_prob": decision.stuck_prob,
            "risk_prob": decision.failure_risk_prob,
            "target_prob": decision.target_found_prob,
            "true_event": EVENT_TYPES[int(data["event"][step])],
            "true_replan": REPLAN_ACTIONS[int(data["replan"][step])],
            "stop": intervention is not None and intervention.should_stop,
        }
        rows.append(row)
        if row["stop"]:
            stops.append(row)

    print(f"episode={args.episode}")
    print(f"steps={len(rows)} mode={args.mode} confidence={args.confidence} consecutive={args.consecutive} warmup={args.warmup}")
    print(f"stops={len(stops)}")
    if stops:
        for row in stops[: args.top]:
            print(
                f"STOP t={row['step']:03d} pred={row['pred_event']}/{row['pred_replan']} "
                f"conf=({row['pred_event_conf']:.2f},{row['pred_replan_conf']:.2f}) "
                f"stuck={row['stuck_prob']:.2f} risk={row['risk_prob']:.2f} "
                f"true={row['true_event']}/{row['true_replan']}"
            )

    interesting = sorted(
        rows,
        key=lambda row: max(row["pred_replan_conf"], row["stuck_prob"], row["risk_prob"]),
        reverse=True,
    )[: args.top]
    print("top_decisions")
    for row in interesting:
        print(
            f"t={row['step']:03d} pred={row['pred_event']:<22} {row['pred_event_conf']:.2f} "
            f"replan={row['pred_replan']:<20} {row['pred_replan_conf']:.2f} "
            f"stuck={row['stuck_prob']:.2f} risk={row['risk_prob']:.2f} target={row['target_prob']:.2f} "
            f"true={row['true_event']}/{row['true_replan']}"
        )


if __name__ == "__main__":
    main()
