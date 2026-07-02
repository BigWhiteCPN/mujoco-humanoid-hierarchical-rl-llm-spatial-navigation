#!/usr/bin/env python3
"""Smoke-test online BridgeAdvisor inference on a recorded episode."""

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
from bridge.constants import EVENT_TYPES, REPLAN_ACTIONS


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BridgeAdvisor frame-by-frame on one .npz episode.")
    parser.add_argument("--checkpoint", default="runs/runtime_v2_v3_v4_finetune/best.pt")
    parser.add_argument("--episode", default=None)
    parser.add_argument("--data-dir", default="data/runtime_v2_v3_v4")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--print-every", type=int, default=5)
    args = parser.parse_args()

    if args.episode:
        episode_path = Path(args.episode)
    else:
        episodes = sorted((Path(args.data_dir) / "episodes").glob("*.npz"))
        if not episodes:
            raise FileNotFoundError(f"No .npz episodes found under {args.data_dir}/episodes")
        episode_path = episodes[0]

    advisor = BridgeAdvisor(args.checkpoint, device=args.device)
    data = np.load(episode_path, allow_pickle=False)
    total = int(data["event"].shape[0])
    limit = min(total, int(args.max_steps))
    correct_event = 0
    correct_replan = 0
    print(f"checkpoint={args.checkpoint}")
    print(f"episode={episode_path} total_steps={total} shown_steps={limit}")

    for step in range(limit):
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
        true_event = EVENT_TYPES[int(data["event"][step])]
        true_replan = REPLAN_ACTIONS[int(data["replan"][step])]
        correct_event += int(decision.event == true_event)
        correct_replan += int(decision.replan == true_replan)
        if step % args.print_every == 0 or step == limit - 1:
            print(
                f"t={step:03d} "
                f"event={decision.event:<23} true={true_event:<23} "
                f"replan={decision.replan:<22} true={true_replan:<22} "
                f"stuck={decision.stuck_prob:.2f} risk={decision.failure_risk_prob:.2f} "
                f"target={decision.target_found_prob:.2f} "
                f"cand={decision.candidate_index}"
            )

    print(f"online_prefix_event_acc={correct_event / max(limit, 1):.3f}")
    print(f"online_prefix_replan_acc={correct_replan / max(limit, 1):.3f}")


if __name__ == "__main__":
    main()
