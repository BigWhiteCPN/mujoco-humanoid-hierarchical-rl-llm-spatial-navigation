#!/usr/bin/env python3
"""Run one forward/loss pass without requiring pytest."""

from __future__ import annotations

try:
    from scripts._bootstrap import ensure_project_root
except ModuleNotFoundError:
    from _bootstrap import ensure_project_root

ensure_project_root()

import torch

from bridge.constants import EVENT_TYPES, REPLAN_ACTIONS
from bridge.losses import bridge_loss
from bridge.model import BridgeNet, BridgeNetConfig


def main() -> None:
    batch_size = 2
    sequence_len = 4
    max_candidates = 3
    batch = {
        "maps": torch.rand(batch_size, sequence_len, 5, 64, 64),
        "state": torch.rand(batch_size, sequence_len, 22),
        "memory": torch.rand(batch_size, sequence_len, 12),
        "task": torch.randint(0, 2, (batch_size, sequence_len, 3)),
        "skill": torch.randint(0, 3, (batch_size, sequence_len)),
        "candidates": torch.rand(batch_size, sequence_len, max_candidates, 8),
        "candidate_mask": torch.ones(batch_size, sequence_len, max_candidates, dtype=torch.bool),
        "event": torch.randint(0, len(EVENT_TYPES), (batch_size, sequence_len)),
        "replan": torch.randint(0, len(REPLAN_ACTIONS), (batch_size, sequence_len)),
        "success": torch.rand(batch_size, sequence_len),
        "stuck": torch.rand(batch_size, sequence_len),
        "target_found": torch.rand(batch_size, sequence_len),
        "failure_risk": torch.rand(batch_size, sequence_len),
        "cost": torch.rand(batch_size, sequence_len),
        "info_gain": torch.rand(batch_size, sequence_len),
        "candidate_score_target": torch.rand(batch_size, sequence_len, max_candidates),
    }
    model = BridgeNet(BridgeNetConfig(hidden_dim=64, fusion_layers=1, max_candidates=max_candidates))
    outputs = model(batch)
    loss, metrics = bridge_loss(outputs, batch)
    print(f"forward/loss smoke ok: loss={float(loss.detach()):.4f}, event_acc={metrics['event_acc']:.3f}")


if __name__ == "__main__":
    main()
