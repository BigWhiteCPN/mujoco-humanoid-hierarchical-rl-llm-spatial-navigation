from __future__ import annotations

import torch

from bridge.constants import EVENT_TYPES, REPLAN_ACTIONS
from bridge.losses import bridge_loss
from bridge.model import BridgeNet, BridgeNetConfig


def make_batch(batch_size=2, sequence_len=4, max_candidates=3):
    return {
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


def test_bridge_model_forward_and_loss():
    cfg = BridgeNetConfig(hidden_dim=64, num_heads=4, fusion_layers=1, max_candidates=3)
    model = BridgeNet(cfg)
    batch = make_batch(max_candidates=3)
    outputs = model(batch)

    assert outputs["event_logits"].shape == (2, 4, len(EVENT_TYPES))
    assert outputs["replan_logits"].shape == (2, 4, len(REPLAN_ACTIONS))
    assert outputs["failure_risk_logit"].shape == (2, 4)
    assert outputs["candidate_scores"].shape == (2, 4, 3)

    loss, metrics = bridge_loss(outputs, batch)
    assert torch.isfinite(loss)
    assert metrics["loss"] > 0
