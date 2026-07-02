"""Training losses and metrics for BridgeNet."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def bridge_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    event_weight: torch.Tensor | None = None,
    replan_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    event_loss = F.cross_entropy(outputs["event_logits"].flatten(0, 1), batch["event"].flatten(), weight=event_weight)
    replan_loss = F.cross_entropy(
        outputs["replan_logits"].flatten(0, 1),
        batch["replan"].flatten(),
        weight=replan_weight,
    )

    success_loss = F.binary_cross_entropy_with_logits(outputs["success_logit"], batch["success"])
    stuck_loss = F.binary_cross_entropy_with_logits(outputs["stuck_logit"], batch["stuck"])
    failure_risk_target = batch.get("failure_risk")
    if failure_risk_target is None:
        failure_risk_target = torch.zeros_like(batch["stuck"])
    failure_risk_loss = F.binary_cross_entropy_with_logits(outputs["failure_risk_logit"], failure_risk_target)
    target_loss = F.binary_cross_entropy_with_logits(outputs["target_found_logit"], batch["target_found"])
    cost_loss = F.smooth_l1_loss(outputs["cost"], batch["cost"])
    info_loss = F.smooth_l1_loss(outputs["info_gain"], batch["info_gain"])

    candidate_mask = batch["candidate_mask"].float()
    candidate_error = (outputs["candidate_scores"] - batch["candidate_score_target"]) ** 2
    candidate_loss = (candidate_error * candidate_mask).sum() / candidate_mask.sum().clamp_min(1.0)

    loss = (
        event_loss
        + replan_loss
        + 0.5 * success_loss
        + 0.7 * stuck_loss
        + 0.7 * failure_risk_loss
        + 0.5 * target_loss
        + 0.5 * cost_loss
        + 0.7 * info_loss
        + 0.5 * candidate_loss
    )

    with torch.no_grad():
        event_acc = (
            outputs["event_logits"].argmax(dim=-1).eq(batch["event"]).float().mean().item()
        )
        replan_acc = (
            outputs["replan_logits"].argmax(dim=-1).eq(batch["replan"]).float().mean().item()
        )
        risk_prob = torch.sigmoid(outputs["failure_risk_logit"])
        risk_acc = risk_prob.ge(0.5).eq(failure_risk_target.ge(0.5)).float().mean().item()
    metrics = {
        "loss": float(loss.detach().cpu()),
        "event_loss": float(event_loss.detach().cpu()),
        "replan_loss": float(replan_loss.detach().cpu()),
        "success_loss": float(success_loss.detach().cpu()),
        "stuck_loss": float(stuck_loss.detach().cpu()),
        "failure_risk_loss": float(failure_risk_loss.detach().cpu()),
        "target_loss": float(target_loss.detach().cpu()),
        "cost_loss": float(cost_loss.detach().cpu()),
        "info_loss": float(info_loss.detach().cpu()),
        "candidate_loss": float(candidate_loss.detach().cpu()),
        "event_acc": event_acc,
        "replan_acc": replan_acc,
        "failure_risk_acc": risk_acc,
    }
    return loss, metrics


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}
