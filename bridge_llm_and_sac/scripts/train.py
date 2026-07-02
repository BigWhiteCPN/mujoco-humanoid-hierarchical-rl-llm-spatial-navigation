#!/usr/bin/env python3
"""Train BridgeNet on recorded or synthetic bridge episodes."""

from __future__ import annotations

try:
    from scripts._bootstrap import ensure_project_root
except ModuleNotFoundError:
    from _bootstrap import ensure_project_root

ensure_project_root()

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from bridge.constants import EVENT_TYPES, REPLAN_ACTIONS
from bridge.dataset import BridgeEpisodeDataset, infer_shapes
from bridge.losses import bridge_loss, move_batch_to_device
from bridge.model import BridgeNet, BridgeNetConfig
from bridge.splits import split_dataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mean_metrics(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {}
    keys = items[0].keys()
    return {key: float(np.mean([m[key] for m in items])) for key in keys}


def class_weights_for_subset(subset, key: str, num_classes: int, device: torch.device) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for idx in subset.indices:
        sample = subset.dataset[idx]
        values = sample[key].reshape(-1)
        counts += torch.bincount(values, minlength=num_classes).float()
    present = counts > 0
    weights = torch.zeros_like(counts)
    weights[present] = counts[present].sum() / counts[present].clamp_min(1.0)
    weights[present] = weights[present] / weights[present].mean().clamp_min(1e-6)
    return weights.to(device)


def load_compatible_checkpoint(model: torch.nn.Module, checkpoint_path: str | Path) -> tuple[int, int, list[str]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    source_state = checkpoint["model_state"]
    target_state = model.state_dict()
    compatible = {}
    skipped = []
    for name, tensor in source_state.items():
        if name in target_state and tuple(tensor.shape) == tuple(target_state[name].shape):
            compatible[name] = tensor
        else:
            skipped.append(name)
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    skipped.extend(unexpected)
    return len(compatible), len(missing), skipped


def run_epoch(
    model,
    loader,
    optimizer,
    device,
    train: bool,
    event_weight: torch.Tensor | None = None,
    replan_weight: torch.Tensor | None = None,
) -> dict[str, float]:
    model.train(train)
    metrics_list: list[dict[str, float]] = []
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        with torch.set_grad_enabled(train):
            outputs = model(batch)
            loss, metrics = bridge_loss(outputs, batch, event_weight=event_weight, replan_weight=replan_weight)
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        metrics_list.append(metrics)
    return mean_metrics(metrics_list)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the cross-attention LLM-to-SAC bridge.")
    parser.add_argument("--data-dir", default="data/synthetic")
    parser.add_argument("--output-dir", default="runs/bridge_v1")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--failure-horizon", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--fusion-layers", type=int, default=2)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument(
        "--split-by",
        choices=["episode", "window"],
        default="episode",
        help="Use episode-level splits to avoid leakage from overlapping windows.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--init-checkpoint", default=None, help="Optional checkpoint to initialize model weights.")
    parser.add_argument("--patience", type=int, default=0, help="Stop after this many epochs without val loss improvement.")
    parser.add_argument(
        "--balanced-class-loss",
        action="store_true",
        help="Use inverse-frequency weights for event/replan classification heads.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = BridgeEpisodeDataset(
        args.data_dir,
        sequence_len=args.sequence_len,
        stride=args.stride,
        failure_horizon=args.failure_horizon,
    )
    shapes = infer_shapes(dataset)
    cfg = BridgeNetConfig(
        map_channels=shapes["map_channels"],
        state_dim=shapes["state_dim"],
        memory_dim=shapes["memory_dim"],
        candidate_dim=shapes["candidate_dim"],
        max_candidates=shapes["max_candidates"],
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        fusion_layers=args.fusion_layers,
    )

    train_set, val_set, split_info = split_dataset(dataset, args.val_fraction, args.seed, args.split_by)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )

    device = torch.device(args.device)
    model = BridgeNet(cfg).to(device)
    if args.init_checkpoint:
        loaded, missing, skipped = load_compatible_checkpoint(model, args.init_checkpoint)
        skipped_preview = ",".join(skipped[:8])
        print(
            f"initialized_from={args.init_checkpoint} loaded={loaded} "
            f"missing={missing} skipped={len(skipped)} skipped_preview={skipped_preview}"
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    event_weight = replan_weight = None
    if args.balanced_class_loss:
        if not isinstance(train_set, Subset):
            raise ValueError("--balanced-class-loss currently requires a Subset train split")
        event_weight = class_weights_for_subset(train_set, "event", len(EVENT_TYPES), device)
        replan_weight = class_weights_for_subset(train_set, "replan", len(REPLAN_ACTIONS), device)
        print(f"event_weight={event_weight.detach().cpu().numpy().round(3).tolist()}")
        print(f"replan_weight={replan_weight.detach().cpu().numpy().round(3).tolist()}")

    (output_dir / "config.json").write_text(
        json.dumps({"args": vars(args), "model": asdict(cfg), "dataset_shapes": shapes, "split": split_info}, indent=2),
        encoding="utf-8",
    )

    best_val = float("inf")
    epochs_without_improvement = 0
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            train=True,
            event_weight=event_weight,
            replan_weight=replan_weight,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            optimizer,
            device,
            train=False,
            event_weight=event_weight,
            replan_weight=replan_weight,
        )
        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} "
            f"val_event_acc={val_metrics['event_acc']:.3f} val_replan_acc={val_metrics['replan_acc']:.3f} "
            f"val_risk_acc={val_metrics['failure_risk_acc']:.3f}"
        )
        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": asdict(cfg),
            "val_metrics": val_metrics,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            epochs_without_improvement = 0
            torch.save(checkpoint, output_dir / "best.pt")
        else:
            epochs_without_improvement += 1
            if args.patience > 0 and epochs_without_improvement >= args.patience:
                print(f"early_stop epoch={epoch} best_val={best_val:.4f}")
                break


if __name__ == "__main__":
    main()
