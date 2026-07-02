#!/usr/bin/env python3
"""Evaluate a BridgeNet checkpoint with per-class metrics."""

from __future__ import annotations

try:
    from scripts._bootstrap import ensure_project_root
except ModuleNotFoundError:
    from _bootstrap import ensure_project_root

ensure_project_root()

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from bridge.constants import EVENT_TYPES, REPLAN_ACTIONS
from bridge.dataset import BridgeEpisodeDataset, infer_shapes
from bridge.losses import bridge_loss, move_batch_to_device
from bridge.model import BridgeNet, BridgeNetConfig
from bridge.splits import split_dataset


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def config_from_checkpoint(checkpoint: dict) -> BridgeNetConfig:
    cfg = checkpoint.get("config")
    if not cfg:
        raise ValueError("Checkpoint does not contain model config")
    return BridgeNetConfig(**cfg)


def confusion_matrix(pred: np.ndarray, target: np.ndarray, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for truth, guess in zip(target.reshape(-1), pred.reshape(-1)):
        matrix[int(truth), int(guess)] += 1
    return matrix


def class_report(matrix: np.ndarray, names: list[str]) -> list[dict[str, float | str | int]]:
    rows = []
    for idx, name in enumerate(names):
        tp = float(matrix[idx, idx])
        fp = float(matrix[:, idx].sum() - matrix[idx, idx])
        fn = float(matrix[idx, :].sum() - matrix[idx, idx])
        support = int(matrix[idx, :].sum())
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
        rows.append(
            {
                "class": name,
                "support": support,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return rows


def print_report(title: str, matrix: np.ndarray, names: list[str]) -> None:
    rows = class_report(matrix, names)
    print(f"\n{title}")
    print("class                       support  precision  recall  f1")
    for row in rows:
        if row["support"] == 0:
            continue
        print(
            f"{row['class']:<27} {row['support']:>7d} "
            f"{row['precision']:>9.3f} {row['recall']:>7.3f} {row['f1']:>5.3f}"
        )
    print("confusion_matrix rows=true cols=pred")
    print(matrix.tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a BridgeNet checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split", choices=["val", "train", "all"], default="val")
    parser.add_argument("--split-by", choices=["episode", "window"], default=None)
    parser.add_argument("--val-fraction", type=float, default=None)
    parser.add_argument("--sequence-len", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--failure-horizon", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    run_config = load_json(checkpoint_path.with_name("config.json"))
    train_args = run_config.get("args", {})

    sequence_len = args.sequence_len or int(train_args.get("sequence_len", 16))
    stride = args.stride or int(train_args.get("stride", 4))
    failure_horizon = args.failure_horizon if args.failure_horizon is not None else int(train_args.get("failure_horizon", 8))
    split_by = args.split_by or str(train_args.get("split_by", "episode"))
    val_fraction = args.val_fraction if args.val_fraction is not None else float(train_args.get("val_fraction", 0.15))
    seed = args.seed if args.seed is not None else int(train_args.get("seed", 11))

    dataset = BridgeEpisodeDataset(
        args.data_dir,
        sequence_len=sequence_len,
        stride=stride,
        failure_horizon=failure_horizon,
    )
    if args.split == "all":
        eval_set = dataset
        split_info = {"split": "all", "windows": len(dataset)}
    else:
        train_set, val_set, split_info = split_dataset(dataset, val_fraction, seed, split_by)
        eval_set = val_set if args.split == "val" else train_set
        split_info = dict(split_info)
        split_info["split"] = args.split

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = config_from_checkpoint(ckpt)
    shapes = infer_shapes(dataset)
    if cfg.map_channels != shapes["map_channels"] or cfg.state_dim != shapes["state_dim"]:
        raise ValueError(f"Checkpoint/data shape mismatch: checkpoint={cfg}, data_shapes={shapes}")

    device = torch.device(args.device)
    model = BridgeNet(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    loader = DataLoader(eval_set, batch_size=args.batch_size, shuffle=False)
    event_pred = []
    event_true = []
    replan_pred = []
    replan_true = []
    losses = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            outputs = model(batch)
            loss, metrics = bridge_loss(outputs, batch)
            losses.append(metrics)
            event_pred.append(outputs["event_logits"].argmax(dim=-1).detach().cpu().numpy())
            event_true.append(batch["event"].detach().cpu().numpy())
            replan_pred.append(outputs["replan_logits"].argmax(dim=-1).detach().cpu().numpy())
            replan_true.append(batch["replan"].detach().cpu().numpy())

    event_pred_arr = np.concatenate(event_pred, axis=0)
    event_true_arr = np.concatenate(event_true, axis=0)
    replan_pred_arr = np.concatenate(replan_pred, axis=0)
    replan_true_arr = np.concatenate(replan_true, axis=0)

    event_matrix = confusion_matrix(event_pred_arr, event_true_arr, len(EVENT_TYPES))
    replan_matrix = confusion_matrix(replan_pred_arr, replan_true_arr, len(REPLAN_ACTIONS))
    mean_loss = float(np.mean([item["loss"] for item in losses]))
    mean_risk_acc = float(np.mean([item.get("failure_risk_acc", 0.0) for item in losses]))

    print(f"checkpoint={checkpoint_path}")
    print(f"data_dir={args.data_dir}")
    print(f"split_info={split_info}")
    print(f"loss={mean_loss:.4f}")
    print(f"failure_risk_acc={mean_risk_acc:.3f}")
    print_report("event report", event_matrix, EVENT_TYPES)
    print_report("replan report", replan_matrix, REPLAN_ACTIONS)

    if args.output_json:
        output = {
            "checkpoint": str(checkpoint_path),
            "data_dir": args.data_dir,
            "split_info": split_info,
            "loss": mean_loss,
            "failure_risk_acc": mean_risk_acc,
            "event_confusion": event_matrix.tolist(),
            "replan_confusion": replan_matrix.tolist(),
            "event_report": class_report(event_matrix, EVENT_TYPES),
            "replan_report": class_report(replan_matrix, REPLAN_ACTIONS),
        }
        Path(args.output_json).write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
