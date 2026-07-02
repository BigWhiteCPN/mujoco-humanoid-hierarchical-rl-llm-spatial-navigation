"""Runtime inference helper for BridgeNet.

The advisor is deliberately passive: it turns recent bridge snapshots into
event/replan suggestions, but it does not command the robot by itself.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .constants import EVENT_TYPES, REPLAN_ACTIONS, SKILL_TO_ID
from .model import BridgeNet, BridgeNetConfig
from .recorder import RecorderConfig, TaskSpec
from .snapshot import BridgeSnapshot, BridgeSnapshotBuilder


@dataclass(frozen=True)
class BridgeDecision:
    event: str
    replan: str
    event_confidence: float
    replan_confidence: float
    success_prob: float
    stuck_prob: float
    target_found_prob: float
    cost: float
    info_gain: float
    candidate_index: int | None
    candidate_score: float | None
    candidate_xy_norm: tuple[float, float] | None
    event_top3: list[tuple[str, float]]
    replan_top3: list[tuple[str, float]]
    failure_risk_prob: float = 0.0


class BridgeAdvisor:
    """Keep a rolling sequence window and predict bridge-level decisions."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device | None = None,
        sequence_len: int = 16,
        recorder_config: RecorderConfig | None = None,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        model_config = checkpoint.get("config")
        if not model_config:
            raise ValueError(f"{self.checkpoint_path} does not contain a model config")
        self.model_config = BridgeNetConfig(**model_config)
        self.model = BridgeNet(self.model_config).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"], strict=False)
        self.model.eval()

        self.sequence_len = int(sequence_len)
        if self.sequence_len <= 0:
            raise ValueError("sequence_len must be positive")
        self.recorder_config = recorder_config or RecorderConfig(
            map_channels=self.model_config.map_channels,
            state_dim=self.model_config.state_dim,
            memory_dim=self.model_config.memory_dim,
            candidate_dim=self.model_config.candidate_dim,
            max_candidates=self.model_config.max_candidates,
        )
        self.frames: deque[dict[str, np.ndarray | int]] = deque(maxlen=self.sequence_len)
        self.builder: BridgeSnapshotBuilder | None = None
        self.segment_active = False
        self._last_pos: np.ndarray | None = None
        self._last_distance: float | None = None
        self._last_visited_ratio: float | None = None

    def reset(self) -> None:
        self.frames.clear()
        self._last_pos = None
        self._last_distance = None
        self._last_visited_ratio = None
        if self.builder is not None:
            self.builder.finish_segment()
        self.segment_active = False

    def start_segment(self) -> None:
        self.segment_active = True
        self.frames.clear()
        self._last_pos = None
        self._last_distance = None
        self._last_visited_ratio = None
        if self.builder is not None:
            self.builder.start_segment()

    def finish_segment(self) -> None:
        self.segment_active = False
        if self.builder is not None:
            self.builder.finish_segment()

    def _ensure_builder(self, task_spec: TaskSpec) -> BridgeSnapshotBuilder:
        if self.builder is None or self.builder.task_spec != task_spec:
            self.builder = BridgeSnapshotBuilder(self.recorder_config, task_spec)
            if self.segment_active:
                self.builder.start_segment()
        return self.builder

    def observe_runtime(
        self,
        env: Any,
        *,
        memory: Any = None,
        topo_map: Any = None,
        task_spec: TaskSpec | None = None,
        skill: str | int = "explore_frontier",
    ) -> BridgeDecision:
        task = task_spec or TaskSpec()
        builder = self._ensure_builder(task)
        snapshot = builder.build(
            env=env,
            memory=memory,
            topo_map=topo_map,
            prev_pos=self._last_pos,
            prev_distance=self._last_distance,
            prev_visited_ratio=self._last_visited_ratio,
        )
        self._last_pos = snapshot.robot_pos.copy()
        self._last_distance = snapshot.distance_to_subgoal
        self._last_visited_ratio = snapshot.visited_ratio
        return self.observe_snapshot(snapshot, task_spec=task, skill=skill)

    def observe_snapshot(
        self,
        snapshot: BridgeSnapshot,
        *,
        task_spec: TaskSpec | None = None,
        skill: str | int = "explore_frontier",
    ) -> BridgeDecision:
        task = task_spec or TaskSpec()
        self.record_frame(
            maps=snapshot.maps,
            state=snapshot.state,
            memory=snapshot.memory,
            task=task.to_ids(),
            skill=skill,
            candidates=snapshot.candidates,
        )
        return self.predict()

    def record_frame(
        self,
        *,
        maps: np.ndarray,
        state: np.ndarray,
        memory: np.ndarray,
        task: np.ndarray,
        skill: str | int,
        candidates: np.ndarray,
        candidate_mask: np.ndarray | None = None,
    ) -> None:
        if isinstance(skill, str):
            skill_id = SKILL_TO_ID.get(skill, SKILL_TO_ID["idle"])
        else:
            skill_id = int(skill)
        candidate_arr = np.asarray(candidates, dtype=np.float32)
        state_arr = self._fit_last_dim(np.asarray(state, dtype=np.float32), self.model_config.state_dim)
        memory_arr = self._fit_last_dim(np.asarray(memory, dtype=np.float32), self.model_config.memory_dim)
        candidate_arr = self._fit_last_dim(candidate_arr, self.model_config.candidate_dim)
        if candidate_mask is None:
            candidate_mask = np.any(np.abs(candidate_arr) > 1e-6, axis=-1)
        self.frames.append(
            {
                "maps": np.asarray(maps, dtype=np.float32),
                "state": state_arr,
                "memory": memory_arr,
                "task": np.asarray(task, dtype=np.int64),
                "skill": np.asarray(skill_id, dtype=np.int64),
                "candidates": candidate_arr,
                "candidate_mask": np.asarray(candidate_mask, dtype=bool),
            }
        )

    @staticmethod
    def _fit_last_dim(array: np.ndarray, target_dim: int) -> np.ndarray:
        current_dim = int(array.shape[-1])
        if current_dim == target_dim:
            return array
        if current_dim > target_dim:
            return array[..., :target_dim]
        padded = np.zeros((*array.shape[:-1], target_dim), dtype=array.dtype)
        padded[..., :current_dim] = array
        return padded

    def _padded_frames(self) -> list[dict[str, np.ndarray | int]]:
        if not self.frames:
            raise ValueError("No frames recorded; call observe_snapshot() or record_frame() first")
        frames = list(self.frames)
        while len(frames) < self.sequence_len:
            frames.insert(0, frames[0])
        return frames[-self.sequence_len :]

    def _batch(self) -> dict[str, torch.Tensor]:
        frames = self._padded_frames()
        arrays = {
            key: np.stack([frame[key] for frame in frames], axis=0)
            for key in ("maps", "state", "memory", "task", "skill", "candidates", "candidate_mask")
        }
        return {
            "maps": torch.from_numpy(arrays["maps"]).unsqueeze(0).float().to(self.device),
            "state": torch.from_numpy(arrays["state"]).unsqueeze(0).float().to(self.device),
            "memory": torch.from_numpy(arrays["memory"]).unsqueeze(0).float().to(self.device),
            "task": torch.from_numpy(arrays["task"]).unsqueeze(0).long().to(self.device),
            "skill": torch.from_numpy(arrays["skill"]).unsqueeze(0).long().to(self.device),
            "candidates": torch.from_numpy(arrays["candidates"]).unsqueeze(0).float().to(self.device),
            "candidate_mask": torch.from_numpy(arrays["candidate_mask"]).unsqueeze(0).bool().to(self.device),
        }

    def predict(self) -> BridgeDecision:
        batch = self._batch()
        with torch.no_grad():
            outputs = self.model(batch)

        event_probs = torch.softmax(outputs["event_logits"][0, -1], dim=-1).detach().cpu().numpy()
        replan_probs = torch.softmax(outputs["replan_logits"][0, -1], dim=-1).detach().cpu().numpy()
        event_id = int(event_probs.argmax())
        replan_id = int(replan_probs.argmax())

        candidate_scores = outputs["candidate_scores"][0, -1].detach().cpu().numpy()
        candidate_mask = batch["candidate_mask"][0, -1].detach().cpu().numpy()
        candidate_index = None
        candidate_score = None
        candidate_xy_norm = None
        if bool(candidate_mask.any()):
            masked_scores = np.where(candidate_mask, candidate_scores, -np.inf)
            candidate_index = int(masked_scores.argmax())
            candidate_score = float(masked_scores[candidate_index])
            candidate = batch["candidates"][0, -1, candidate_index].detach().cpu().numpy()
            candidate_xy_norm = (float(candidate[4]), float(candidate[5]))

        return BridgeDecision(
            event=EVENT_TYPES[event_id],
            replan=REPLAN_ACTIONS[replan_id],
            event_confidence=float(event_probs[event_id]),
            replan_confidence=float(replan_probs[replan_id]),
            success_prob=float(torch.sigmoid(outputs["success_logit"][0, -1]).detach().cpu()),
            stuck_prob=float(torch.sigmoid(outputs["stuck_logit"][0, -1]).detach().cpu()),
            target_found_prob=float(torch.sigmoid(outputs["target_found_logit"][0, -1]).detach().cpu()),
            cost=float(outputs["cost"][0, -1].detach().cpu()),
            info_gain=float(outputs["info_gain"][0, -1].detach().cpu()),
            candidate_index=candidate_index,
            candidate_score=candidate_score,
            candidate_xy_norm=candidate_xy_norm,
            event_top3=self._topk(event_probs, EVENT_TYPES),
            replan_top3=self._topk(replan_probs, REPLAN_ACTIONS),
            failure_risk_prob=float(torch.sigmoid(outputs["failure_risk_logit"][0, -1]).detach().cpu()),
        )

    @staticmethod
    def _topk(values: np.ndarray, names: list[str], k: int = 3) -> list[tuple[str, float]]:
        order = np.argsort(values)[::-1][:k]
        return [(names[int(idx)], float(values[int(idx)])) for idx in order]
