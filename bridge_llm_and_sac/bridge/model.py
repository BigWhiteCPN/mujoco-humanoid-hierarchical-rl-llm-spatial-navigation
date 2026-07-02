"""Cross-attention bridge network for LLM-guided SAC navigation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .constants import (
    CONSTRAINT_TYPES,
    EVENT_TYPES,
    INTENT_TYPES,
    REPLAN_ACTIONS,
    SKILL_TYPES,
    TARGET_TYPES,
)


@dataclass
class BridgeNetConfig:
    map_channels: int = 5
    state_dim: int = 22
    memory_dim: int = 12
    candidate_dim: int = 8
    hidden_dim: int = 128
    map_token_grid: int = 8
    num_heads: int = 4
    fusion_layers: int = 2
    temporal_layers: int = 1
    dropout: float = 0.1
    max_candidates: int = 8
    num_intents: int = len(INTENT_TYPES)
    num_targets: int = len(TARGET_TYPES)
    num_constraints: int = len(CONSTRAINT_TYPES)
    num_skills: int = len(SKILL_TYPES)
    num_events: int = len(EVENT_TYPES)
    num_replan_actions: int = len(REPLAN_ACTIONS)


class MapEncoder(nn.Module):
    """Encode robot-centric map layers into spatial tokens."""

    def __init__(self, in_channels: int, hidden_dim: int, token_grid: int):
        super().__init__()
        self.token_grid = token_grid
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(4, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((token_grid, token_grid)),
        )
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, maps: torch.Tensor) -> torch.Tensor:
        features = self.net(maps)
        tokens = features.flatten(2).transpose(1, 2)
        return self.proj(tokens)


class MLPEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TaskEncoder(nn.Module):
    """Encode structured task ids into one task token."""

    def __init__(self, cfg: BridgeNetConfig):
        super().__init__()
        self.intent = nn.Embedding(cfg.num_intents, cfg.hidden_dim)
        self.target = nn.Embedding(cfg.num_targets, cfg.hidden_dim)
        self.constraint = nn.Embedding(cfg.num_constraints, cfg.hidden_dim)
        self.norm = nn.LayerNorm(cfg.hidden_dim)

    def forward(self, task: torch.Tensor) -> torch.Tensor:
        if task.shape[-1] != 3:
            raise ValueError(f"task must have shape [..., 3], got {tuple(task.shape)}")
        token = self.intent(task[..., 0]) + self.target(task[..., 1]) + self.constraint(task[..., 2])
        return self.norm(token)


class CrossModalFusion(nn.Module):
    """Fuse task/state queries with map, memory, and candidate context."""

    def __init__(self, hidden_dim: int, num_heads: int, fusion_layers: int, dropout: float):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.self_fusion = nn.TransformerEncoder(layer, num_layers=fusion_layers)
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.context_norm = nn.LayerNorm(hidden_dim)

    def forward(self, query_tokens: torch.Tensor, context_tokens: torch.Tensor) -> torch.Tensor:
        query_tokens = self.query_norm(query_tokens)
        context_tokens = self.context_norm(context_tokens)
        attended, _ = self.cross_attn(query_tokens, context_tokens, context_tokens, need_weights=False)
        fused = query_tokens + attended
        return self.self_fusion(fused)


class BridgeNet(nn.Module):
    """Task-conditioned cross-attention model for event-driven replanning."""

    def __init__(self, cfg: BridgeNetConfig):
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden_dim

        self.map_encoder = MapEncoder(cfg.map_channels, h, cfg.map_token_grid)
        self.state_encoder = MLPEncoder(cfg.state_dim, h, cfg.dropout)
        self.memory_encoder = MLPEncoder(cfg.memory_dim, h, cfg.dropout)
        self.candidate_encoder = MLPEncoder(cfg.candidate_dim, h, cfg.dropout)
        self.task_encoder = TaskEncoder(cfg)
        self.skill_embedding = nn.Embedding(cfg.num_skills, h)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, h))
        self.type_embedding = nn.Embedding(6, h)
        self.fusion = CrossModalFusion(h, cfg.num_heads, cfg.fusion_layers, cfg.dropout)
        self.temporal = nn.GRU(
            input_size=h,
            hidden_size=h,
            num_layers=cfg.temporal_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.temporal_layers > 1 else 0.0,
        )
        self.temporal_norm = nn.LayerNorm(h)

        self.event_head = nn.Linear(h, cfg.num_events)
        self.replan_head = nn.Linear(h, cfg.num_replan_actions)
        self.success_head = nn.Linear(h, 1)
        self.stuck_head = nn.Linear(h, 1)
        self.failure_risk_head = nn.Linear(h, 1)
        self.target_found_head = nn.Linear(h, 1)
        self.cost_head = nn.Linear(h, 1)
        self.info_gain_head = nn.Linear(h, 1)
        self.candidate_score_head = nn.Sequential(
            nn.Linear(h * 2, h),
            nn.SiLU(),
            nn.Linear(h, 1),
        )

        nn.init.normal_(self.cls_token, std=0.02)

    def _add_type(self, tokens: torch.Tensor, type_id: int) -> torch.Tensor:
        type_ids = torch.full((tokens.shape[0], tokens.shape[1]), type_id, device=tokens.device, dtype=torch.long)
        return tokens + self.type_embedding(type_ids)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        maps = batch["maps"]
        state = batch["state"]
        memory = batch["memory"]
        task = batch["task"]
        skill = batch["skill"]
        candidates = batch["candidates"]

        bsz, seq_len = maps.shape[:2]
        bt = bsz * seq_len

        maps_flat = maps.reshape(bt, maps.shape[2], maps.shape[3], maps.shape[4])
        state_flat = state.reshape(bt, state.shape[-1])
        memory_flat = memory.reshape(bt, memory.shape[-1])
        task_flat = task.reshape(bt, task.shape[-1])
        skill_flat = skill.reshape(bt)
        cand_flat = candidates.reshape(bt, candidates.shape[-2], candidates.shape[-1])

        map_tokens = self._add_type(self.map_encoder(maps_flat), 1)
        state_token = self._add_type(self.state_encoder(state_flat).unsqueeze(1), 2)
        task_token = self._add_type(self.task_encoder(task_flat).unsqueeze(1), 3)
        skill_token = self._add_type(self.skill_embedding(skill_flat).unsqueeze(1), 4)
        memory_token = self._add_type(self.memory_encoder(memory_flat).unsqueeze(1), 5)
        candidate_tokens = self._add_type(self.candidate_encoder(cand_flat), 0)

        cls = self.cls_token.expand(bt, -1, -1)
        query_tokens = torch.cat([cls, task_token, state_token, skill_token], dim=1)
        context_tokens = torch.cat([map_tokens, memory_token, candidate_tokens], dim=1)

        fused = self.fusion(query_tokens, context_tokens)
        frame_embedding = fused[:, 0].reshape(bsz, seq_len, self.cfg.hidden_dim)
        temporal, _ = self.temporal(frame_embedding)
        temporal = self.temporal_norm(temporal)

        temporal_flat = temporal.reshape(bt, self.cfg.hidden_dim)
        cand_context = temporal_flat.unsqueeze(1).expand(-1, candidate_tokens.shape[1], -1)
        candidate_scores = self.candidate_score_head(torch.cat([cand_context, candidate_tokens], dim=-1))
        candidate_scores = candidate_scores.squeeze(-1).reshape(bsz, seq_len, candidates.shape[-2])

        return {
            "event_logits": self.event_head(temporal),
            "replan_logits": self.replan_head(temporal),
            "success_logit": self.success_head(temporal).squeeze(-1),
            "stuck_logit": self.stuck_head(temporal).squeeze(-1),
            "failure_risk_logit": self.failure_risk_head(temporal).squeeze(-1),
            "target_found_logit": self.target_found_head(temporal).squeeze(-1),
            "cost": self.cost_head(temporal).squeeze(-1),
            "info_gain": self.info_gain_head(temporal).squeeze(-1),
            "candidate_scores": candidate_scores,
        }
