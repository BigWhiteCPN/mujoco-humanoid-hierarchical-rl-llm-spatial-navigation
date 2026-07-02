"""Training components for the LLM-to-SAC cognitive bridge."""

from .constants import EVENT_TYPES, REPLAN_ACTIONS, SKILL_TYPES
from .arbitration import BridgeIntervention, BridgeInterventionConfig, BridgeInterventionPolicy
from .recorder import BridgeEpisodeRecorder, RecorderConfig, StepLabels, StepSignals, TaskSpec
from .sidecar import BridgeSidecarCollector
from .snapshot import BridgeSnapshotBuilder
from .hooks import install_navigation_sidecar_hook, restore_navigation_hook


def __getattr__(name: str):
    if name in {"BridgeAdvisor", "BridgeDecision"}:
        from .advisor import BridgeAdvisor, BridgeDecision

        return {"BridgeAdvisor": BridgeAdvisor, "BridgeDecision": BridgeDecision}[name]
    if name in {"BridgeNet", "BridgeNetConfig"}:
        from .model import BridgeNet, BridgeNetConfig

        return {"BridgeNet": BridgeNet, "BridgeNetConfig": BridgeNetConfig}[name]
    raise AttributeError(name)

__all__ = [
    "BridgeNet",
    "BridgeNetConfig",
    "BridgeAdvisor",
    "BridgeDecision",
    "BridgeIntervention",
    "BridgeInterventionConfig",
    "BridgeInterventionPolicy",
    "BridgeEpisodeRecorder",
    "BridgeSidecarCollector",
    "BridgeSnapshotBuilder",
    "install_navigation_sidecar_hook",
    "restore_navigation_hook",
    "RecorderConfig",
    "StepLabels",
    "StepSignals",
    "TaskSpec",
    "EVENT_TYPES",
    "REPLAN_ACTIONS",
    "SKILL_TYPES",
]
