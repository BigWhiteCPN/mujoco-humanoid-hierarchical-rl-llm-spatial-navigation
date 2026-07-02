"""Shared categorical vocabularies for bridge training."""

INTENT_TYPES = [
    "explore",
    "search_place",
    "navigate_to_place",
    "inspect_place",
    "return_to_start",
    "recall_memory",
]

TARGET_TYPES = [
    "none",
    "meeting_room",
    "pantry",
    "door",
    "office",
    "unknown_landmark",
]

CONSTRAINT_TYPES = [
    "none",
    "prefer_safe",
    "prefer_fast",
    "avoid_failed_paths",
    "stop_when_found",
]

SKILL_TYPES = [
    "idle",
    "go_to",
    "explore_frontier",
    "scan",
    "recover",
    "return_to_node",
]

EVENT_TYPES = [
    "continue",
    "subgoal_completed",
    "navigation_stuck",
    "target_candidate_found",
    "path_invalidated",
    "low_information_gain",
    "need_scan",
    "need_replan",
]

REPLAN_ACTIONS = [
    "continue_current",
    "interrupt_and_scan",
    "switch_subgoal",
    "go_to_target_candidate",
    "ask_llm_replan",
    "declare_unreachable",
]

NUM_TASK_FIELDS = 3


def make_id_map(items):
    return {name: idx for idx, name in enumerate(items)}


INTENT_TO_ID = make_id_map(INTENT_TYPES)
TARGET_TO_ID = make_id_map(TARGET_TYPES)
CONSTRAINT_TO_ID = make_id_map(CONSTRAINT_TYPES)
SKILL_TO_ID = make_id_map(SKILL_TYPES)
EVENT_TO_ID = make_id_map(EVENT_TYPES)
REPLAN_TO_ID = make_id_map(REPLAN_ACTIONS)
