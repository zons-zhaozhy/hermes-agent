"""Tool implementations for the decision-tree world model plugin.

Four tools:
- world_model_init: Create a new decision tree for a task
- world_model_view: View current tree state (compact or full)
- world_model_update: Update tree with new action or result
- world_model_check: Run FOLLOW_THROUGH, PERF_GAP, and stagnation checks

All tools operate on the current session's decision tree, persisted at
~/.hermes/decision_tree/<profile>/<session_id>.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from agent.decision_tree import (
    WorldModel,
    _optional_float,
)

logger = logging.getLogger(__name__)

# Tool names — keep in sync with plugin.yaml
TOOL_INIT = "world_model_init"
TOOL_VIEW = "world_model_view"
TOOL_UPDATE = "world_model_update"
TOOL_CHECK = "world_model_check"


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

INIT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "task_summary": {
            "type": "string",
            "description": "One paragraph summary of the task and key constraints (max 300 chars).",
        },
        "open_questions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "3-8 unknowns that most affect success.",
        },
    },
    "required": ["task_summary"],
}

VIEW_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "mode": {
            "type": "string",
            "enum": ["compact", "full", "summary", "open_actions"],
            "description": "compact=prompt-friendly view, full=complete tree, summary=one-liner, open_actions=pending actions ranked by rating.",
        },
    },
}

UPDATE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["add_action", "attach_result", "update_belief", "set_active", "advance_round"],
            "description": "What to do: add_action (plan a new action), attach_result (record eval), update_belief (revise ratings), set_active (switch branch), advance_round (start new round).",
        },
        "parent_node_id": {
            "type": "string",
            "description": "Parent node to attach to (for add_action).",
        },
        "node_id": {
            "type": "string",
            "description": "Target node (for attach_result, update_belief, set_active).",
        },
        "action_title": {
            "type": "string",
            "description": "Title of the action (for add_action). Keep it SMALL — one concrete single-iteration change.",
        },
        "action_description": {
            "type": "string",
            "description": "What the action does (1-2 sentences).",
        },
        "difficulty": {
            "type": "integer",
            "minimum": 1,
            "maximum": 5,
            "description": "1-5 difficulty. Prefer <= 3. 1=easy(tweak param), 5=hard(restructure algorithm).",
        },
        "expected_vs_baseline": {
            "type": "number",
            "description": "Expected improvement factor (>1 means better).",
        },
        "rationale": {
            "type": "string",
            "description": "Why this action is expected to work, grounded in evidence.",
        },
        "status": {
            "type": "string",
            "enum": ["PASSED", "FAILED", "TIMEOUT", "COMPILE_ERROR", "PARTIAL"],
            "description": "Evaluation result status.",
        },
        "score": {
            "type": "number",
            "description": "Numeric score from evaluation (lower is better for latency).",
        },
        "latency_ms": {
            "type": "number",
            "description": "Measured latency in milliseconds.",
        },
        "baseline_ms": {
            "type": "number",
            "description": "Baseline/reference latency for comparison.",
        },
        "speedup": {
            "type": "number",
            "description": "Speedup factor vs baseline (>1 is improvement).",
        },
        "rating": {
            "type": "number",
            "minimum": 0,
            "maximum": 10,
            "description": "Updated belief rating 0-10.",
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "Updated confidence 0-1.",
        },
        "notes": {
            "type": "string",
            "description": "Notes about what was learned from this result. Include FOLLOW_THROUGH (did implementation match intent?) and UPDATE_BELIEF (what changed).",
        },
    },
    "required": ["action"],
}

CHECK_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "node_id": {
            "type": "string",
            "description": "Node to check (typically the one that just got a result).",
        },
        "claimed_intent": {
            "type": "string",
            "description": "What you claim this action achieved, for FOLLOW_THROUGH comparison.",
        },
    },
    "required": ["node_id", "claimed_intent"],
}


# ---------------------------------------------------------------------------
# Handler functions
# ---------------------------------------------------------------------------

def _get_session_id(context: Dict[str, Any]) -> Optional[str]:
    """Best-effort extract of session id from invocation context."""
    for key in ("session_id", "session_key", "conversation_id", "task_name"):
        val = context.get(key)
        if val:
            return str(val)
    return None


def _get_wm_or_none(context: Dict[str, Any]) -> Optional[WorldModel]:
    """Get the WorldModel for the current session, only if already initialized.
    
    Does NOT auto-create. Returns None if no tree file exists.
    This prevents silent creation of empty trees when the agent hasn't
    called world_model_init yet.
    """
    session_id = _get_session_id(context)
    if not session_id:
        return None
    wm = WorldModel(task_name=session_id)
    if not wm.load():
        return None
    return wm


# ---- world_model_init ------------------------------------------------------


def handle_init(context: Dict[str, Any], params: Dict[str, Any]) -> dict:
    """Create a new decision tree. Overwrites any existing tree for this session."""
    session_id = _get_session_id(context) or "default"
    wm = WorldModel(task_name=session_id)
    wm.init_tree(
        kernel_summary=params.get("task_summary", ""),
        open_questions=params.get("open_questions", []),
    )
    wm.save()
    logger.info("world_model_init: created tree for session=%s", session_id)
    return {"status": "ok", "message": "Decision tree initialized", "task": session_id}


# ---- world_model_view ------------------------------------------------------


def handle_view(context: Dict[str, Any], params: Dict[str, Any]) -> dict:
    """View the current decision tree state."""
    wm = _get_wm_or_none(context)
    if wm is None or not wm._loaded:
        return {"status": "error", "message": "No decision tree found. Use world_model_init first."}

    mode = params.get("mode", "compact")

    if mode == "summary":
        return {"status": "ok", "content": wm.summary()}
    elif mode == "open_actions":
        actions = wm.get_open_actions(10)
        items = []
        for a in actions:
            items.append({
                "node_id": a.node_id,
                "parent_id": a.parent_id,
                "title": a.action.title if a.action else "",
                "difficulty": a.action.difficulty if a.action else "?",
                "expected_vs_baseline": a.action.expected_vs_baseline if a.action else None,
                "score": a.action.score if a.action else 0.0,
                "rating": a.rating,
                "confidence": a.confidence,
            })
        return {"status": "ok", "mode": mode, "open_actions": items}
    elif mode == "full":
        data = wm._serialize()
        raw = json.dumps(data, indent=2)
        return {"status": "ok", "mode": mode, "content": raw[:8000]}
    else:  # compact
        return {"status": "ok", "mode": mode, "content": wm.compact_view()}


# ---- world_model_update ----------------------------------------------------


def handle_update(context: Dict[str, Any], params: Dict[str, Any]) -> dict:
    """Update the decision tree with new action or result."""
    wm = _get_wm_or_none(context)
    if wm is None or not wm._loaded:
        return {"status": "error", "message": "No decision tree found. Use world_model_init first."}

    action = params["action"]

    if action == "add_action":
        parent_id = params.get("parent_node_id", wm.active_leaf_id)
        if parent_id not in wm.nodes:
            return {"status": "error", "message": f"Parent node {parent_id!r} not found"}
        try:
            node_id = wm.add_action(
                parent_id=parent_id,
                title=params.get("action_title", ""),
                description=params.get("action_description", ""),
                difficulty=params.get("difficulty", 3),
                expected_vs_baseline=_optional_float(params.get("expected_vs_baseline")),
                rationale=params.get("rationale", ""),
                confidence=_optional_float(params.get("confidence")) or 0.3,
            )
            wm.set_active_leaf(node_id)
            wm.save()
            return {"status": "ok", "node_id": node_id, "summary": wm.summary()}
        except ValueError as exc:
            logger.warning("[DecisionTree] set_active failed: %s", exc)
            return {"status": "error", "message": str(exc)}

    elif action == "attach_result":
        node_id = params.get("node_id", wm.active_leaf_id)
        if node_id not in wm.nodes:
            return {"status": "error", "message": f"Node {node_id!r} not found"}
        wm.attach_result(
            node_id=node_id,
            status=params.get("status", ""),
            score=_optional_float(params.get("score")),
            latency_ms=_optional_float(params.get("latency_ms")),
            baseline_ms=_optional_float(params.get("baseline_ms")),
            speedup=_optional_float(params.get("speedup")),
        )
        wm.save()
        return {"status": "ok", "node_id": node_id, "summary": wm.summary()}

    elif action == "update_belief":
        node_id = params.get("node_id", wm.active_leaf_id)
        if node_id not in wm.nodes:
            return {"status": "error", "message": f"Node {node_id!r} not found"}
        wm.update_belief(
            node_id=node_id,
            rating=_optional_float(params.get("rating")),
            confidence=_optional_float(params.get("confidence")),
            notes=params.get("notes", ""),
        )
        wm.save()
        return {"status": "ok", "node_id": node_id, "summary": wm.summary()}

    elif action == "set_active":
        node_id = params.get("node_id", "")
        if node_id not in wm.nodes:
            return {"status": "error", "message": f"Node {node_id!r} not found"}
        wm.set_active_leaf(node_id)
        wm.save()
        return {"status": "ok", "active_leaf_id": node_id, "summary": wm.summary()}

    elif action == "advance_round":
        wm.advance_round()
        wm.save()
        return {"status": "ok", "round": wm.round_index, "summary": wm.summary()}

    return {"status": "error", "message": f"Unknown action: {action!r}"}


# ---- world_model_check -----------------------------------------------------


def handle_check(context: Dict[str, Any], params: Dict[str, Any]) -> dict:
    """Run anti-self-deception checks on a node."""
    wm = _get_wm_or_none(context)
    if wm is None or not wm._loaded:
        return {"status": "error", "message": "No decision tree found. Use world_model_init first."}

    node_id = params["node_id"]
    claimed_intent = params.get("claimed_intent", "")

    report = wm.check_self_deception(node_id, claimed_intent)

    # Auto-save after check (beliefs may have been flagged)
    wm.save()

    return {
        "status": "ok",
        "has_gaps": report.has_gaps,
        "follow_through_ok": report.follow_through_ok,
        "follow_through_detail": report.follow_through_detail,
        "performance_gap": report.performance_gap,
        "performance_gap_detail": report.performance_gap_detail,
        "sibling_conflicts": report.sibling_conflicts,
        "zero_confidence_nodes": report.zero_confidence_nodes,
        "stagnation_alert": report.stagnation_alert,
        "stagnation_detail": report.stagnation_detail,
        "recommendation": (
            "SWITCH_BRANCH" if report.stagnation_alert
            else "INVESTIGATE_GAPS" if report.has_gaps
            else "CONTINUE"
        ),
        "summary": wm.summary(),
    }
