"""Hooks for the decision-tree world model plugin.

Two hooks:
1. post_tool_call — auto-captures tool results for evidence-based belief updates
2. pre_verify — enforces stagnation checks before agent claims "done"

Zero agent compliance required: the hooks run automatically in the agent loop.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from agent.decision_tree import WorldModel

logger = logging.getLogger(__name__)

# Tools whose results we auto-capture for evidence
_AUTO_CAPTURE_TOOLS = frozenset({
    "terminal",
    "browser_navigate",
    "web_extract",
    "read_file",
    "search_files",
    "execute_code",
    "mcp",
})

# Tools whose success we treat as "evidence of progress"
_EVIDENCE_TOOLS = frozenset({
    "terminal",
    "write_file",
    "patch",
})


def _get_session_id(kwargs: Dict[str, Any]) -> Optional[str]:
    """Extract session id from hook kwargs. Best-effort across call sites.

    Single source of truth for session_id extraction across tools.py
    and hooks.py — keep the key list in sync.
    """
    for key in ("session_id", "session_key", "conversation_id", "task_name"):
        val = kwargs.get(key)
        if val:
            return str(val)
    return None


def _get_wm_or_none(kwargs: Dict[str, Any]) -> Optional[WorldModel]:
    """Get WorldModel for this session, only if already initialized.

    Does NOT auto-create. Returns None if no tree file exists.
    """
    session_id = _get_session_id(kwargs)
    if not session_id:
        return None
    wm = WorldModel(task_name=session_id)
    if not wm.load():
        return None
    return wm


def _tree_storage_dir() -> Path:
    """Get the decision tree storage directory, matching _default_storage_dir()."""
    home = os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))
    return Path(home) / "decision_tree"


# ---------------------------------------------------------------------------
# post_tool_call — auto-capture evidence
# ---------------------------------------------------------------------------

def on_post_tool_call(**kwargs: Any) -> None:
    """Auto-capture tool results as evidence for the active decision branch.

    Called automatically after every tool execution. Only acts when:
    1. A decision tree already exists for this session (agent must init it)
    2. The tool name is in _AUTO_CAPTURE_TOOLS
    3. The tool succeeded (can derive status from result)

    Captures:
    - Tool name + success/failure → evidence for current branch
    - Auto-advances round when evidence is written
    - Saves tree after update
    """
    try:
        tool_name = kwargs.get("tool_name", "")
        if tool_name not in _AUTO_CAPTURE_TOOLS:
            return

        wm = _get_wm_or_none(kwargs)
        if wm is None:
            return

        # Determine success/failure
        result = kwargs.get("result", {})
        status = _derive_status(tool_name, result)
        if not status:
            return  # Can't determine outcome, skip

        # Find the current active leaf (or root's first child)
        active_id = wm.active_leaf_id
        if active_id == "root":
            children = wm.get_children("root")
            if children:
                active_id = children[0].node_id
            else:
                return  # No action node to attach evidence to

        # If this is an evidence tool (terminal, write_file, patch), log it
        if tool_name in _EVIDENCE_TOOLS and active_id in wm.nodes:
            node = wm.nodes[active_id]
            existing_notes = node.notes
            evidence_line = f"[EVIDENCE r{wm.round_index}] {tool_name}: {status}"
            if existing_notes:
                node.notes = f"{existing_notes}\n{evidence_line}" if evidence_line not in existing_notes else existing_notes
            else:
                node.notes = evidence_line
            node.last_updated_round = wm.round_index

            # Auto-advance round and track result for stagnation detection
            wm.round_index += 1
            wm.round_results.append(status)
            if len(wm.round_results) > wm.stagnation_window * 3:
                wm.round_results = wm.round_results[-wm.stagnation_window * 3:]

            wm.save()
            logger.debug("decision_tree hook: auto-captured %s → %s for %s (round %d)",
                        tool_name, status, active_id, wm.round_index)

    except Exception:
        logger.warning("post_tool_call hook error", exc_info=True)


def _derive_status(tool_name: str, result: Any) -> Optional[str]:
    """Derive a status string from tool result."""
    if isinstance(result, dict):
        if "status" in result:
            return str(result["status"]).upper()
        if result.get("error"):
            return "FAILED"
        if "exit_code" in result:
            return "PASSED" if result.get("exit_code") == 0 else "FAILED"
        if "content" in result and result["content"]:
            return "PASSED"
    if result is not None:
        return "PASSED"
    return None


# ---------------------------------------------------------------------------
# pre_verify — anti-self-deception gate
# ---------------------------------------------------------------------------

def on_pre_verify(**kwargs: Any) -> Optional[Dict[str, str]]:
    """Check for stagnation before the agent claims completion.

    Runs automatically when the agent is about to verify/finish a task.
    If the decision tree shows stagnation (N consecutive non-PASSED rounds),
    returns a follow-up instruction to keep the agent working.

    Returns:
        {"action": "continue", "message": "..."} to block premature stop
        None to let the stop proceed
    """
    try:
        wm = _get_wm_or_none(kwargs)
        if wm is None:
            return None  # No tree = nothing to check

        # Run stagnation check
        if len(wm.round_results) >= wm.stagnation_window:
            recent = wm.round_results[-wm.stagnation_window:]
            if all(r not in ("PASSED", "PARTIAL") for r in recent):
                return {
                    "action": "continue",
                    "message": (
                        f"Decision tree stagnation: last {wm.stagnation_window} "
                        f"rounds all non-PASSED ({recent}). Switch branch via "
                        f"world_model_update(set_active=<node_id>) or "
                        f"world_model_view(mode='open_actions') to see alternatives."
                    ),
                }

        # Check for zero-confidence open nodes (plans without evidence)
        zero_ids = []
        for nid, n in wm.nodes.items():
            if n.is_open and n.confidence == 0.0 and n.rating == 0.0:
                zero_ids.append(nid)
        if len(zero_ids) >= 2:
            return {
                "action": "continue",
                "message": (
                    f"Decision tree has {len(zero_ids)} actions with zero confidence. "
                    f"Fill them with evidence or prune. Run world_model_check("
                    f"node_id='{wm.active_leaf_id}', claimed_intent='<your intent>')."
                ),
            }

        return None

    except Exception:
        logger.warning("pre_verify hook error", exc_info=True)
        return None
