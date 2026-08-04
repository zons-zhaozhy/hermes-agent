"""decision-tree plugin — persistent belief-based decision tree for agent execution.

Wires three behaviours:

1. ``post_tool_call`` hook — auto-captures tool results as evidence for the
   active decision branch.  Zero agent compliance required.

2. ``pre_verify`` hook — enforces stagnation checks before the agent claims
   completion.  Blocks premature stops.

3. Four tools — ``world_model_init``, ``world_model_view``,
   ``world_model_update``, ``world_model_check`` — let the agent explicitly
   maintain a K-Search-style co-evolving world model.

Architecture:
    agent/decision_tree.py          (core data structures, no Hermes deps)
    plugins/decision_tree/tools.py   (tool handlers)
    plugins/decision_tree/hooks.py   (auto-track hooks)

Based on K-Search (arXiv 2602.19128): three-layer isolation —
WM maintenance → Action ranking → Codegen — with FOLLOW_THROUGH,
PERF_GAP, and SELF_CHECK anti-self-deception checks.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from .hooks import on_post_tool_call, on_pre_verify
from .tools import (
    CHECK_SCHEMA,
    INIT_SCHEMA,
    TOOL_CHECK,
    TOOL_INIT,
    TOOL_UPDATE,
    TOOL_VIEW,
    UPDATE_SCHEMA,
    VIEW_SCHEMA,
    handle_check,
    handle_init,
    handle_update,
    handle_view,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared: session_id extraction (single source of truth)
# ---------------------------------------------------------------------------

def _extract_session_id(kwargs: Dict[str, Any]) -> Optional[str]:
    """Extract session id from any context dict (tools or hooks).

    Kept in one place so tools.py and hooks.py never diverge on key names.
    """
    for key in ("session_id", "session_key", "conversation_id", "task_name"):
        val = kwargs.get(key)
        if val:
            return str(val)
    return None


# ---------------------------------------------------------------------------
# check_fn: gating tools by tree existence
# ---------------------------------------------------------------------------

def _tree_storage_dir() -> Path:
    home = os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))
    return Path(home) / "decision_tree"


def _any_tree_exists() -> bool:
    """Check if any decision tree file exists for this profile.

    Used as check_fn on view/update/check tools so they only appear
    after the agent has explicitly called world_model_init at least once.
    """
    storage = _tree_storage_dir()
    if not storage.exists():
        return False
    return any(storage.glob("*.json"))


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Register tools and hooks with the Hermes plugin system."""

    # --- Tools ----------------------------------------------------------

    # world_model_init is ALWAYS visible — the agent needs it to start.
    ctx.register_tool(
        name=TOOL_INIT,
        toolset="decision_tree",
        schema=INIT_SCHEMA,
        handler=handle_init,
        description="Create a persistent decision tree to track your strategies and beliefs across turns. Call at the start of any complex multi-step task.",
        emoji="🌳",
    )

    # view/update/check are gated — only appear when a tree already exists.
    # This keeps their schemas out of every API call for 99% of short tasks.
    _gated = {"check_fn": lambda _: _any_tree_exists()}

    ctx.register_tool(
        name=TOOL_VIEW,
        toolset="decision_tree",
        schema=VIEW_SCHEMA,
        handler=handle_view,
        description="View your current decision tree state. Use before making decisions to see what you've tried, what failed, and what open actions exist.",
        emoji="👁️",
        **_gated,
    )

    ctx.register_tool(
        name=TOOL_UPDATE,
        toolset="decision_tree",
        schema=UPDATE_SCHEMA,
        handler=handle_update,
        description="Update your decision tree: plan a new action, attach a result, revise belief ratings, or switch to a different branch.",
        emoji="✏️",
        **_gated,
    )

    ctx.register_tool(
        name=TOOL_CHECK,
        toolset="decision_tree",
        schema=CHECK_SCHEMA,
        handler=handle_check,
        description="Run anti-self-deception checks: FOLLOW_THROUGH (did implementation match intent?), PERF_GAP (expected vs observed?), stagnation detection, and sibling conflicts.",
        emoji="🔍",
        **_gated,
    )

    # --- Hooks ----------------------------------------------------------
    ctx.register_hook("post_tool_call", on_post_tool_call)
    ctx.register_hook("pre_verify", on_pre_verify)

    logger.info("decision-tree plugin registered: 4 tools + 2 hooks")
