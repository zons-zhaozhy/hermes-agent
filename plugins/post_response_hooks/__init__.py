"""
Post-Response Hooks Plugin for Hermes Agent.

Bridges the post_response_hooks framework (agent/post_response_hooks.py)
into the official plugin lifecycle via the `post_llm_call` hook.

This is the ONLY correct integration point — no modifications to run_agent.py
are needed. The plugin receives session_id, user_message, and assistant_response
from the official hook and passes them to our custom hooks.

Hooks loaded from ~/.hermes/hooks/ (configured in config.yaml):
  - bottom_logic_check
  - correction_tracker
  - behavior_regression
"""

import importlib.util
import json
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Import the official hook framework
_AGENT_DIR = Path.home() / ".hermes" / "hermes-agent"
_HOOKS_FRAMEWORK = _AGENT_DIR / "agent" / "post_response_hooks.py"

_hooks_loaded = False
_hooks = []


def _ensure_framework_importable():
    """Add agent dir to sys.path if needed."""
    agent_dir = str(_AGENT_DIR)
    if agent_dir not in sys.path:
        sys.path.insert(0, agent_dir)


def _load_hooks_from_config():
    """Load hooks configured in ~/.hermes/config.yaml."""
    global _hooks_loaded, _hooks

    if _hooks_loaded:
        return _hooks

    _ensure_framework_importable()

    # Load config through the canonical loader (managed-scope overlay +
    # ${ENV_VAR} expansion + profile-aware pathing), never a raw read.
    hook_configs = []
    try:
        from hermes_cli.config import load_config_readonly
        cfg = load_config_readonly() or {}
        agent_cfg = cfg.get("agent") or {}
        hook_configs = agent_cfg.get("post_response_hooks") or []
    except Exception as e:
        logger.warning("Failed to load hook config: %s", e)

    if not hook_configs:
        logger.debug("No post_response_hooks configured")
        _hooks_loaded = True
        return []

    try:
        from agent.post_response_hooks import load_hooks
        _hooks = load_hooks(hook_configs)
        _hooks_loaded = True
        logger.info("Loaded %d post-response hooks via plugin", len(_hooks))
    except Exception as e:
        logger.error("Failed to load hooks framework: %s", e)
        _hooks_loaded = True

    return _hooks


def _on_post_llm_call(**kwargs):
    """Callback for post_llm_call plugin hook.

    Receives from run_agent.py (line 11567-11576):
      - session_id
      - user_message
      - assistant_response
      - conversation_history
      - model
      - platform
    """
    hooks = _load_hooks_from_config()
    if not hooks:
        return

    session_id = kwargs.get("session_id", "unknown")
    user_message = kwargs.get("user_message", "")
    assistant_response = kwargs.get("assistant_response", "")
    model = kwargs.get("model", "")

    context = {
        "session_id": session_id,
        "user_message": user_message,
        "model": model,
        "platform": kwargs.get("platform", ""),
    }

    try:
        from agent.post_response_hooks import run_post_response_checks
        result = run_post_response_checks(hooks, assistant_response, context)

        if result and not result.passed:
            logger.warning(
                "Post-response hook triggered: action=%s session=%s",
                result.action, session_id,
            )
            # For nudges: log but don't modify (model already responded)
            # For blocks: log the blocked content
            if result.message:
                logger.info("Hook message: %s", result.message[:200])
    except Exception as e:
        logger.error("Post-response hook execution failed: %s", e)


def register(ctx):
    """Plugin entry point — register our hooks."""
    ctx.register_hook("post_llm_call", _on_post_llm_call)
    logger.info("post-response-hooks plugin registered")
