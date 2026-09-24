"""read-think-gate plugin — host for the ReadThinkGate (four-axis deliberation gate).

Externalized from ``agent/read_think_gate.py`` + its ``tool_executor`` /
``agent_init`` wiring (fork narrow-waist rule: capability lives at the edges).
Behavior contract is unchanged:

* one ``check_batch`` per assistant_message (the executor's ``pre_tool_batch``
  hook fires exactly once per batch, at the same two dispatch seams the old
  core wiring used);
* four-axis evidence accumulates within a turn and spills to
  ``~/.hermes/cache/four_axis_gate.json`` for the guards plugin's second line;
* cron platforms are exempt (unattended jobs cannot answer gate prompts);
* a crashing gate fails CLOSED at the dispatcher (policy hooks fail closed:
  the crash surfaces as a named block directive in the tool result, never a
  silent allow); the emitter itself is still crash-safe — a dispatch failure
  logs a warning, it cannot take the executor down.

The gate instance is per-session (``_GATES`` registry keyed by session_id,
dropped on session end/reset). Config section: ``read_think_gate`` in
config.yaml — the same section the former core init read.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from .gate import ReadThinkGate, ReadThinkGateConfig

logger = logging.getLogger("plugins.read-think-gate")

# session_id -> gate instance. Cron sessions get an instance with enabled=False,
# mirroring the agent_init exemption the core wiring used to apply.
_GATES: Dict[str, ReadThinkGate] = {}
_ACTIVE_SESSION_ID: Optional[str] = None


def _config_section() -> Dict[str, Any]:
    """Read the ``read_think_gate`` config section; ``{}`` on any failure."""
    try:
        from hermes_cli.config import load_config_readonly

        cfg = load_config_readonly() or {}
        section = cfg.get("read_think_gate")
        return section if isinstance(section, dict) else {}
    except Exception:
        logger.warning("read-think-gate: config read failed; using empty section", exc_info=True)
        return {}


def _gate_for(session_id: str, platform: str) -> ReadThinkGate:
    """Get or create this session's gate, applying the cron exemption.

    Contract:
      Postconditions: never raises; a config failure yields the default
        config, and platform "cron" always yields enabled=False.
    """
    gate = _GATES.get(session_id)
    if gate is not None:
        return gate
    try:
        cfg = ReadThinkGateConfig.from_mapping(_config_section())
    except Exception:
        logger.warning("read-think-gate: config ignored", exc_info=True)
        cfg = ReadThinkGateConfig()
    if platform == "cron":
        import dataclasses as _dc

        cfg = _dc.replace(cfg, enabled=False)
    gate = ReadThinkGate(cfg)
    _GATES[session_id] = gate
    return gate


def _current_gate(session_id: str, platform: str) -> Optional[ReadThinkGate]:
    """Resolve the gate this batch belongs to; None when unkeyed (no session)."""
    if not session_id:
        return None
    return _gate_for(session_id, platform)


def on_session_start(session_id: str = "", **_kwargs: Any) -> None:
    """Key the active session (turn-scoped state restarts per session)."""
    global _ACTIVE_SESSION_ID
    _ACTIVE_SESSION_ID = session_id or None


def on_session_reset(session_id: str = "", **_kwargs: Any) -> None:
    """Drop the session's gate — next batch builds a fresh one."""
    _GATES.pop(session_id, None)
    global _ACTIVE_SESSION_ID
    if _ACTIVE_SESSION_ID == (session_id or None):
        _ACTIVE_SESSION_ID = None


def on_session_end(session_id: str = "", **_kwargs: Any) -> None:
    """Drop the session's gate (mirror of reset; either name may fire)."""
    _GATES.pop(session_id, None)
    global _ACTIVE_SESSION_ID
    if _ACTIVE_SESSION_ID == (session_id or None):
        _ACTIVE_SESSION_ID = None


def pre_llm_call(
    user_message: str = "",
    session_id: str = "",
    platform: str = "",
    is_first_turn: bool = False,
    **_kwargs: Any,
) -> None:
    """Per-turn reset — the turn's user message drives complexity classification.

    The core ``build_turn_context`` used to call ``gate.reset_for_turn(user_message)``
    directly; the hook carries the same payload (user_message + session identity),
    so the gate sees exactly one reset per turn, before the first LLM call fires.
    """
    try:
        gate = _current_gate(session_id or _ACTIVE_SESSION_ID or "", platform)
        if gate is None:
            return
        message = user_message if isinstance(user_message, str) else str(user_message or "")
        gate.reset_for_turn(user_message=message or None)
    except Exception:
        logger.warning("read-think-gate: reset_for_turn failed", exc_info=True)


def pre_tool_batch(
    assistant_content: str = "",
    tool_calls: Optional[list] = None,
    session_id: str = "",
    platform: str = "",
    **_kwargs: Any,
) -> Optional[Dict[str, Any]]:
    """Batch-level gate check; return ``{"action": "block", "message": ...}`` or None.

    Contract:
      Postconditions: a crashing gate RAISES — the dispatcher converts the
        exception into a fail-closed, named block directive (policy hooks
        fail closed; see ``hermes_cli.plugins_dispatch``), so a broken guard
        is visible in the tool result instead of silently allowing writes.
        Only the session-registry lookup itself is guarded (no gate for a
        batch is a legitimate "no decision" state, distinct from a crash).
    """
    gate = _current_gate(session_id, platform)
    if gate is None:
        return None
    names = [str(tc.get("name") or "") for tc in (tool_calls or []) if isinstance(tc, dict)]
    args = [tc.get("args") for tc in (tool_calls or []) if isinstance(tc, dict)]
    block = gate.check_batch(assistant_content or "", names, tool_args=args)
    if block is None:
        return None
    return {"action": "block", "message": block}


def register(ctx) -> None:
    """Plugin-loader contract — without this the loader refuses the plugin
    (``Plugin 'read-think-gate' has no register() function``) and the four-axis
    marker writer never runs, deadlocking every write tool behind the guards
    plugin's secondary line.

    Hook names mirror the ``hooks:`` list in plugin.yaml exactly.
    """
    ctx.register_hook("pre_tool_batch", pre_tool_batch)
    ctx.register_hook("pre_llm_call", pre_llm_call)
    ctx.register_hook("on_session_start", on_session_start)
    ctx.register_hook("on_session_reset", on_session_reset)
    ctx.register_hook("on_session_end", on_session_end)
    logger.info("read-think-gate registered (5 hooks)")
