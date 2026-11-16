"""Credential-pool error context for the turn's recovery chain.

Splits from ``turn_recovery`` so the facade stays under its FILE_LINES cap. The logger
keeps the ``agent.conversation_loop`` name: caplog sites pin that logger, not the module.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("agent.conversation_loop")


def _pool_error_context(agent, classified: Any, error_context: Any) -> Any:
    """What the credential pool sees: the turn's raw context merged with the classifier's hints.

    Contract:
      Preconditions: ``classified`` is a ClassifiedError; ``error_context`` is the turn's
        extracted provider error mapping (or None).
      Postconditions: the returned mapping keeps every raw key, overlays the classifier's
        context, and never raises.

    A Z.AI 1302 transient throttle carries a short ``reset_at`` the pool would otherwise drop
    (it is stored on ``classified.error_context``, not the raw ``extract_api_error_context``);
    without it the pool benches every key for its generic multi-entry 429 TTL (one hour) even
    though Z.AI asked for a brief slowdown. A plan limit (usage-window 402) benches only the
    refused model on the credential, until its reset."""
    raw = error_context if isinstance(error_context, dict) else {}
    classified_ctx = getattr(classified, "error_context", None) or {}
    if not classified_ctx.get("usage_window"):
        return {**raw, **classified_ctx}
    return {**raw, "usage_window": True, **_usage_window_reset_context(agent)}


def _usage_window_reset_context(agent) -> dict[str, float]:
    """``{"reset_at": epoch s}`` for a usage-window 402: when the refused allowance frees up, from the usage
    headers the 402 carried (the API's error body has no reset). Empty when the state names none, so the
    pool keeps its default bench.

    Contract:
      Postconditions: returns ``{}`` or ``{"reset_at": positive epoch seconds}``; never raises."""
    try:
        from agent.credits_tracker import window_stop_resets_at_ms
        from agent.plan_limits import current_model_is_top_tier
        state = getattr(agent, "_credits_state", None)
        resets_ms = (window_stop_resets_at_ms(state, model_is_top_tier=current_model_is_top_tier(agent))
                     if state is not None else None)
    except Exception:
        logger.warning("usage-window reset unavailable for the pool", exc_info=True)
        return {}
    return {"reset_at": resets_ms / 1000} if isinstance(resets_ms, (int, float)) and resets_ms > 0 else {}
