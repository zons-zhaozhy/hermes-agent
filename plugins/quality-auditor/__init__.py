"""quality-auditor plugin — revives the dead-wired quality auditor.

WHY THIS PLUGIN EXISTS
======================
``agent/quality_auditor.py`` (401 lines, 10-dimension audit on an auxiliary
model) has existed since commit 6255435ec5 (2026-07-10) but its call-site
wiring inside ``turn_finalizer`` was wiped by the upstream sync 22d6d2a6f3
(2026-08-04). The module survived, the config survived, the call site did
not — ``quality_audit.jsonl`` stopped growing on 07-21 and nobody noticed
for 28 days (found during the 2026-08-18 gradient-spiral health audit).

Re-wiring inside core files would repeat the same failure: the next upstream
sync can wash it away again. This plugin is the permanent fix — plugins
survive syncs untouched.

ARCHITECTURE (zero core changes)
================================
* ``post_llm_call`` (fired once per turn, turn_finalizer.py:615-634, only
  when a final response was produced and the turn was not interrupted):
  → calls ``fire_quality_audit(...)`` from the existing module.
    Daemon thread + 60s HTTP timeout — zero latency on the turn path.
    Appends to ~/.hermes/state/quality_audit.jsonl.

* ``pre_llm_call`` (fired before the turn's first API call,
  turn_context.py:1197-1260):
  → calls ``get_last_audit_feedback(session_id)`` (1-hour freshness window,
    same-session match). If feedback exists, returns
    ``{"context": "..."}`` which the official injection channel appends to
    the API copy of the user message (compose_user_api_content,
    turn_context.py:54). The stored transcript stays clean and the system
    prompt is never touched — the prompt-cache prefix invariant holds.

Feedback injection is gated to only fire when the previous audit found
issues (get_last_audit_feedback returns None for clean turns), so ordinary
conversation pays zero overhead.

CONFIG (~/.hermes/config.yaml)
=============================
    plugins:
      enabled:
        - quality-auditor          # master switch (this allow-list entry)
    auxiliary:
      quality_auditor:             # which model audits (cheap model advised)
        provider: zai
        model: glm-5-turbo
        base_url: ...
        timeout: 60

OFF-SWITCH
==========
Remove ``quality-auditor`` from ``plugins.enabled`` — the module then never
loads and behavior is identical to pre-plugin state.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Minimum response length for an audit to be meaningful — mirrors the
# module's own guard (fire_quality_audit skips <50 chars); the plugin-level
# gate avoids spawning the import machinery for trivial turns.
_MIN_RESPONSE_CHARS = 50


def _extract_tool_stats(conversation_history) -> tuple[int, list[str]]:
    """Derive (tool_call_count, tool_names) from the turn's transcript.

    The post_llm_call hook carries the full conversation history; assistant
    rows with ``tool_calls`` describe every tool invocation this session.
    We count only rows after the current turn's user message so the audit
    reflects per-turn tool usage (matching the original wiring, which
    passed ``api_call_count``).

    Contract:
      Postconditions: returns (count, names) with count >= 0, count ==
      len(names), names unique, and every element a non-empty str.
    """
    if not conversation_history:
        return 0, []
    # Find the LAST user row — everything after it is the current turn.
    last_user_idx = -1
    for i, msg in enumerate(conversation_history):
        if isinstance(msg, dict) and msg.get("role") == "user":
            last_user_idx = i
    count = 0
    names: list[str] = []
    for msg in conversation_history[last_user_idx + 1:]:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        calls = msg.get("tool_calls") or []
        if not isinstance(calls, list):
            continue
        for call in calls:
            if not isinstance(call, dict):
                continue
            fn = call.get("function") or {}
            name = fn.get("name") if isinstance(fn, dict) else None
            if name:
                names.append(str(name))
                count += 1
    assert len(names) == count and len(set(names)) == len(names)
    return count, names


def on_post_llm_call(**kwargs):
    """Fire the quality audit for the completed turn (daemon thread).

    Contract:
      Postconditions: never raises; returns None; only fires the audit when
      assistant_response length >= _MIN_RESPONSE_CHARS.
    """
    assistant_response = kwargs.get("assistant_response") or ""
    if len(assistant_response.strip()) < _MIN_RESPONSE_CHARS:
        return  # trivial turn — nothing to audit

    user_message = kwargs.get("user_message") or ""
    session_id = kwargs.get("session_id") or ""
    model = kwargs.get("model") or ""
    conversation_history = kwargs.get("conversation_history") or []
    tool_call_count, tool_names = _extract_tool_stats(conversation_history)

    try:
        from agent.quality_auditor import fire_quality_audit
    except Exception:
        logger.exception("quality-auditor: module import failed")
        return

    try:
        fire_quality_audit(
            user_message=user_message,
            assistant_response=assistant_response,
            session_id=session_id,
            model=model,
            tool_call_count=tool_call_count,
            tool_names=tool_names,
        )
    except Exception:
        # fire_quality_audit already swallows internally; this guards the
        # plugin boundary itself (a plugin must never break the host).
        logger.exception("quality-auditor: fire_quality_audit failed")


def on_pre_llm_call(**kwargs):
    """Inject last-turn audit feedback into this turn's user message.

    Returns ``{"context": ...}`` consumed by the official injection channel
    (appended to the API copy of the user message only). Returns None when
    no fresh feedback exists — zero overhead for clean turns.

    Contract:
      Postconditions: never raises; returns None or {"context": non-empty str}.
    """
    session_id = kwargs.get("session_id") or ""
    if not session_id:
        return None
    try:
        from agent.quality_auditor import get_last_audit_feedback
        feedback = get_last_audit_feedback(session_id)
    except Exception:
        logger.exception("quality-auditor: feedback read failed")
        return None
    if not feedback:
        return None
    return {"context": feedback}


def register(ctx) -> None:  # noqa: ANN001 — PluginContext from loader
    ctx.register_hook("post_llm_call", on_post_llm_call)
    ctx.register_hook("pre_llm_call", on_pre_llm_call)
    logger.info("quality-auditor plugin registered")
