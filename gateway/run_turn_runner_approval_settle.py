"""Tell the chat when an exec-approval prompt times out (messaging platforms).

``tools.approval_gateway_wait._await_gateway_decision`` calls ``entry.settle(reason)`` once the
wait ends. Only the TUI registered such a hook, so on Telegram / Slack / WhatsApp a card whose
timer ran out kept live buttons and the user never learned the command did NOT run. The turn
runner registers the hook here right after the prompt was delivered.

Best-effort by design: a failed notice is logged at debug — the approval already resolved as
"no", and nothing here may block the agent thread.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Optional

from agent.i18n import t
from gateway.platforms.base import fit_prefix
from gateway.platforms.base_exec_approval import (
    approval_timeout_seconds, ea_default_reason_text, ea_reason_label_text, format_approval_timed_out_notice)

logger = logging.getLogger(__name__)


def register_timeout_notice(
    runner, approval_data: dict, *, command: str, card_message_id: Optional[str]) -> None:
    """Arm a settle hook that posts the timed-out notice for ``approval_data['request_id']``.

    ``runner`` is the ``TurnRunner`` (for ``_ctx`` and ``_schedule``); ``card_message_id`` is the
    delivered BUTTON card's id when the adapter returned one, so the card itself is edited in place
    (which also drops its buttons). The plain-text prompt passes ``None``: it has no buttons to
    drop and rewriting it would erase the record of what was asked. ``command`` is the
    already-redacted command shown to the user. The notice is skipped when the run is no longer
    current (``ctx._run_still_current``).
    """
    from tools.approval import register_gateway_settle

    request_id = approval_data.get("request_id")
    session_key = runner._ctx.session_key or ""
    if not request_id or not session_key:
        return
    timeout_s = approval_data.get("timeout_s")
    if timeout_s is None:
        timeout_s = approval_timeout_seconds()
    flagged = approval_data.get("description") or ea_default_reason_text()

    def settle(reason: str) -> None:
        if reason != "timeout":
            return  # answered / interrupted / notify_failed already produced their own feedback
        # Same guard as every other late notice in TurnRunner: after /stop, /new or a restart the
        # turn is over and this chat belongs to a newer run — do not edit or post into it.
        still_current = getattr(runner._ctx, "_run_still_current", None)
        if callable(still_current) and not still_current():
            return
        runner._schedule(
            _post_timeout_notice(runner._ctx, command, card_message_id, timeout_s, flagged),
            "Approval timeout notice scheduling error")

    register_gateway_settle(session_key, request_id, settle)


async def _post_timeout_notice(
        ctx, command: str, card_message_id: Optional[str], timeout_s: int, reason: str = "") -> None:
    from gateway.run import _interim_metadata

    from gateway.bot_peer_notices import automatic_notices_allowed

    adapter = ctx._status_adapter
    if not automatic_notices_allowed(getattr(ctx, "source", None)):
        return
    notice = format_approval_timed_out_notice(timeout_s)
    metadata = _interim_metadata(ctx._status_thread_metadata)
    try:
        # Plain markdown, not the card's platform markup: ``edit_message`` re-formats it itself.
        if card_message_id and await _edit_card(adapter, ctx._status_chat_id, card_message_id, notice, command, reason):
            return
        await adapter.send(ctx._status_chat_id, notice, metadata=metadata)
    except Exception:
        logger.debug("Approval timeout notice failed", exc_info=True)


def _timed_out_card_body(adapter, chat_id: str, notice: str, command: str, reason: str = "") -> str:
    """The notice plus the reason and command, which are now the card's only record of the request. It must fit one
    message in the platform's own units (Telegram counts UTF-16), or the edit is truncated in
    place behind a dangling "(1/N)" and keeps streaming-preview state. A run of backticks in the
    command would close the card's fence and let the platform's formatter rewrite, and grow,
    the rest, so it is split with zero-width spaces (which also ride along in a copy)."""
    command, reason = (re.sub(r"`{3,}", lambda run: "\u200b".join(run.group()), s) for s in (command, reason))
    measure, suffix = adapter.message_len_fn_for_chat(chat_id), "..."
    if reason:  # capped like the card's own reason, so the command keeps most of the budget
        notice = f"{notice}\n{ea_reason_label_text()}: {fit_prefix(reason, 500, measure, suffix)}"
    frame = measure(t("gateway.exec_approval.timed_out_card_body", notice=notice, command=""))
    budget = adapter.max_message_length_for_chat(chat_id) - frame - measure(suffix)
    return t("gateway.exec_approval.timed_out_card_body", notice=notice,
             command=fit_prefix(command, max(0, budget), measure, suffix))


async def _edit_card(adapter, chat_id: str, message_id: str, notice: str, command: str, reason: str = "") -> bool:
    """Edit the card in place to the timed-out body; False sends the notice as a new message.
    Platforms whose edit replaces the markup drop the buttons here; Discord's edit keeps them,
    and its card's settle observer removes them."""
    edit: Optional[Callable[..., Any]] = getattr(adapter, "edit_message", None)
    if edit is None:
        return False
    try:
        result = await edit(chat_id, message_id, _timed_out_card_body(adapter, chat_id, notice, command, reason))
    except Exception:
        logger.debug("Approval card edit failed; sending the notice as a new message", exc_info=True)
        return False
    return bool(getattr(result, "success", False))


def _approval_send_outcome(future, timeout: float) -> str:
    """Classify an approval prompt send as ``sent`` / ``failed`` / ``ambiguous``.

    ``ambiguous`` = future timed out but the card may have posted: keep the registration, do NOT re-send.
    Only a DEFINITIVE failure (error result / non-timeout exception / no future) re-asks; logged here.

    Lives beside the approval notice lane it classifies (moved out of ``gateway/run.py``); the
    facade re-exports it through a thin delegator so ``from gateway.run import ...`` callers and
    tests keep resolving the seam they patch.
    """
    import concurrent.futures

    if future is None:
        logger.warning("Prompt send failed: no scheduling future (loop unavailable)")
        return "failed"
    try:
        result = future.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        return "ambiguous"
    except Exception as exc:  # health: allow BLE001 -- a caller-supplied send future can raise anything; every failure is the same verdict ("failed"), and this is the common non-timeout path so a traceback per occurrence would bury the log
        logger.warning("Prompt send failed: %s", exc)
        return "failed"
    if getattr(result, "success", False):
        return "sent"
    # P5(b): a connector DECLINE is not a lane failure. The connector authorized the destination and refused it;
    # re-sending the same content as plain text into that same chat is the exfiltration the egress guard exists
    # to stop. `failed` is the cue to fall back, so a decline needs its own verdict — callers must surface it
    # and send nothing further.
    #
    # CLASSIFY THE STRUCTURED RESPONSE, NOT THE ERROR STRING. The adapter
    # preserves the connector's own dict in `raw_response`; rebuilding a dict
    # from `error` alone loses two things review demonstrated:
    #   * a decline carrying `code: egress_declined` and NO text renders as
    #     "relay egress declined" — no marker colon — so the string check
    #     missed it and the fallback fired into the refused chat;
    #   * `ambiguous: True` (lost ack, mid-write drop) was flattened into a
    #     DEFINITE failure, which re-sends a card that may well have posted.
    # I fixed the text-marker path and tested only the text-marker path.
    from gateway.relay.egress import declined_send

    _raw = getattr(result, "raw_response", None)
    if isinstance(_raw, dict) and _raw.get("ambiguous"):
        # The frame may have been applied. Same physics as a scheduling
        # timeout: possibly-delivered, so never re-send. Checked BEFORE the
        # decline classification because an ambiguous result is a transport
        # outcome, not an authorization one, and this lane has three verdicts
        # rather than the boolean the shared helper answers.
        logger.warning("Prompt send AMBIGUOUS (lost ack): %s", _raw.get("error"))
        return "ambiguous"
    if declined_send(result):
        # Both shapes, one classifier: a structured body, or the uniform
        # decline sentence from an older connector.
        logger.warning(
            "Prompt send DECLINED by connector egress guard: %s",
            getattr(result, "error", None),
        )
        return "declined"
    logger.warning("Prompt send failed: %s", getattr(result, "error", None) or "unknown error")
    return "failed"
