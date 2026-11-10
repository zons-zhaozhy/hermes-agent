"""Outward-text helpers for cron delivery: the one secret-redaction policy and the two wrappers
that splice a job name around a redacted payload (mirror sinks, thread titles).

Split out of ``cron.scheduler_delivery`` so that module stays under its file-size cap; the
delivery lanes import the names from here.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("cron.scheduler")


def _redact_cron_payload(text: str, what: str) -> str:
    """Fail-closed secret redaction for anything a cron job emits outward.

    Every outward lane — chat message, session mirror, bot-chat turn — must apply the same policy,
    so the policy lives in one place. ``force=True`` because this is a safety boundary, not
    logging: the ``security.redact_secrets`` preference governs how much is scrubbed from the
    user's own logs and must not be able to turn scrubbing off on the way out to a chat (same
    reasoning as ``tools/delegation_live_log.py``). Empty input is returned as-is; any failure
    inside the redactor replaces the payload entirely rather than letting an unscanned value out.
    """
    if not text:
        return text
    try:
        from agent.redact import redact_sensitive_text
        return redact_sensitive_text(text, force=True)
    except Exception as e:
        logger.warning("Failed to redact secrets from cron %s: %s", what, e)
        return "[REDACTED - redaction failed]"


def _cron_display_name(job: dict) -> str:
    """Job name/id as it appears in outward-facing text. The mirror sinks and the thread title
    splice the job *name* around the redacted payload, and the name is user-controlled config — a
    name embedding a credential would re-leak it next to the scrubbed body."""
    return _redact_cron_payload(job.get("name") or job.get("id", "cron"), "job name")


def _cron_mirror_message(job: dict, text: str) -> str:
    return f"[Cron delivery: {_cron_display_name(job)}]\n{text}"
