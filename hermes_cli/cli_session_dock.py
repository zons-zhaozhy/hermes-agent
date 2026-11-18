"""Goal, loop + queued-prompt rows for the classic CLI live-work dock.

Sibling of ``cli_process_dock``: the dock paints an active ``/goal``, a standing ``/loop`` with its
countdown, and the prompts waiting in ``/queue`` around the subagent and process blocks, so they are
visible while the agent works instead of only through ``/goal status``, ``/loop status`` and
``/queue list``.
"""
from __future__ import annotations

import math
import time

# Queued prompts drain one per turn; past this many the dock shows a ``+N more`` line.
QUEUE_ROWS = 3

# The gateway's viewer-automation liveness window is seconds wide (sized for the judge's latency
# after a turn), while an agentic turn runs for minutes with no ``session.automation`` traffic —
# without a refresh the gateway takes the goal loop over at every long turn's end and the CLI
# window that submitted the turn never sees the continuations. The dock's 1-second poll carries
# this throttled keepalive: one ``state`` read per active remote manager per window. A dead
# transport fails inside ``_rpc`` (debug log, mirror kept), so the gateway's stamp still ages out
# and its takeover of a genuinely dead viewer is unchanged.
KEEPALIVE_SECONDS = 20.0


def sync_automation(cli) -> None:
    """Re-read the goal/loop managers' mirrored state: once after the gateway flagged a mid-turn
    change (``_mark_automation_stale``), plus the throttled keepalive for active remote managers
    (see ``KEEPALIVE_SECONDS``). One ``state`` round trip each, none while nothing is due."""
    stale = getattr(cli, "_automation_stale", False)
    now = time.monotonic()
    keepalive_due = now - getattr(cli, "_automation_keepalive_at", 0.0) >= KEEPALIVE_SECONDS
    if not stale and not keepalive_due:
        return
    cli._automation_stale = False
    if keepalive_due:
        cli._automation_keepalive_at = now
    if keepalive_due and not stale:
        from hermes_cli.gateway_automation import _Remote
    else:
        _Remote = ()  # the stale path refreshes every manager kind, as before
    for attr in ("_goal_manager", "_loop_manager"):
        mgr = getattr(cli, attr, None)
        if mgr is None or mgr.session_id != getattr(cli, "session_id", None):
            continue
        if stale or (keepalive_due and isinstance(mgr, _Remote) and mgr.is_active()):
            mgr.refresh()


def goal_line(cli) -> str:
    """``GoalManager.status_line()`` for an active or paused goal, else ``''``.

    Reads only the manager the CLI already holds for its current session: the dock refreshes on
    the spinner thread every second and must never be the thing that opens state.db.
    """
    mgr = getattr(cli, "_goal_manager", None)
    if mgr is None or mgr.session_id != getattr(cli, "session_id", None) or not mgr.has_goal():
        return ""
    return mgr.status_line()


def countdown(seconds: float) -> str:
    """Time left until the next wakeup, rounded UP: whole minutes until the last one, then seconds.

    Minute steps keep an idle prompt's dock to one repaint a minute (the spinner thread repaints only
    when the dock text changes); per-second repaints of a 45-minute wait would fight terminal viewport
    restoration for nothing.
    """
    left = max(0, math.ceil(seconds))
    if left < 60:
        return f"{left}s"
    hours, minutes = divmod(math.ceil(left / 60), 60)
    return f"{hours}h{minutes:02d}m" if hours else f"{minutes}m"


def loop_row(cli, now: float | None = None) -> tuple[str, str, str]:
    """``(badge, line, prompt)`` for an active or paused /loop, else ``('', '', '')``.

    ``line`` is the expanded dock row (``↻ loop · every 5m · 2 runs · next in 4m · <prompt>``),
    ``badge`` the collapsed heading's count fragment (``↻ loop next in 4m``). Same rule as :func:`goal_line`:
    only the manager the CLI already holds is read, never state.db.
    """
    mgr = getattr(cli, "_loop_manager", None)
    # One read of ``state``: the input thread's ``/loop stop`` sets it to None under this spinner-thread poll.
    s = mgr.state if mgr is not None and mgr.session_id == getattr(cli, "session_id", None) else None
    if s is None or s.status not in {"active", "paused"}:
        return "", "", ""
    runs = f"{s.ticks_fired}/{s.times} runs" if s.times else f"{s.ticks_fired} run{'s' if s.ticks_fired != 1 else ''}"
    if s.status == "paused":
        reason = f" · {s.paused_reason}" if s.paused_reason else ""
        return "⏸ loop paused", f"⏸ loop paused{reason} · {runs} · {s.prompt}", s.prompt
    if s.awaiting_response:
        when = "running now"
    else:
        remaining = s.next_due_at - (time.time() if now is None else now)
        when = f"next in {countdown(remaining)}" if remaining > 0 else "due, waiting for idle"
    return f"↻ loop {when}", f"↻ loop · {s.cadence_label()} · {runs} · {when} · {s.prompt}", s.prompt


def _prompt_text(item) -> str:
    if isinstance(item, tuple):  # (text, images) from an image-attached submit
        item = item[0]
    # TimelineNotification carries a compact title; voice/seeded sentinels wrap their text.
    text = getattr(item, "display_text", None) or getattr(item, "text", None) or str(item)
    return " ".join(str(text).split())


def queued_prompts(cli) -> list[str]:
    """One-line previews of the turns waiting in ``_pending_input``, oldest first.

    Automation payloads (goal continuations, /loop wakeups, background-process notices) are
    excluded: they are Hermes's own work, and counting them made a single typed message look
    like the tail of a ten-deep queue.
    """
    from hermes_cli.cli_queue_priority import is_automation_prompt
    from hermes_cli.gateway_tools_notify import gateway_store
    owned: list = []
    store = gateway_store(cli) if getattr(cli, "session_id", None) else None
    if store is not None:
        # The gateway's FIFO runs first (#134889 N5): its queued admissions as the owner last published them.
        from hermes_cli.gateway_queue import published_queue
        owned = published_queue(cli, store)
    pending = getattr(cli, "_pending_input", None)
    if pending is None:
        rows = owned
    else:
        with pending.mutex:
            rows = owned + list(pending.queue)
    return [text for text in map(_prompt_text, rows) if text and not is_automation_prompt(text)]
