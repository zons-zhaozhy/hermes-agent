"""Shared ``status_line`` rendering for goals.

The owner ``GoalManager`` (gateway or standalone CLI) and the viewer-side
``GatewayGoalManager`` mirror render the same lines; the barrier branch differs by process:
liveness probes belong to the barrier's owner, while a mirror trusts the mirrored fields.
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hermes_cli.goals import GoalState


def status_meta(s: GoalState) -> str:
    """The ``turns/subgoals/contract/gates`` fragment every status_line branch shares.

    Contract:
        Postconditions: returns a non-empty string starting ``"<used>/<max> turns"``.
    """
    turns = f"{s.turns_used}/{s.max_turns} turns"
    sub = (
        f", {len(s.subgoals)} subgoal{'s' if len(s.subgoals) != 1 else ''}"
        if s.subgoals
        else ""
    )
    con = ", contract" if s.has_contract() else ""
    gat = (
        f", {len(s.gates)} gate{'s' if len(s.gates) != 1 else ''}"
        if s.gates
        else ""
    )
    return f"{turns}{sub}{con}{gat}"


def active_status_line(s: GoalState, meta: str, *, remote_barriers: bool = False) -> str:
    """The active branch of ``status_line``. ``remote_barriers`` renders a session/pid barrier
    from the mirrored fields without a liveness probe: the barrier lives in the owner process's
    process_registry / pid namespace, and a viewer process probing its own (empty) registry
    repaints the owner's parked goal as active. The owner's lazy clear plus the mirror refresh
    carry the lifted state instead.

    Contract:
        Preconditions: ``s.status == "active"``.
        Postconditions: starts with ``"⏳ Goal (parked"`` while a barrier renders, else
            ``"⊙ Goal (active"``.
    """
    from hermes_cli.goals import _pid_alive, _session_waiting  # late: the owner's probe seams

    if s.waiting_on_session and (remote_barriers or _session_waiting(s.waiting_on_session)):
        return f"⏳ Goal (parked on {s.waiting_reason or f'session {s.waiting_on_session}'}, {meta}): {s.goal}"
    if s.waiting_on_pid and (remote_barriers or _pid_alive(s.waiting_on_pid)):
        return f"⏳ Goal (parked on {s.waiting_reason or f'pid {s.waiting_on_pid}'}, {meta}): {s.goal}"
    if s.waiting_until and time.time() < s.waiting_until:
        remaining = int(s.waiting_until - time.time())
        wr = s.waiting_reason or f"{remaining}s"
        return f"⏳ Goal (parked {remaining}s — {wr}, {meta}): {s.goal}"
    return f"⊙ Goal (active, {meta}): {s.goal}"
