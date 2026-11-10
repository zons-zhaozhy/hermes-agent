"""Interrupt-demotion policy for a busy session's follow-up.

Two protections turn ``busy_input_mode: interrupt`` into a queued follow-up:

* ``#30170`` — the running agent owns active ``delegate_task`` children; ``interrupt()``
  cascades through ``_active_children`` and aborts in-flight subagent work.
* ``#56391`` — context compression is rotating the session id; an interrupt would start a
  new turn on the pre-rotation parent and fork orphaned siblings.

The subagent protection is a cost trade rather than a correctness requirement, so it has an
opt-in escape (``display.busy_interrupt_overrides_subagents``, default OFF). The compression
protection is a correctness requirement and stays unconditional.

Lives in its own module: ``run_busy`` and ``run_inbound`` both need the same decision, and
duplicating the rule is how the two paths drift apart.
"""

from __future__ import annotations

from typing import Any, Optional

# Reasons double as the operator-facing log/ack fragment, mirroring the issue each protects.
SUBAGENT_DEMOTION_REASON = "because the running agent has active subagents (#30170)"
COMPRESSION_DEMOTION_REASON = "because context compression is in flight (#56391)"

_TRUTHY = frozenset({"true", "1", "yes", "on"})


def parse_busy_interrupt_overrides_subagents(raw: Any) -> bool:
    """Parse ``display.busy_interrupt_overrides_subagents``; anything unrecognized is OFF.

    Default OFF keeps #30170's protection: an unexpected value must never silently start
    killing delegated work.

    Contract:
      Preconditions: ``raw`` is the config value, a bool, or a string (None when unset).
      Postconditions: returns a bool; only an explicit truthy token enables it.
    """
    if isinstance(raw, bool):
        return raw
    return str(raw or "").strip().lower() in _TRUTHY


def subagent_demotion_reason(
    runner: Any, *, has_active_subagents: bool, source: Any = None
) -> Optional[str]:
    """Reason string when this follow-up must be demoted for active subagents, else ``None``.

    The opt-in flag is resolved through the runner's profile-aware accessor when present, so a
    multiplexed secondary profile uses its own ``display`` section rather than the launch
    profile's.

    Contract:
      Preconditions: ``has_active_subagents`` is the caller's own detection result.
      Postconditions: ``None`` exactly when no subagents are active or the opt-in is on;
      otherwise the fixed #30170 reason.
    """
    if not has_active_subagents:
        return None
    resolve = getattr(runner, "_effective_busy_interrupt_overrides_subagents", None)
    if callable(resolve):
        if resolve(source):
            return None
    elif getattr(runner, "_busy_interrupt_overrides_subagents", False):
        return None
    return SUBAGENT_DEMOTION_REASON


def compression_demotion_reason(*, compression_in_flight: bool) -> Optional[str]:
    """Reason string when this follow-up must be demoted for in-flight compression, else ``None``.

    Not configurable on purpose (#56391): the session id rotates under the turn, so an
    interrupt strands the sibling sessions the rotation just created.
    """
    return COMPRESSION_DEMOTION_REASON if compression_in_flight else None
