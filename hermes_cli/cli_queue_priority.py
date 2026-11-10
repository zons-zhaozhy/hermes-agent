"""User input outranks queued automation prompts in the classic CLI's pending queue.

Goal continuations, /loop wakeups and background-process notices share ``_pending_input``
with human input. When the user interrupts a running turn (busy_input_mode=interrupt), the
interrupt handover re-queues their message at the tail — behind any automation prompts
earlier post-turn hooks already enqueued. This sibling classifies machine-generated payloads
and reorders the queue so human input runs first; the live-work dock reuses the same
predicate so its queued count reflects only user work.
"""

from __future__ import annotations

from typing import Any

# Prefixes of Hermes-generated user-role payloads, aligned with the machine-message list in
# plugins/read_think_gate and gateway/run_notifications' fold header
# ("[IMPORTANT: {n} background process results arrived ...").
_AUTOMATION_MARKERS = (
    "[Continuing toward your standing goal]",
    "[/loop wakeup #",
    "[IMPORTANT: ",
)


def is_automation_prompt(payload: Any) -> bool:
    """Whether a pending-queue payload is Hermes-generated, not human input."""
    if isinstance(payload, tuple):  # (text, images) bundles from an image-attached submit
        payload = payload[0] if payload else None
    text = getattr(payload, "display_text", None) or getattr(payload, "text", payload)
    if not isinstance(text, str) or not text:
        return False
    return any(text.startswith(marker) for marker in _AUTOMATION_MARKERS)


def promote_user_input_over_automation(pending: Any) -> None:
    """Reorder ``pending`` so queued human input precedes automation prompts.

    Each group keeps its internal order. Called on the consumer thread (inside the
    chat()/process_loop call stack) right after a real user message lands — never from the
    input thread, which only ever ``put()``s while the consumer is blocked in a turn.

    Contract:
      Preconditions: ``pending`` is a ``queue.Queue`` (or absent), drained only by its own
      consumer; callers hold no queue lock.
      Postconditions: every item is preserved exactly once; non-automation payloads precede
      automation payloads; within each group the original order is unchanged.
    """
    if pending is None or pending.empty():
        return
    with pending.mutex:
        items = list(pending.queue)
        reordered = ([i for i in items if not is_automation_prompt(i)]
                     + [i for i in items if is_automation_prompt(i)])
        if reordered == items:
            return
        # Directly on the deque: ``put()`` would re-acquire ``pending.mutex`` (a
        # non-reentrant lock) and deadlock the consumer.
        pending.queue.clear()
        pending.queue.extend(reordered)
        if reordered:  # wake one blocked consumer, not one per item
            pending.not_empty.notify()
