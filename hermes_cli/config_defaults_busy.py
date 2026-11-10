"""Pure-data factory for the busy-input policy defaults under ``display``.

Kept out of ``config_defaults.py`` (over its line cap on main): the comment that explains
each knob is longer than the key itself, and the busy policy is one coherent topic.
"""


def _busy_input_defaults() -> dict[str, object]:
    """``display`` keys governing what a follow-up does while a turn is running."""
    return {
        "busy_input_mode": "interrupt",  # interrupt | queue | steer
        # interrupt mode only: fire the interrupt even while the running agent drives
        # delegate_task children (default off keeps #30170's protection). On, the interrupted
        # turn's delegated work is redone on the next turn.
        "busy_interrupt_overrides_subagents": False,
        # steer mode: false hides only the "Steered into current run" bubble; steering itself still
        # happens.
        "busy_steer_ack_enabled": True,
    }
