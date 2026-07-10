"""Regression tests for #33271: terminal recovery after interrupt.

When the user interrupts a running agent turn by typing a new message,
prompt_toolkit may have an in-flight ``CSI 6n`` cursor-position query whose
reply (``ESC[<row>;<col>R``) arrives on stdin after the input parser has torn
down. The reply then leaks as literal text (``^[[19;1R``) and the VT100 parser
can stall, accepting no further keystrokes — the terminal appears frozen.

The recovery path lives in ``HermesCLI._recover_terminal_after_interrupt()``,
which is invoked from ``process_loop``'s ``finally`` block only when
``self._last_turn_interrupted`` is set. It must:
  1. Drain stray escape bytes from the OS input buffer (``flush_stdin``).
  2. Clear the physical terminal via ``_clear_prompt_toolkit_screen()`` —
     the raw ``ESC[2J`` from ``erase_screen()`` is what unsticks the VT100
     parser; ``renderer.reset()`` alone does NOT send any escape sequence
     and cannot recover a stalled parser.
  3. Trigger an incremental redraw via ``app.invalidate()``.

Critical: the original implementation called ``_force_full_redraw()`` which
additionally replayed ``_OUTPUT_HISTORY``. The replay caused the interrupt
response Panel to be printed twice (once during the turn via ``_cprint``,
once during replay). The fix calls ``_clear_prompt_toolkit_screen()``
directly — screen erase + renderer reset, WITHOUT history replay.

These tests exercise the real method (not a re-implementation of its logic),
and assert that the finally block actually wires it in behind the interrupt
guard.
"""

import inspect
import re
from unittest.mock import MagicMock, patch

import pytest

from cli import HermesCLI


@pytest.fixture
def bare_cli():
    """A HermesCLI with no __init__ — we only exercise the recovery helper."""
    return object.__new__(HermesCLI)


class TestRecoverTerminalAfterInterrupt:
    """Directly exercise HermesCLI._recover_terminal_after_interrupt()."""

    def test_drains_stdin_then_clears_screen(self, bare_cli):
        """Happy path: flush_stdin runs, then screen cleared and app invalidated."""
        mock_app = MagicMock()
        bare_cli._app = mock_app
        bare_cli._clear_prompt_toolkit_screen = MagicMock()
        with patch("hermes_cli.curses_ui.flush_stdin") as mock_flush:
            bare_cli._recover_terminal_after_interrupt()

        mock_flush.assert_called_once()
        bare_cli._clear_prompt_toolkit_screen.assert_called_once_with(mock_app)
        mock_app.invalidate.assert_called_once()

    def test_screen_clear_still_runs_when_flush_fails(self, bare_cli):
        """A flush_stdin failure (no TTY, non-POSIX) must not skip screen clear.

        The two recovery steps are independent — losing the stdin drain must
        never leave the VT100 parser stalled.
        """
        mock_app = MagicMock()
        bare_cli._app = mock_app
        bare_cli._clear_prompt_toolkit_screen = MagicMock()
        with patch(
            "hermes_cli.curses_ui.flush_stdin", side_effect=OSError("no tty")
        ):
            bare_cli._recover_terminal_after_interrupt()  # must not raise

        bare_cli._clear_prompt_toolkit_screen.assert_called_once_with(mock_app)
        mock_app.invalidate.assert_called_once()

    def test_flush_runs_before_screen_clear(self, bare_cli):
        """Order matters: drain stray bytes first so they don't arrive mid-redraw."""
        events = []

        mock_app = MagicMock()
        bare_cli._app = mock_app
        bare_cli._clear_prompt_toolkit_screen = MagicMock(
            side_effect=lambda *a, **kw: events.append("clear")
        )
        mock_app.invalidate = MagicMock(side_effect=lambda: events.append("invalidate"))
        with patch(
            "hermes_cli.curses_ui.flush_stdin",
            side_effect=lambda: events.append("flush"),
        ):
            bare_cli._recover_terminal_after_interrupt()

        assert events == ["flush", "clear", "invalidate"]

    def test_no_crash_without_app(self, bare_cli):
        """When _app is not set (e.g. non-interactive mode), must not raise."""
        bare_cli._app = None
        with patch("hermes_cli.curses_ui.flush_stdin"):
            bare_cli._recover_terminal_after_interrupt()  # must not raise

    def test_no_crash_when_screen_clear_raises(self, bare_cli):
        """_clear_prompt_toolkit_screen exceptions are caught and silenced."""
        mock_app = MagicMock()
        bare_cli._app = mock_app
        bare_cli._clear_prompt_toolkit_screen = MagicMock(
            side_effect=RuntimeError("no screen")
        )
        with patch("hermes_cli.curses_ui.flush_stdin"):
            bare_cli._recover_terminal_after_interrupt()  # must not raise

    def test_no_force_full_redraw_called(self, bare_cli):
        """_force_full_redraw must NOT be called — it replays history, causing
        duplicate Panel output.  Only _clear_prompt_toolkit_screen (which skips
        replay) should be used.
        """
        bare_cli._force_full_redraw = MagicMock()
        bare_cli._clear_prompt_toolkit_screen = MagicMock()
        mock_app = MagicMock()
        bare_cli._app = mock_app
        with patch("hermes_cli.curses_ui.flush_stdin"):
            bare_cli._recover_terminal_after_interrupt()

        bare_cli._force_full_redraw.assert_not_called()

    def test_flush_stdin_is_tty_gated(self):
        """The real flush_stdin is a no-op on non-TTY stdin (piped/redirected).

        Under pytest stdin is not a TTY, so this must return cleanly without
        touching termios.
        """
        from hermes_cli.curses_ui import flush_stdin

        flush_stdin()  # must not raise in a non-TTY test environment


class TestFinallyBlockWiring:
    """The recovery helper is only useful if process_loop actually calls it.

    These guard against the helper silently becoming dead code (the fix being
    present but never invoked), which a unit test of the helper alone can't
    catch.
    """

    def test_recovery_is_invoked_behind_interrupt_guard(self):
        src = inspect.getsource(HermesCLI.run)
        # The recovery call must be gated on _last_turn_interrupted so it only
        # fires after an actual interrupt, not on every normal turn.
        guard = re.search(
            r"if self\._last_turn_interrupted:\s*\n\s*"
            r"self\._recover_terminal_after_interrupt\(\)",
            src,
        )
        assert guard, (
            "process_loop's finally block must call "
            "_recover_terminal_after_interrupt() guarded by "
            "self._last_turn_interrupted"
        )

    def test_recovery_helper_exists(self):
        assert hasattr(HermesCLI, "_recover_terminal_after_interrupt")
        assert callable(HermesCLI._recover_terminal_after_interrupt)
