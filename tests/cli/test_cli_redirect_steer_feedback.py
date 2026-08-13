"""Regression tests for CLI busy-path feedback when redirect() degrades to steer().

Background
----------
Classic CLI default ``busy_input_mode == "interrupt"`` routes user input
typed while the agent runs through ``agent.redirect(text)``.  But
``redirect()`` silently degrades to ``steer()`` while a tool is mid-flight
(run_agent.py redirect(), commit cbf5b05c70): the text lands in
``_pending_steer`` and rides on the LAST tool result once the current
command finishes — the turn is NOT actually interrupted.

The CLI used to print "↪ Redirected current turn" in that case, an illusion:
the command kept running and the message was only delivered after it
finished.  These tests exercise ``_redirect_was_steered()``, the detector
that lets handle_enter report the real state ("message queued mid-command,
current command continues; Ctrl+C interrupts for real") instead.
"""

from __future__ import annotations

import importlib
import sys
from unittest.mock import MagicMock, patch


def _make_cli():
    """Build a HermesCLI instance with prompt_toolkit stubbed out.

    Mirrors the helper in ``test_cli_steer_busy_path.py``.
    """
    _clean_config = {
        "model": {
            "default": "anthropic/claude-opus-4.6",
            "base_url": "https://openrouter.ai/api/v1",
            "provider": "auto",
        },
        "display": {"compact": False, "tool_progress": "all"},
        "agent": {},
        "terminal": {"env_type": "local"},
    }
    clean_env = {"LLM_MODEL": "", "HERMES_MAX_ITERATIONS": ""}
    prompt_toolkit_stubs = {
        "prompt_toolkit": MagicMock(),
        "prompt_toolkit.history": MagicMock(),
        "prompt_toolkit.styles": MagicMock(),
        "prompt_toolkit.patch_stdout": MagicMock(),
        "prompt_toolkit.application": MagicMock(),
        "prompt_toolkit.layout": MagicMock(),
        "prompt_toolkit.layout.processors": MagicMock(),
        "prompt_toolkit.filters": MagicMock(),
        "prompt_toolkit.layout.dimension": MagicMock(),
        "prompt_toolkit.layout.menus": MagicMock(),
        "prompt_toolkit.widgets": MagicMock(),
        "prompt_toolkit.key_binding": MagicMock(),
        "prompt_toolkit.completion": MagicMock(),
        "prompt_toolkit.formatted_text": MagicMock(),
        "prompt_toolkit.auto_suggest": MagicMock(),
    }
    with patch.dict(sys.modules, prompt_toolkit_stubs), patch.dict(
        "os.environ", clean_env, clear=False
    ):
        import cli as _cli_mod

        _cli_mod = importlib.reload(_cli_mod)
        with patch.object(_cli_mod, "get_tool_definitions", return_value=[]), patch.dict(
            _cli_mod.__dict__, {"CLI_CONFIG": _clean_config}
        ):
            return _cli_mod.HermesCLI()


class TestRedirectWasSteered:
    """``_redirect_was_steered`` distinguishes steer-degraded redirects."""

    def test_true_when_message_landed_in_pending_steer(self):
        cli = _make_cli()
        agent = MagicMock()
        agent._pending_steer = "please stop"
        agent._pending_redirect = None
        cli.agent = agent
        assert cli._redirect_was_steered() is True

    def test_false_when_redirect_really_happened(self):
        cli = _make_cli()
        agent = MagicMock()
        agent._pending_steer = None
        agent._pending_redirect = {"text": "please stop"}
        cli.agent = agent
        assert cli._redirect_was_steered() is False

    def test_false_when_no_pending_steer_at_all(self):
        cli = _make_cli()
        agent = MagicMock()
        agent._pending_steer = None
        agent._pending_redirect = None
        cli.agent = agent
        assert cli._redirect_was_steered() is False

    def test_false_when_agent_is_none(self):
        cli = _make_cli()
        cli.agent = None
        assert cli._redirect_was_steered() is False

    def test_false_when_agent_lacks_pending_attrs(self):
        """Older agents without the steer fields must not be misclassified."""
        cli = _make_cli()
        agent = MagicMock()
        del agent._pending_steer
        del agent._pending_redirect
        cli.agent = agent
        assert cli._redirect_was_steered() is False
