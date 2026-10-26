import importlib
import re
import sys
from unittest.mock import MagicMock, patch

_cli_mod = None

def _make_cli(user_message_preview=None):
    global _cli_mod
    clean_config = {
        "model": {
            "default": "anthropic/claude-opus-4.6",
            "base_url": "https://openrouter.ai/api/v1",
            "provider": "auto",
        },
        "display": {
            "compact": False,
            "tool_progress": "all",
            "user_message_preview": user_message_preview or {"first_lines": 2, "last_lines": 2},
        },
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
    with patch.dict(sys.modules, prompt_toolkit_stubs), patch.dict("os.environ", clean_env, clear=False):
        import cli as mod

        mod = importlib.reload(mod)
        _cli_mod = mod
        with patch.object(mod, "get_tool_definitions", return_value=[]), patch.dict(mod.__dict__, {"CLI_CONFIG": clean_config}):
            return mod.HermesCLI()

class TestSubmittedUserMessagePreview:
    def test_default_preview_shows_first_two_lines_and_last_two_lines(self):
        cli = _make_cli()

        rendered = cli._format_submitted_user_message_preview(
            "line1\nline2\nline3\nline4\nline5\nline6"
        )

        assert "line1" in rendered
        assert "line2" in rendered
        assert "line5" in rendered
        assert "line6" in rendered
        assert "line3" not in rendered
        assert "line4" not in rendered
        assert "(+2 more lines)" in rendered

    def test_preview_can_hide_last_lines(self):
        cli = _make_cli({"first_lines": 2, "last_lines": 0})

        rendered = cli._format_submitted_user_message_preview(
            "line1\nline2\nline3\nline4\nline5\nline6"
        )

        assert "line1" in rendered
        assert "line2" in rendered
        assert "line5" not in rendered
        assert "line6" not in rendered
        assert "(+4 more lines)" in rendered


class TestUserMessageHighlight:
    def test_highlight_color_marks_echoed_text_with_accent_fg(self):
        cli = _make_cli({"first_lines": 2, "last_lines": 2, "highlight": "color"})

        rendered = cli._format_submitted_user_message_preview(
            "line1\nline2\nline3\nline4\nline5\nline6"
        )

        # 期望: every echoed text line carries bold + an explicit hex fg (the skin accent)
        assert re.search(r"\[bold #[0-9A-Fa-f]{6}\]line1", rendered)
        assert re.search(r"\[bold #[0-9A-Fa-f]{6}\]line2", rendered)
        assert re.search(r"\[bold #[0-9A-Fa-f]{6}\]line5", rendered)
        assert re.search(r"\[bold #[0-9A-Fa-f]{6}\]line6", rendered)

    def test_highlight_none_keeps_plain_bold(self):
        cli = _make_cli({"first_lines": 2, "last_lines": 2, "highlight": "none"})

        rendered = cli._format_submitted_user_message_preview(
            "line1\nline2\nline3\nline4\nline5\nline6"
        )

        # 期望: echoed text is bold with no color/background span (the classic look);
        # the leading glyph keeps its accent color by design
        assert "[bold]line2" in rendered
        assert "[bold]line5" in rendered
        assert " on " not in rendered
        assert not re.search(r"\[bold #[0-9A-Fa-f]{6}\]line\d", rendered)

    def test_highlight_background_uses_on_marker(self):
        cli = _make_cli({"first_lines": 2, "last_lines": 2, "highlight": "background"})

        rendered = cli._format_submitted_user_message_preview(
            "line1\nline2\nline3\nline4\nline5\nline6"
        )

        # 期望: echoed text is black bold on an accent background span
        assert re.search(r"\[bold #000000 on #[0-9A-Fa-f]{6}\]line1", rendered)
        assert re.search(r"\[bold #000000 on #[0-9A-Fa-f]{6}\]line6", rendered)
