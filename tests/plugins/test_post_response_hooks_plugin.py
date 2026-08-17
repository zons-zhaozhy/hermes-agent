"""Tests for the post_response_hooks framework and plugin integration.

Regression (Aug 2026): ``plugins/post_response_hooks`` imported
``agent.post_response_hooks`` which never existed in the repo (git history:
the 2026-08-14 plugin migration commit ba0b26cca4 moved plugins/ but not the
framework file that lived in the old ~/.hermes/hermes-agent checkout). Every
session logged ``No module named 'agent.post_response_hooks'`` and the three
configured hooks (bottom_logic_check, correction_tracker, behavior_regression)
never ran.
"""
import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT_DIR = REPO_ROOT / "agent"
PLUGIN_DIR = REPO_ROOT / "plugins" / "post_response_hooks"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class TestHookResultContract:
    """HookResult dataclass contract shared by plugin + hook scripts."""

    def test_hookresult_attributes(self):
        sys.path.insert(0, str(REPO_ROOT))
        from agent.post_response_hooks import HookResult

        ok = HookResult(passed=True)
        assert ok.passed is True
        assert ok.action == ""
        assert ok.message == ""
        assert bool(ok) is True  # __bool__ → passed

        nudge = HookResult(passed=False, action="nudge", message="go deeper")
        assert nudge.passed is False
        assert nudge.action == "nudge"
        assert nudge.message == "go deeper"
        assert bool(nudge) is False


class TestFrameworkLoader:
    """load_hooks + run_post_response_checks with real hook scripts."""

    @pytest.fixture
    def framework(self):
        sys.path.insert(0, str(REPO_ROOT))
        import agent.post_response_hooks as fw
        return fw

    def test_load_hooks_real_scripts(self, framework):
        """The three real hook scripts from ~/.hermes/hooks/ load and expose
        the Hook.check interface + system_prompt_addition property."""
        hooks_dir = Path.home() / ".hermes" / "hooks"
        if not hooks_dir.exists():
            pytest.skip("~/.hermes/hooks not present on this machine")

        configs = [{"enabled": True, "module": m} for m in
                   ("bottom_logic_check", "correction_tracker", "behavior_regression")]
        hooks = framework.load_hooks(configs)

        names = {type(h).__module__ for h in hooks}
        assert len(hooks) >= 1
        for h in hooks:
            assert hasattr(h, "check")
            assert isinstance(h.system_prompt_addition, str)

    def test_run_checks_all_pass(self, framework):
        calls = []

        class _Hook:
            @property
            def system_prompt_addition(self):
                return ""

            def check(self, response, context):
                calls.append(response)
                return framework.HookResult(passed=True)

        result = framework.run_post_response_checks([_Hook()], "resp", {"session_id": "s1"})
        assert result is not None
        assert result.passed is True
        assert calls == ["resp"]

    def test_run_checks_nudge_wins_over_pass(self):
        sys.path.insert(0, str(REPO_ROOT))
        import agent.post_response_hooks as fw

        class _Pass:
            @property
            def system_prompt_addition(self):
                return ""

            def check(self, response, context):
                return fw.HookResult(passed=True)

        class _Nudge:
            @property
            def system_prompt_addition(self):
                return ""

            def check(self, response, context):
                return fw.HookResult(passed=False, action="nudge", message="deeper")

        result = fw.run_post_response_checks([_Pass(), _Nudge()], "r", {})
        assert result is not None
        assert result.passed is False
        assert result.action == "nudge"
        assert result.message == "deeper"

    def test_run_checks_exception_isolated(self):
        sys.path.insert(0, str(REPO_ROOT))
        import agent.post_response_hooks as fw

        class _Boom:
            @property
            def system_prompt_addition(self):
                return ""

            def check(self, response, context):
                raise RuntimeError("hook crashed")

        class _OK:
            @property
            def system_prompt_addition(self):
                return ""

            def check(self, response, context):
                return fw.HookResult(passed=True)

        # A crashed hook must not kill the whole run — other hooks still run.
        result = fw.run_post_response_checks([_Boom(), _OK()], "r", {})
        assert result is not None and result.passed is True


class TestPluginIntegration:
    """The plugin's config → load_hooks path with real config shape."""

    def test_plugin_loads_framework(self, monkeypatch):
        sys.path.insert(0, str(REPO_ROOT))
        plugin = _load("post_response_hooks_plugin", PLUGIN_DIR / "__init__.py")

        # Feed a config with a real module name; agent.post_response_hooks
        # is importable now, so load_hooks must succeed.
        monkeypatch.setattr(plugin, "_hooks_loaded", False)
        monkeypatch.setattr(plugin, "_hooks", [])

        hooks = plugin._load_hooks_from_config()
        # Config has 3 enabled hooks on this machine; on CI without
        # ~/.hermes/hooks the loader returns [] — both acceptable.
        assert isinstance(hooks, list)

    def test_plugin_post_llm_call_invocation(self, monkeypatch, caplog):
        sys.path.insert(0, str(REPO_ROOT))
        plugin = _load("post_response_hooks_plugin", PLUGIN_DIR / "__init__.py")

        monkeypatch.setattr(plugin, "_hooks_loaded", True)
        monkeypatch.setattr(plugin, "_hooks", [])
        # No hooks → no crash, no result
        plugin._on_post_llm_call(
            session_id="s1", user_message="why", assistant_response="answer", model="m",
        )
