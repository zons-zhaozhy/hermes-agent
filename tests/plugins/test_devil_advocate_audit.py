"""devil-advocate-audit cron 死锁回归测试。

背景（2026-09-24 实录）：日学习 cron job（4162e5ea，无 delegate_task 工具、
无用户在场）被本插件 armed 后全工具冻结——正门（delegate_task）与豁免
（用户明示）两个出口在 cron 场景都不可达，构成无解死锁，整轮失败。

根治：platform=="cron" 时不 armed，降级为仅注入提醒（有 delegate 工具的
cron 会话仍被提醒引导自行反方审查）。

Contract:
  Preconditions: 以独立会话 state 运行，judge 全部 monkeypatch（零真实 LLM 调用）
  Postconditions: 全部断言通过 = cron 不死锁、交互会话拦截语义不变
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PLUGIN_DIR = _REPO_ROOT / "plugins" / "devil-advocate-audit"


def _load_plugin():
    """按 PluginManager 命名约定加载插件 __init__.py（相对导入可用）。"""
    if "hermes_plugins" not in sys.modules:
        ns = types.ModuleType("hermes_plugins")
        ns.__path__ = []
        sys.modules["hermes_plugins"] = ns
    spec = importlib.util.spec_from_file_location(
        "hermes_plugins.devil_advocate_audit",
        _PLUGIN_DIR / "__init__.py",
        submodule_search_locations=[str(_PLUGIN_DIR)],
    )
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "hermes_plugins.devil_advocate_audit"
    mod.__path__ = [str(_PLUGIN_DIR)]
    sys.modules["hermes_plugins.devil_advocate_audit"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def plugin(monkeypatch):
    """加载插件并 stub 掉所有 LLM judge 与 yinyang 合并通道（零网络）。"""
    mod = _load_plugin()
    monkeypatch.setattr(mod, "_is_major_decision", lambda text: True)
    monkeypatch.setattr(mod, "_user_waived", lambda text: False)
    monkeypatch.setattr(mod, "_delegate_is_review", lambda goals: True)
    monkeypatch.setattr(mod, "_judge_user_side", lambda message: None)
    # 重置模块级查找失败缓存，避免跨用例污染
    monkeypatch.setattr(mod, "_YINYANG_LOOKUP_FAILED", False)
    yield mod
    from plugins._shared_state import get_session_state

    state = get_session_state("cron-test-1", mod._NAMESPACE)
    state.clear()
    get_session_state("cron-test-2", mod._NAMESPACE).clear()
    get_session_state("cli-test-1", mod._NAMESPACE).clear()
    get_session_state("cli-test-2", mod._NAMESPACE).clear()
    get_session_state("cli-test-3", mod._NAMESPACE).clear()


class TestCronNoDeadlock:
    """cron 平台：judge 判定重大决策也不 armed、不冻结工具。"""

    def test_cron_platform_does_not_arm(self, plugin):
        sid = "cron-test-1"
        ret = plugin.on_pre_llm_call(
            session_id=sid, task_id="t1", user_message="决定上生产部署方案A",
            platform="cron",
        )
        from plugins._shared_state import get_session_state

        st = get_session_state(sid, plugin._NAMESPACE)
        # 期望: cron 不进入 armed 冻结态（正门/豁免双出口在无人场景不可达，
        # armed 即无解死锁——2026-09-24 日学习 job 整轮失败实录）
        assert not st.get("armed")

    def test_cron_platform_tool_not_blocked(self, plugin):
        sid = "cron-test-2"
        plugin.on_pre_llm_call(
            session_id=sid, task_id="t2", user_message="决定上生产部署方案A",
            platform="cron",
        )
        # 期望: cron 会话 terminal 工具不被本插件 block
        directive = plugin.on_pre_tool_call(
            session_id=sid, tool_name="terminal", args={"command": "date"},
        )
        assert directive is None

    def test_interactive_platform_still_arms(self, plugin):
        """交互平台（cli）拦截语义保持：重大决策仍 armed 冻结。"""
        sid = "cli-test-1"
        plugin.on_pre_llm_call(
            session_id=sid, task_id="t3", user_message="决定上生产部署方案A",
            platform="cli",
        )
        # 期望: armed 置位（红牌注入由 on_pre_llm_call 返回值承担）
        from plugins._shared_state import get_session_state

        assert get_session_state(sid, plugin._NAMESPACE).get("armed") is True
        directive = plugin.on_pre_tool_call(
            session_id=sid, tool_name="terminal", args={"command": "date"},
        )
        assert directive is not None and directive.get("action") == "block"


class TestDelegateGateUnchanged:
    """正门语义回归：delegate_task 始终放行；成功反方审查解除 armed。"""

    def test_delegate_tool_always_allowed(self, plugin):
        sid = "cli-test-2"
        plugin.on_pre_llm_call(
            session_id=sid, task_id="t4", user_message="决定上生产部署方案A",
            platform="cli",
        )
        directive = plugin.on_pre_tool_call(
            session_id=sid, tool_name="delegate_task",
            args={"tasks": [{"goal": "反方审查该方案，只找漏洞"}]},
        )
        assert directive is None

    def test_successful_review_disarms(self, plugin):
        sid = "cli-test-3"
        plugin.on_pre_llm_call(
            session_id=sid, task_id="t5", user_message="决定上生产部署方案A",
            platform="cli",
        )
        plugin.on_post_tool_call(
            session_id=sid, tool_name="delegate_task", status="ok",
            args={"goal": "反方审查该方案，只找漏洞"},
        )
        from plugins._shared_state import get_session_state

        assert get_session_state(sid, plugin._NAMESPACE).get("reviewed") is True
        directive = plugin.on_pre_tool_call(
            session_id=sid, tool_name="terminal", args={"command": "date"},
        )
        assert directive is None
