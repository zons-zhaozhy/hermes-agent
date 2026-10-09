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


class TestJudgeOutageNoDeadlock:
    """判定通道挂掉（本地判定模型超时/无兜底）时不得死锁。

    背景（2026-10-09 实录）：auxiliary.devil_advocate_delegate 指向本地
    127.0.0.1:11434 且无 fallback_chain，四次委派全部 20s 超时 →
    llm_judge_bool 返回 None → _delegate_is_review=False → reviewed 永不
    写入 → armed 冻结全部非 delegate 工具；同源配置的 devil_advocate_waive
    一并超时，用户豁免出口同样失效 = 双出口死锁。
    """

    def test_marker_delegate_disarms_when_judge_fails(self, plugin, monkeypatch):
        sid = "cli-test-4"
        # 判定通道故障：judge 返回 None（超时 fail-open 的真实形状）
        monkeypatch.setattr(plugin, "_delegate_is_review", lambda goals: None)
        plugin.on_pre_llm_call(
            session_id=sid, task_id="t6", user_message="决定上生产部署方案A",
            platform="cli",
        )
        plugin.on_post_tool_call(
            session_id=sid, tool_name="delegate_task", status="ok",
            args={"goal": "反方审查该方案，只找漏洞"},
        )
        from plugins._shared_state import get_session_state

        st = get_session_state(sid, plugin._NAMESPACE)
        # 期望: 门禁公布的审查语义标记（_BLOCK_MSG_TEMPLATE 明示）足以解锁——
        # 判定通道挂掉不等于死锁
        assert st.get("reviewed") is True
        st.clear()

    def test_waive_phrase_disarms_when_judge_fails(self, plugin, monkeypatch):
        sid = "cli-test-5"
        monkeypatch.setattr(plugin, "_is_major_decision", lambda text: True)
        monkeypatch.setattr(plugin, "_user_waived", lambda text: None)
        plugin.on_pre_llm_call(
            session_id=sid, task_id="t7", user_message="豁免反方审查",
            platform="cli",
        )
        from plugins._shared_state import get_session_state

        st = get_session_state(sid, plugin._NAMESPACE)
        # 期望: 门禁公布给用户的口令（"说'豁免反方审查'即可"）字面命中即豁免，
        # 不依赖可能挂掉的判定通道
        assert st.get("waived") is True
        assert not st.get("armed")
        directive = plugin.on_pre_tool_call(
            session_id=sid, tool_name="terminal", args={"command": "date"},
        )
        assert directive is None
        st.clear()


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


class TestJudgeCallBudget:
    """judge 调用必须显式带窗口，并落在框架有界钩子预算内。

    背景：三处 judge 调用点分属 post_tool_call / pre_llm_call，两者都在
    hermes_cli/plugins_dispatch.py 的 _HOOK_TIMEOUT_BOUNDED_HOOKS 内，预算
    _HOOK_CALLBACK_TIMEOUT_SECS=30s，超时即 abandon（非 fail-closed 钩子直接
    skip）。llm_judge_bool 的 timeout 形参默认 20.0 且原样传给 call_llm，
    auxiliary.<task>.timeout（config.yaml）在这条路径不生效；judge 默认 20s
    加回退主模型的耗时远超预算 → 钩子被丢弃 → 状态永不写入 = 审查门死锁。
    """

    def test_every_judge_call_passes_bounded_timeout(self, monkeypatch):
        mod = _load_plugin()  # 不复用 plugin fixture：它把被测函数本身 stub 掉了
        from hermes_cli.plugins_dispatch import _HOOK_CALLBACK_TIMEOUT_SECS

        seen: list[float | None] = []

        def _recorder(task: str, **kwargs) -> None:
            seen.append(kwargs.get("timeout"))

        monkeypatch.setattr(mod, "llm_judge_bool", _recorder)
        mod._is_major_decision("决定上生产部署方案A")
        mod._delegate_is_review("帮我查一下这家公司的资料")
        mod._user_waived("这个方案不用再过审了")
        # 期望: 三处 judge 调用全部显式传 timeout；最坏三次尝试（主 + fallback_chain
        # + 主模型回退）累计 ≤ 钩子预算一半，留半量余量
        assert len(seen) == 3, f"judge 调用点数量变了: {seen}"
        assert all(
            t is not None and 0 < t <= _HOOK_CALLBACK_TIMEOUT_SECS / 2 for t in seen
        ), seen
