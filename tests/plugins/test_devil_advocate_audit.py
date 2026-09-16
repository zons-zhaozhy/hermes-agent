"""Tests for devil-advocate-audit plugin (LLM judge 版).

行为契约（按"每个重大决策未经反方审查都该被拦"语义独立推导）：
- judge 判 True 且未审查 → 注入红牌
- judge 判 False → None
- judge 抛异常（fail-open）→ None
- 同一消息 hash 去重 → 第二次不调 judge
- judge_calls 超 30 → 不再调 judge（成本阀，非提醒上限）
- 红牌无每会话次数上限：第 N 次重大决策仍触发
- delegate 委派判定为反方审查类且 success → reviewed=True 静默
- delegate 委派判定为非审查类（只读/跑腿）→ 不置 reviewed，红牌照发
- delegate 委派 status!=success → 不置 reviewed
- delegate judge 失败（fail-open）→ 不置 reviewed
- env 开关 → None
- 钩子异常 → None（fail-open）
"""
import importlib
from typing import Any, Dict, Generator, List

import pytest

from plugins._shared_state import clear_session

SID = "s1"


@pytest.fixture(autouse=True)
def _clean_state() -> Generator[None, None, None]:
    clear_session(SID)
    yield
    clear_session(SID)


@pytest.fixture
def plugin() -> Any:
    return importlib.import_module("plugins.devil-advocate-audit")


def _msg(text: str) -> Dict[str, str]:
    return {"session_id": SID, "user_message": text}


def _delegate(goal: str, status: str = "success") -> Dict[str, Any]:
    return {
        "session_id": SID,
        "tool_name": "delegate_task",
        "args": {"tasks": [{"goal": goal}]},
        "result": {},
        "status": status,
    }


def _resp(content: str) -> Any:
    m = type("M", (), {"content": content})()
    c = type("C", (), {"message": m})()
    return type("R", (), {"choices": [c]})()


def _resp_by_key(review: bool) -> Any:
    return _resp('{"decision": true, "review": %s}' % ("true" if review else "false"))


@pytest.fixture
def mock_judge(monkeypatch: pytest.MonkeyPatch) -> List[Any]:
    calls: List[Any] = []

    def fake_call_llm(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs.get("messages"))
        return _resp_by_key(True)

    monkeypatch.setattr(
        "agent.auxiliary_client.call_llm", fake_call_llm)
    return calls


def test_judge_true_triggers(plugin, mock_judge):
    assert mock_judge == []  # judge 尚未被调
    out = plugin.on_pre_llm_call(**_msg("我决定整个系统迁移到新的存储架构"))
    assert out is not None and "反方审查" in out["context"]
    assert len(mock_judge) == 2  # 决策判定 + 豁免判定


def test_judge_false_no_trigger(plugin, monkeypatch):
    monkeypatch.setattr(
        "agent.auxiliary_client.call_llm",
        lambda *a, **k: _resp('{"decision": false}'))
    assert plugin.on_pre_llm_call(**_msg("随便聊聊今天天气")) is None


def test_judge_exception_fail_open(plugin, monkeypatch):
    def boom(*a: Any, **k: Any) -> Any:
        raise RuntimeError("no aux client")
    monkeypatch.setattr(
        "agent.auxiliary_client.call_llm", boom)
    assert plugin.on_pre_llm_call(**_msg("决定采用新方案")) is None


def test_message_dedup(plugin, mock_judge):
    plugin.on_pre_llm_call(**_msg("方案A定稿，全面切换"))
    plugin.on_pre_llm_call(**_msg("方案A定稿，全面切换"))
    assert len(mock_judge) == 2  # 首次=决策+豁免两判；第二次命中 hash 去重零调用


def test_judge_call_cap(plugin, mock_judge):
    for i in range(35):
        plugin.on_pre_llm_call(**_msg(f"消息 {i}"))
    # 30 次上限后不再调
    assert len(mock_judge) <= 30


def test_reminder_no_session_cap(plugin, mock_judge):
    """红牌无每会话上限：第 3、4 次重大决策仍须触发（防长会话尾部免检）。"""
    assert plugin.on_pre_llm_call(**_msg("决策一：迁移存储")) is not None
    assert plugin.on_pre_llm_call(**_msg("决策二：替换模型")) is not None
    assert plugin.on_pre_llm_call(**_msg("决策三：上生产部署")) is not None
    assert plugin.on_pre_llm_call(**_msg("决策四：全面重构")) is not None


def test_review_delegate_silences(plugin, monkeypatch):
    monkeypatch.setattr(
        "agent.auxiliary_client.call_llm",
        lambda *a, **k: _resp_by_key(True))
    plugin.on_post_tool_call(**_delegate(
        "只找漏洞地审查这个方案：列出所有设计缺陷与风险"))
    assert plugin.on_pre_llm_call(**_msg("任何消息")) is None


def test_non_review_delegate_does_not_silence(plugin, monkeypatch):
    """只读查询类委派不免检——否则一次普通委派短路整个机制。"""
    monkeypatch.setattr(
        "agent.auxiliary_client.call_llm",
        lambda *a, **k: _resp_by_key(False))
    plugin.on_post_tool_call(**_delegate("读取日志文件并统计错误聚类"))
    out = plugin.on_pre_llm_call(**_msg("我决定整个系统切换新架构"))
    assert out is not None and "反方审查" in out["context"]


def test_failed_delegate_does_not_silence(plugin, monkeypatch):
    monkeypatch.setattr(
        "agent.auxiliary_client.call_llm",
        lambda *a, **k: _resp_by_key(True))
    plugin.on_post_tool_call(**_delegate(
        "只找漏洞地审查这个方案", status="error"))
    out = plugin.on_pre_llm_call(**_msg("我决定整个系统切换新架构"))
    assert out is not None  # 委派失败≠已审查


def test_delegate_judge_fail_open(plugin, monkeypatch):
    def boom(*a: Any, **k: Any) -> Any:
        raise RuntimeError("no aux client")
    monkeypatch.setattr(
        "agent.auxiliary_client.call_llm", boom)
    plugin.on_post_tool_call(**_delegate("只找漏洞地审查这个方案"))
    # judge 失败 fail-open：不置 reviewed；pre_llm 同样 fail-open 返回 None
    assert plugin.on_pre_llm_call(**_msg("方案定稿")) is None


def test_env_disable(monkeypatch, plugin):
    monkeypatch.setenv("DEVIL_ADVOCATE_AUDIT_DISABLE", "1")
    assert plugin.on_pre_llm_call(**_msg("重大决策")) is None


def test_fail_open_on_bad_state(plugin, monkeypatch):
    monkeypatch.setattr(plugin, "_count",
                        lambda sid, key="count": (_ for _ in ()).throw(
                            RuntimeError("x")))
    assert plugin.on_pre_llm_call(**_msg("方案定稿")) is None


# ── 动作拦截（armed → block 非 delegate 工具）──────────────────────

def _tool(tool_name: str, sid: str = SID) -> Dict[str, Any]:
    return {"session_id": sid, "tool_name": tool_name, "args": {}}


@pytest.fixture
def armed(plugin: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """让会话进入 armed 状态：重大决策 + 未豁免（waive 判 False）。"""
    monkeypatch.setattr(
        "agent.auxiliary_client.call_llm",
        lambda *a, **k: _resp('{"decision": true, "waive": false}'))
    out = plugin.on_pre_llm_call(**_msg("我决定整个系统迁移到新存储架构"))
    assert out is not None  # 前置：红牌已注入=armed 已置位


def test_armed_blocks_non_delegate_tool(plugin: Any, armed: None) -> None:
    out = plugin.on_pre_tool_call(**_tool("terminal"))
    assert out is not None and out["action"] == "block"
    assert "terminal" in out["message"] and "反方审查" in out["message"]


def test_armed_allows_delegate_gate(plugin: Any, armed: None) -> None:
    assert plugin.on_pre_tool_call(**_tool("delegate_task")) is None


def test_reviewed_disarms_block(plugin: Any, armed: None,
                                monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "agent.auxiliary_client.call_llm",
        lambda *a, **k: _resp_by_key(True))
    plugin.on_post_tool_call(**_delegate(
        "只找漏洞地审查这个方案：列出所有设计缺陷与风险"))
    assert plugin.on_pre_tool_call(**_tool("terminal")) is None


def test_waived_disarms_block(plugin: Any,
                              monkeypatch: pytest.MonkeyPatch) -> None:
    # 同一条消息既判为重大决策又判为豁免 → waived 置位不武装
    monkeypatch.setattr(
        "agent.auxiliary_client.call_llm",
        lambda *a, **k: _resp('{"decision": true, "waive": true}'))
    out = plugin.on_pre_llm_call(**_msg("方案定了，我拍板豁免反方审查"))
    assert out is None  # 不注入红牌
    assert plugin.on_pre_tool_call(**_tool("terminal")) is None


def test_not_armed_allows_all(plugin: Any, mock_judge: List[Any]) -> None:
    # 未判为重大决策的会话：工具一律放行
    assert plugin.on_pre_tool_call(**_tool("terminal")) is None


def test_pre_tool_call_fail_open(plugin: Any, armed: None,
                                 monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(sid: str, namespace: str = "") -> Any:
        raise RuntimeError("state store gone")
    monkeypatch.setattr(plugin, "get_session_state", boom)
    assert plugin.on_pre_tool_call(**_tool("terminal")) is None
