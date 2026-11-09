"""ReadThinkGate intent and investigation contracts.

The gate must answer two questions in order before a write:

1. Did the user ask for a modification, rather than an explanation/review?
2. Has the modification target and its surrounding behavior been investigated?

These tests derive the expected verdicts from that lifecycle. Reading a file,
creating a new path, or exhausting a round budget must not substitute for an
actual modification request and a complete investigation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from plugins.read_think_gate_host import ReadThinkGate, ReadThinkGateConfig


def _strict_gate(tmp_path: Path) -> ReadThinkGate:
    gate = ReadThinkGate(
        ReadThinkGateConfig.from_mapping(
            {
                "enabled": True,
                "complexity_adaptive": False,
                "use_llm_judge": False,
                "use_llm_intent_classifier": False,
                "min_read_only_calls": 2,
                "min_reasoning_chars": 200,
                "max_reasoning_rounds": 6,
            }
        ),
        session_id=f"intent-{tmp_path.name}",
    )
    return gate


def _intent_classifier_gate(tmp_path: Path) -> ReadThinkGate:
    gate = ReadThinkGate(
        ReadThinkGateConfig.from_mapping(
            {
                "enabled": True,
                "complexity_adaptive": False,
                "use_llm_judge": False,
                "use_llm_intent_classifier": True,
            }
        ),
        session_id=f"intent-llm-{tmp_path.name}",
    )
    return gate


def _record_complete_investigation(gate: ReadThinkGate, target: Path, *, content: str) -> None:
    gate._read_only_count = 3
    gate._investigation_evidence = [
        f"read_file: {target}",
        "search_files(callers): callers",
        "search_files(tests): tests",
        "terminal(readonly): git log -S target",
    ]
    gate._files_read = {str(target)}
    gate._four_axis_found = {"影响面", "原意图", "根因", "风险"}
    gate._satisfied = False
    gate._reasoning_rounds = 0


def test_read_only_request_blocks_write_even_after_complete_investigation(tmp_path: Path) -> None:
    target = tmp_path / "app.py"
    target.write_text("value = 1\n", encoding="utf-8")
    gate = _strict_gate(tmp_path)
    gate.reset_for_turn("帮我看看这个文件为什么返回错误，不要修改")
    _record_complete_investigation(gate, target, content="x" * 300)

    result = gate.check_batch(
        "已定位根因，并检查了调用方、测试、历史意图、影响面和风险。" * 10,
        ["patch"],
        [{"path": str(target), "old": "value = 1", "new": "value = 2"}],
    )

    assert result is not None
    assert "用户意图" in result
    assert "只读" in result


def test_ambiguous_request_blocks_write(tmp_path: Path) -> None:
    target = tmp_path / "app.py"
    target.write_text("value = 1\n", encoding="utf-8")
    gate = _strict_gate(tmp_path)
    gate.reset_for_turn("帮我处理一下这个")
    _record_complete_investigation(gate, target, content="x" * 300)

    result = gate.check_batch(
        "x" * 300,
        ["patch"],
        [{"path": str(target), "old": "value = 1", "new": "value = 2"}],
    )

    assert result is not None
    assert "用户意图" in result
    assert "不明确" in result


def test_explanation_and_modification_language_requires_disambiguation(
    tmp_path: Path,
) -> None:
    gate = _strict_gate(tmp_path)

    gate.reset_for_turn("怎么修复这个错误？")

    assert gate.user_intent == "ambiguous"


def test_modification_request_still_requires_full_investigation(tmp_path: Path) -> None:
    target = tmp_path / "app.py"
    target.write_text("value = 1\n", encoding="utf-8")
    gate = _strict_gate(tmp_path)
    gate.reset_for_turn("修一下这个文件的错误")

    # The target was read, but nothing establishes callers, tests, current
    # behavior, acceptance criteria, or historical intent.
    gate._read_only_count = 1
    gate._files_read = {str(target)}
    gate._investigation_evidence = [f"read_file: {target}"]

    result = gate.check_batch(
        "我准备改成 value = 2。",
        ["patch"],
        [{"path": str(target), "old": "value = 1", "new": "value = 2"}],
    )

    assert result is not None
    assert "调查" in result
    assert "调用方" in result or "搜索" in result


def test_new_file_does_not_bypass_intent_or_investigation(tmp_path: Path) -> None:
    target = tmp_path / "new_module.py"
    gate = _strict_gate(tmp_path)
    gate.reset_for_turn("解释一下这个项目应该怎么组织，不要创建文件")

    result = gate.check_batch(
        "x" * 300,
        ["write_file"],
        [{"path": str(target), "content": "x = 1\n"}],
    )

    assert result is not None
    assert "用户意图" in result


def test_modification_with_complete_evidence_unlocks_write(tmp_path: Path) -> None:
    target = tmp_path / "app.py"
    target.write_text("value = 1\n", encoding="utf-8")
    gate = _strict_gate(tmp_path)
    gate.reset_for_turn("修复这个文件里的错误，并补测试")
    _record_complete_investigation(gate, target, content="x" * 300)

    result = gate.check_batch(
        "当前行为：value=1 导致失败。调用方已检查，测试已定位，git log 原意图明确。"
        "调用方覆盖 app.main 与 app.worker，现有测试覆盖成功和失败路径。"
        "项目配置要求保持公开 API 不变，兄弟实现复用同一个校验器。"
        "影响面、原意图、根因、风险均已核实；最坏风险是旧调用方收到不同返回值，"
        "但可通过兼容测试发现并恢复。验收标准是 focused test 与相关回归全部通过；"
        "验证命令为 scripts/run_tests.sh tests/agent/test_app.py。",
        ["patch"],
        [{"path": str(target), "old": "value = 1", "new": "value = 2"}],
    )

    assert result is None
    assert gate.is_satisfied


def test_default_intent_guard_does_not_add_a_model_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import plugins.read_think_gate_host as host

    calls: list[str] = []
    monkeypatch.setattr(
        host,
        "_classify_intent_via_llm",
        lambda message, history=None: calls.append(message),
    )
    gate = ReadThinkGate(
        ReadThinkGateConfig.from_mapping({"use_llm_intent_classifier": False}),
        session_id=f"no-cost-{tmp_path.name}",
    )
    gate.reset_for_turn("修复这个文件里的错误")

    assert calls == []
    assert gate.user_intent == "modify"


def test_default_intent_classifier_handles_ambiguous_intent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import plugins.read_think_gate_host as host

    calls: list[str] = []
    monkeypatch.setattr(
        host,
        "_classify_intent_via_llm",
        lambda message, history=None: calls.append(message) or host.INTENT_MODIFY,
    )
    gate = ReadThinkGate(ReadThinkGateConfig(), session_id=f"semantic-{tmp_path.name}")
    gate.reset_for_turn("帮我处理一下这个")

    assert calls == ["帮我处理一下这个"]
    assert gate.user_intent == "modify"


def test_terse_confirmation_authorizes_the_proposed_modification(tmp_path: Path) -> None:
    history = [
        {
            "role": "assistant",
            "content": "方案：先补失败的回归测试，再修复解析器并运行相关测试。",
        }
    ]
    gate = _strict_gate(tmp_path)

    gate.reset_for_turn("同意，按这个做", history)

    assert gate.user_intent == "modify"


def test_exact_confirmation_authorizes_the_proposed_modification(tmp_path: Path) -> None:
    history = [
        {
            "role": "assistant",
            "content": "方案：先补失败的回归测试，再修复解析器并运行相关测试。确认执行请回复：确认",
        }
    ]
    gate = _strict_gate(tmp_path)

    gate.reset_for_turn("确认", history)

    assert gate.user_intent == "modify"


@pytest.mark.parametrize(
    "message",
    ("确认执行", "确认，执行", "同意执行", "批准执行", "确认，按这个方案执行"),
)
def test_execution_confirmation_authorizes_the_proposed_modification(
    tmp_path: Path,
    message: str,
) -> None:
    history = [
        {
            "role": "assistant",
            "content": "方案：先补失败的回归测试，再修复解析器并运行相关测试。确认执行请回复：确认执行",
        }
    ]
    gate = _strict_gate(tmp_path)

    gate.reset_for_turn(message, history)

    assert gate.user_intent == "modify"


def test_bare_execution_word_authorizes_proposed_modification(
    tmp_path: Path,
) -> None:
    """裸「执行」单独成句、承接具体修改方案 = 修改授权。

    回归锚：2026-11-09 实测用户原话「执行」被判 read_only——词表只有
    「执行吧/确认执行」，裸二字漏网，三连误拦。
    """
    history = [
        {
            "role": "assistant",
            "content": "方案：修复注册跳转与校验、删除内部信息暴露，共三处。确认请回复：执行",
        }
    ]
    gate = _strict_gate(tmp_path)

    gate.reset_for_turn("执行", history)

    assert gate.user_intent == "modify"


def test_execution_sentence_with_trailing_question_still_authorizes(
    tmp_path: Path,
) -> None:
    """授权句 + 追问句混合消息：句级判定，问号不整体否决授权。

    回归锚：2026-11-09 实测「执行\n\n门禁拦截是不是还有问题？到底咋回事？」
    被整条消息级「含？即否决」规则判 read_only。
    """
    history = [
        {
            "role": "assistant",
            "content": "方案：修复注册跳转与校验、删除内部信息暴露，共三处。确认请回复：执行",
        }
    ]
    gate = _strict_gate(tmp_path)

    gate.reset_for_turn("执行\n\n门禁拦截是不是还有问题？到底咋回事？", history)

    assert gate.user_intent == "modify"


def test_question_alone_stays_read_only(tmp_path: Path) -> None:
    history = [
        {
            "role": "assistant",
            "content": "方案：先补失败的回归测试，再修复解析器并运行相关测试。确认执行请回复：确认执行",
        }
    ]
    gate = _strict_gate(tmp_path)

    gate.reset_for_turn("门禁拦截是不是还有问题？到底咋回事？", history)

    # 纯问句不得授权写操作（判定为 read_only 或 ambiguous 均阻断，皆可接受）
    assert gate.user_intent != "modify"


@pytest.mark.parametrize(
    "message",
    ("执行吗？", "好吗？", "不行吗？", "不执行吗？"),
)
def test_interrogative_or_negated_confirmation_does_not_authorize(
    tmp_path: Path,
    message: str,
) -> None:
    """疑问式确认和否定式确认不得因为句内含有「执行/好/行」而放行写操作。"""
    history = [
        {
            "role": "assistant",
            "content": "方案：修复注册跳转与校验、删除内部信息暴露，共三处。确认请回复：执行",
        }
    ]
    gate = _strict_gate(tmp_path)

    gate.reset_for_turn(message, history)

    assert gate.user_intent != "modify"


def test_confirmation_check_is_hard_read_only_intent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import plugins.read_think_gate_host as host

    history = [
        {
            "role": "assistant",
            "content": "方案：先补失败的回归测试，再修复解析器并运行相关测试。",
        }
    ]
    calls: list[str] = []
    monkeypatch.setattr(
        host,
        "_classify_intent_via_llm",
        lambda message, history=None: calls.append(message) or host.INTENT_MODIFY,
    )
    gate = _intent_classifier_gate(tmp_path)

    gate.reset_for_turn("确认一下", history)

    assert gate.user_intent == "read_only"
    assert calls == []


def test_deterministic_intents_do_not_call_or_yield_to_semantic_classifier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import plugins.read_think_gate_host as host

    calls: list[str] = []

    def classify(message, history=None):
        calls.append(message)
        return host.INTENT_MODIFY

    monkeypatch.setattr(host, "_classify_intent_via_llm", classify)
    history = [
        {
            "role": "assistant",
            "content": "方案：修复解析器并运行测试。确认执行请回复：确认执行",
        }
    ]
    gate = _intent_classifier_gate(tmp_path)

    gate.reset_for_turn("不要修改", history)
    assert gate.user_intent == "read_only"
    gate.reset_for_turn("确认执行", history)
    assert gate.user_intent == "modify"

    assert calls == []


def test_semantic_classifier_resolves_ambiguous_intent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import plugins.read_think_gate_host as host

    calls: list[str] = []

    def classify(message, history=None):
        calls.append(message)
        return host.INTENT_MODIFY

    monkeypatch.setattr(host, "_classify_intent_via_llm", classify)
    gate = _intent_classifier_gate(tmp_path)

    gate.reset_for_turn("帮我处理一下这个")

    assert calls == ["帮我处理一下这个"]
    assert gate.user_intent == "modify"


def test_semantic_classifier_failure_falls_back_conservatively(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import plugins.read_think_gate_host as host

    monkeypatch.setattr(host, "_classify_intent_via_llm", lambda message, history=None: None)
    gate = _intent_classifier_gate(tmp_path)

    gate.reset_for_turn("帮我处理一下这个")

    assert gate.user_intent == "ambiguous"


def test_confirmation_cannot_authorize_without_a_proposal(tmp_path: Path) -> None:
    gate = _strict_gate(tmp_path)

    gate.reset_for_turn("同意", [{"role": "assistant", "content": "根因已经定位。"}])

    assert gate.user_intent == "ambiguous"


def test_repeated_blocks_hand_control_back_instead_of_implicitly_allowing(
    tmp_path: Path,
) -> None:
    gate = _strict_gate(tmp_path)
    gate.reset_for_turn("修复这个问题")
    gate._reasoning_rounds = gate._active_profile.max_reasoning_rounds

    result = gate.check_batch(
        "仍在调查。",
        ["write_file"],
        [{"path": str(tmp_path / "new_module.py"), "content": "x = 1\n"}],
    )

    assert result is not None
    assert "停止自动重试" in result
    assert not gate.is_satisfied
