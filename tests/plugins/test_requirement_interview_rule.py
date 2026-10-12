"""契约测试：开工前采访规则的不可破坏面。

规则文本是 session 级 system prompt 冻结段的一部分——改错不会报错，只会静默改变行为。
因此对三类硬约束加机器断言：finish_guard 依赖的字面前缀、每轮问数上限、分层拷问的三条
原则，外加子代理/批处理的跳过语义。
"""

from plugins.requirement_interview import _INTERVIEW_RULE, _interview_section


def test_rule_keeps_finish_guard_exemption_prefix() -> None:
    """finish_guard 按字面「必须先确认：」识别合法澄清提问；丢了它，提问会被拦成死锁。"""
    assert "必须先确认：" in _INTERVIEW_RULE  # 期望: 豁免前缀在位


def test_rule_keeps_per_round_question_bound() -> None:
    assert "3-6" in _INTERVIEW_RULE  # 期望: 每轮 3-6 问的上限仍在


def test_rule_states_the_three_layered_grilling_principles() -> None:
    assert "前置条件已定" in _INTERVIEW_RULE  # 期望: 一轮只问前置已定的问题
    assert "推荐答案" in _INTERVIEW_RULE  # 期望: 每题带推荐答案
    assert "子代理" in _INTERVIEW_RULE  # 期望: 事实类问题先派子代理查


def test_rule_ends_interview_only_when_the_frontier_is_empty() -> None:
    assert "前沿清空" in _INTERVIEW_RULE  # 期望: 结束条件锚在待决前沿为空,不是问满几轮


def test_subagent_and_batch_sessions_receive_no_rule() -> None:
    assert _interview_section({"platform": "subagent"}) == ""  # 期望: 子代理无 clarify 通道
    assert _interview_section({"platform": "batch"}) == ""  # 期望: 批处理同样跳过
    assert _interview_section({"platform": "cli"}) == _INTERVIEW_RULE  # 期望: 交互会话注入原文
