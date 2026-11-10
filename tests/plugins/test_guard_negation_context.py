"""守卫负例面：否定与引用语境不得当成违规断言（2026-11-10 CLI 事故回归）。

事故形态：用户问「高危漏洞都处理好了吗」，助手答「这次没有绕过这个问题，
直接修了根因」，no-bypass 把「绕过这个问题」当违规命中，往用户正文追加
「[NO-BYPASS] 你的回复包含绕过模式……违反零妥协原则」——用户看到的就是
「不说人话」。同族问题在 finish_guard 根因门禁上已修（见
test_finish_guard_rootcause_gate.py R5）。

契约：命中词前后处于否定（未/没有/不存在/禁止/不允许/避免）或引用
（「」/宣称）语境时，是边界陈述而非违规断言，必须放行。
"""
from __future__ import annotations

from plugins.discipline import no_bypass, no_pushback

_NEGATED_BYPASS_TEXTS = (
    "这次没有绕过这个问题，直接修了根因。",
    "不存在「后面再修」这种处理方式。",
    "规则禁止绕过这个限制。",
    "绕过这个问题的做法被我们否掉了。",
    "不允许「后面再修」。",
)

_REAL_BYPASS_TEXTS = (
    "可以先绕过这个问题，后面再修。",
    "这个 bug 先不修这个，用 workaround 顶上。",
)


def test_negated_bypass_mention_not_appended():
    """否定/引用语境不追加 [NO-BYPASS] 纠正指令。"""
    for text in _NEGATED_BYPASS_TEXTS:
        out = no_bypass._on_transform_llm_output(response_text=text)
        assert out == "", text  # 期望: 不改写用户答案
        blocked = no_bypass._on_pre_tool_call(
            tool_name="send_message", args={"message": text})
        assert blocked == {}, text  # 期望: 不拦发消息


def test_real_bypass_still_caught():
    """真绕过表述仍然拦，否定豁免不得把判定整条放过。"""
    for text in _REAL_BYPASS_TEXTS:
        out = no_bypass._on_transform_llm_output(response_text=text)
        assert "[NO-BYPASS]" in out, text  # 期望: 真违规仍纠正


def test_quoted_pushback_question_not_appended():
    """引用/否定语境里的推责问句不是推责。"""
    text = "我不会说「要不要我继续？」这种话，直接做完。"
    assert no_pushback._on_transform_llm_output(response_text=text) == ""  # 期望: 不改写


def test_real_pushback_still_caught():
    """真推责问句仍然纠正。"""
    text = "改动已完成，要不要我继续处理下一批？"
    out = no_pushback._on_transform_llm_output(response_text=text)
    assert "[NO-PUSHBACK]" in out  # 期望: 仍纠正
