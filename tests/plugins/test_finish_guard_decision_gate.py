"""finish_guard v2.3 拍板请求自包含（bare decision ask）回归测试。

背景（2026-10-26 用户点名）：收轮消息说「两方案你拍板后实施」「推荐方案 A，
零代码改动」，但选项内容只定义在早前轮次（且可能已被上下文压缩卷走）——
用户只看这一条消息无从拍板。根因=finish_guard 对拍板类请示走 _LEGIT_ASK_RE
合法豁免直接放行，无自包含校验。

判定契约：拍板请求信号（拍板/请确认/必须先确认/选哪个…）与引用式选项标签
（方案A/B、两方案/N个方案、A还是B）共存，且标签在全文无定义形态
（「方案A：…」「方案 A 后接≥6实义字符」「A：… B：…」）→ 判空洞。
执行汇报（无拍板信号）不触发；含定义形态的自包含拍板不触发。
"""

import logging

import pytest

from plugins import finish_guard as fg


def _assert_diagnosed_without_rewrite(
    response_text: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="plugins.finish_guard"):
        out = fg._transform_llm_output(response_text=response_text)
    assert out is None
    assert "拍板请求不自包含" in caplog.text


def test_bare_label_ask_flagged(caplog: pytest.LogCaptureFixture) -> None:
    """引用「方案A/B」标签且全文无定义 → 诊断进日志（db71411a 实例形态）。"""
    response = "反方审查完成（推荐方案 A，零代码改动）。请拍板后实施。"
    assert fg._is_bare_decision_ask(response) is True
    _assert_diagnosed_without_rewrite(response, caplog)


def test_bare_count_ask_flagged(caplog: pytest.LogCaptureFixture) -> None:
    """引用「两方案」计数且全文无定义 → 诊断进日志（db71411a 实例形态）。"""
    response = (
        "反方审查子代理已派出，只读审查按群路由不同项目空间两方案。"
        "审查结果回来后我给方案对比与推荐，你拍板后实施。"
    )
    assert fg._is_bare_decision_ask(response) is True
    _assert_diagnosed_without_rewrite(response, caplog)


def test_bare_ab_ask_flagged(caplog: pytest.LogCaptureFixture) -> None:
    """「A 还是 B」裸字母形态且无定义 → 诊断进日志。"""
    response = "两种做法 A 还是 B，请你拍板一下，我按你选的执行。"
    assert fg._is_bare_decision_ask(response) is True
    _assert_diagnosed_without_rewrite(response, caplog)


def test_selfcontained_ask_not_flagged() -> None:
    """同轮复述选项内容（定义形态存在）→ 不改写（返回 None，契约要求）。"""
    out = fg._transform_llm_output(
        response_text="需要你拍板：方案 A：单机器人多群共享 profile，零配置改动，"
        "但跨群记忆互通；方案 B：每项目一个 profile，彻底隔离，须逐群配置。"
        "建议选 B；默认动作=不回复则维持现状。"
    )
    assert out is None  # 期望: None（不改写必须返回 None，后续 transform 仍会执行）


def test_execution_report_not_flagged() -> None:
    """执行汇报含「方案 A」字样但无拍板请求信号 → 不触发（返回 None）。"""
    out = fg._transform_llm_output(
        response_text="已按方案 A 完成部署并验证，路由命中正确，全链路 200。"
    )
    assert out is None  # 期望: None（无拍板信号不改写）


def test_open_question_without_labels_not_flagged() -> None:
    """拍板信号存在但无选项标签（开放问题，如「绑哪个项目空间？」）→ 不触发。"""
    out = fg._transform_llm_output(
        response_text="必须先确认：当前群绑哪个项目空间？请给项目名与仓库路径。"
        "默认动作=不回复则不绑定。"
    )
    assert out is None  # 期望: None（开放性问题无标签不改写）


def test_pre_verify_nudges_bare_ask() -> None:
    """pre_verify 通道同样拦截空洞拍板收尾（改过文件的回合）。"""
    r = fg._on_pre_verify(
        final_response="推荐方案 A，审查通过。等你拍板后实施。",
        changed_paths=["some/file.py"],
        attempt=0,
    )
    assert r.get("action") == "continue"  # 期望: 强制续跑重写收尾
    assert "自包含" in r.get("message", "")  # 期望: 续跑消息指明补选项内容


def test_pre_verify_attempt_cap_holds() -> None:
    """attempt 达上限仍放行，防续跑死循环（与既有五类契约一致）。"""
    r = fg._on_pre_verify(
        final_response="推荐方案 A，等你拍板。",
        changed_paths=["some/file.py"],
        attempt=fg._MAX_PRE_VERIFY_NUDGES,
    )
    assert r == {}  # 期望: {}（attempt 达上限放行）


def test_short_content_after_label_still_bare(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """标签后短属性词（<6 实义字符）不算定义 → 仍只记诊断。"""
    response = "两方案审查完毕，方案 A 零改动。请拍板。"
    assert fg._is_bare_decision_ask(response) is True
    _assert_diagnosed_without_rewrite(response, caplog)


def test_pre_verify_zero_edit_with_own_pending_todos(tmp_path, monkeypatch) -> None:
    """T8：零写入回合 + 本会话 todo 有 active 项 → 状态判据命中强制续跑。

    用户主诉形态：方案输出后以「是否按X开工？如无异议我按序施工」停轮，
    本回合零文件写入、句式黑名单不覆盖该形态——由 todo 账本状态判据兜住
    （登记不等于完成）。
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from plugins.finish_guard import todo_debt

    todo_debt._update_ledger("s_ownz", [
        {"id": "w1", "content": "修复僵尸行", "status": "in_progress"},
        {"id": "w2", "content": "修复截断", "status": "pending"},
    ])
    r = fg._on_pre_verify(
        session_id="s_ownz",
        attempt=0,
        final_response="是否按上述三件开工？如无异议我按序施工。",
        changed_paths=[],
    )
    assert r.get("action") == "continue"  # 期望: todo 未清+零写入 → 强制续跑
    assert "todo" in r.get("message", "")  # 期望: 续跑消息指向未完成 todo


def test_pre_verify_zero_edit_no_todo_passes(tmp_path, monkeypatch) -> None:
    """T9负例：零写入且本会话无未清 todo → 放行（纯问答回合不受影响）。"""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    r = fg._on_pre_verify(
        session_id="s_clean",
        attempt=0,
        final_response="这是对用户问题的直接回答，无未清事项。",
        changed_paths=[],
    )
    assert r == {}  # 期望: 无状态欠账时零写入回合照常放行
