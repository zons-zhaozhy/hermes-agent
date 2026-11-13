"""builder_gate 负例覆盖测试（新护栏资产强制随附，violating-must-red）。

事故形态（2026-11-13 用户令）：零文件写入回合以提案式问句收尾停轮——
「如果你想，我可以把这份复盘沉淀成 skill」——把本该当轮执行的建设动作
降格为提案退回用户。finish_guard 的零写入分支只执法 todo 债务，纯文本
回合的提案式收尾结构性放行（其代码注释明示 zero-edit turn is a
question/status answer），本插件补此缺口。

本文件锁五件事：
  B1 零写入 + 提案式收尾 → 强制续跑（真实事故句式必须红）；
  B2 合法拍板请示豁免（不可逆/必须先确认）→ 放行；
  B3 写入回合 → 放行（归 finish_guard，不重复执法）；
  B4 完成证据报告豁免（commit 哈希 + 完成动词）→ 放行；
  B5 attempt 上限与短回复豁免 → 放行。
"""
from __future__ import annotations

from plugins import builder_gate as bg

# 真实事故句式（本会话 2026-11-13 原句变体）
_VIOLATING_TAIL = (
    "综合以上，落地的坎坷来自四类系统性缺口，业界已有对应解法。"
    "如果你想，我可以把这份复盘沉淀成一个 skill，把五条默认动作固化下来。"
)
_LEGIT_DECISION_TAIL = (
    "诊断已完成，修复方案涉及生产库覆盖，属不可逆操作，必须先确认："
    "是否执行（默认：不回复则不执行）。"
)
_WRITE_TURN_TAIL = "改动已提交。要不要我继续做第二批？"  # 写入回合归 finish_guard
_DONE_REPORT_TAIL = (
    "全部完成并已推送，commit 3f2a9c1。测试 24/24 绿，剩余边界：未在内网验证。"
    "需要的话我可以继续出离线包。"
)
_PLAIN_CLOSE = "根因是配置漂移，已说明四类缺口与对应业界实践，结论如上。"


def test_b1_violating_tail_forces_continue():
    """B1：零写入 + 提案式收尾触发强制续跑。"""
    out = bg._on_pre_verify(attempt=0, final_response=_VIOLATING_TAIL, changed_paths=[])
    assert out.get("action") == "continue"  # 期望: 真实事故句式必须被拦
    assert "提案" in out.get("message", "")  # 期望: 消息点名提案式形态


def test_b1_signature_variants_unit():
    """B1 单元面：各签名措辞变体全命中，豁免形态全放行。"""
    for phrase in ("要不要我", "是否需要我", "需不需要我", "如果你想，我可以",
                   "需要的话我可以", "我可以帮你整理"):
        assert bg._is_analyst_signature(f"分析结论如下，共四类系统性缺口需要治理，{phrase}后续。") is True  # 期望: 变体命中
    assert bg._is_analyst_signature("") is False  # 期望: 空串不拦
    assert bg._is_analyst_signature("好的。") is False  # 期望: 短回复不拦


def test_b2_legit_decision_exempt():
    """B2：真须拍板的请示（含豁免词）放行。"""
    out = bg._on_pre_verify(attempt=0, final_response=_LEGIT_DECISION_TAIL, changed_paths=[])
    assert out == {}  # 期望: 合法拍板请示不被拦


def test_b3_write_turn_not_ours():
    """B3：changed_paths 非空的回合放行——写入回合收尾归 finish_guard。"""
    out = bg._on_pre_verify(attempt=0, final_response=_WRITE_TURN_TAIL, changed_paths=["a.py"])
    assert out == {}  # 期望: 不重复执法


def test_b4_done_report_exempt():
    """B4：带 commit 哈希+完成动词的报告性收尾放行。"""
    assert bg._is_analyst_signature(_DONE_REPORT_TAIL) is False  # 期望: 完成证据豁免


def test_b5_attempt_cap_and_plain_close():
    """B5：attempt 达上限放行；正常陈述收尾放行。"""
    assert bg._on_pre_verify(attempt=1, final_response=_VIOLATING_TAIL, changed_paths=[]) == {}  # 期望: 超限不续跑
    assert bg._on_pre_verify(attempt=0, final_response=_PLAIN_CLOSE, changed_paths=[]) == {}  # 期望: 正常收尾放行


def test_zero_edit_optin_declared():
    """契约面：零写入回合咨询声明存在（缺失则零写入回合结构性绕过）。"""
    assert getattr(bg._on_pre_verify, "pre_verify_zero_edit", False) is True  # 期望: opt-in 声明在位
