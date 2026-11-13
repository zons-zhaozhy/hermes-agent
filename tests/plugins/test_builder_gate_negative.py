"""builder_gate 负例覆盖测试（新护栏资产强制随附，violating-must-red）。

事故形态（2026-11-13 用户令）：零文件写入回合以提案式问句收尾停轮——
「如果你想，我可以把这份复盘沉淀成 skill」——把本该当轮执行的建设动作
降格为提案退回用户。finish_guard 的零写入分支只执法 todo 债务，纯文本
回合的提案式收尾结构性放行（其代码注释明示 zero-edit turn is a
question/status answer），本插件补此缺口。

判定方式为辅助 LLM 语义判定（plugins/_llm_judge.llm_judge_bool，
2026-11-13 二次修正：禁关键词穷举）。单元测试不调真实 LLM——judge 桩化为
True/False/None/异常四态，本文件锁的是判定链路接线：谁送判、送什么、
结果如何路由、哪些路径不送判。

本文件锁六件事：
  B1 零写入 + judge 判提案式（True）→ 强制续跑；
  B1u 单元面：四态路由（True 拦 / False 放 / None 放 / 异常放）+ 送判参数；
  B2 judge 判非提案（False，含合法拍板形态）→ 放行；
  B3 写入回合 → 放行（归 finish_guard，不重复执法）；
  B4 完成证据豁免（commit 哈希 + 完成动词）→ 不送判直接放行；
  B5 attempt 上限与短回复豁免 → 放行（短回复不送判）。
"""
from __future__ import annotations

import pytest

from plugins import builder_gate as bg

# 真实事故句式（本会话 2026-11-13 原句变体；judge 桩返回 True 模拟其判定）
_VIOLATING_TAIL = (
    "综合以上，落地的坎坷来自四类系统性缺口，业界已有对应解法。"
    "如果你想，我可以把这份复盘沉淀成一个 skill，把五条默认动作固化下来。"
)
# E2E 实测漏网变体（hermes chat 2026-11-13，首版正则放行、语义可判）
_E2E_VARIANT_TAIL = (
    "现象与根因已说明如上，属 Maven 通用机制。"
    "如需我在具体项目里实测复现并落盘报告，直接给出该项目路径即可。"
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
_PLAIN_CLOSE = "根因是配置漂移，已说明四类缺口与对应业界实践，结论如上，剩余边界为零。"


@pytest.fixture
def judge_stub(monkeypatch):
    """可编程 judge 桩：记录送参，按队列/标量返回。"""
    calls: list[dict] = []

    def make(retval):
        def _fake(task, system, text, timeout):
            calls.append({"task": task, "system": system, "text": text, "timeout": timeout})
            if isinstance(retval, Exception):
                raise retval
            return retval

        monkeypatch.setattr(bg, "llm_judge_bool", _fake)
        return _fake

    make.calls = calls  # type: ignore[attr-defined]
    return make


def test_b1_violating_tail_forces_continue(judge_stub):
    """B1：零写入 + judge 判提案式（True）→ 强制续跑。"""
    judge_stub(True)
    out = bg._on_pre_verify(attempt=0, final_response=_VIOLATING_TAIL, changed_paths=[])
    assert out.get("action") == "continue"  # 期望: 判定 True 必须续跑
    assert "提案" in out.get("message", "")  # 期望: 消息点名提案式形态


def test_b1u_judge_routing_and_args(judge_stub):
    """B1u 单元面：四态路由 + 送判参数（task/system/末段全文/超时）。"""
    for retval, expect in ((True, True), (False, False), (None, False)):
        judge_stub(retval)
        assert bg._is_analyst_signature(_E2E_VARIANT_TAIL) is expect  # 期望: True拦/False·None放
    judge_stub(RuntimeError("aux down"))
    assert bg._is_analyst_signature(_E2E_VARIANT_TAIL) is False  # 期望: 异常 fail-open
    judge_stub(True)
    bg._is_analyst_signature(_VIOLATING_TAIL)
    c = judge_stub.calls[-1]
    assert c["task"] == "判定助手回复末段是否提案式收尾"  # 期望: 判定任务名固定
    assert c["system"] == bg._JUDGE_SYSTEM  # 期望: 语义规则全文送判
    assert "如果你想，我可以把这份复盘沉淀成一个 skill" in c["text"]  # 期望: 末段原文送判
    assert c["timeout"] == bg._JUDGE_TIMEOUT  # 期望: 超时受控
    assert bg._is_analyst_signature("") is False  # 期望: 空串不送判
    assert bg._is_analyst_signature("好的。") is False  # 期望: 短回复不送判
    assert len(judge_stub.calls) == 5  # 期望: 空串/短回复两次未新增 judge 调用（3路由+1异常+1送参）


def test_b2_legit_decision_exempt(judge_stub):
    """B2：judge 判非提案（False，合法拍板形态归语义规则）→ 放行。"""
    judge_stub(False)
    out = bg._on_pre_verify(attempt=0, final_response=_LEGIT_DECISION_TAIL, changed_paths=[])
    assert out == {}  # 期望: 非提案判定放行


def test_b3_write_turn_not_ours(judge_stub):
    """B3：changed_paths 非空的回合放行——写入回合收尾归 finish_guard，judge 不送。"""
    judge_stub(True)  # 即便判 True 也不归本插件执法
    out = bg._on_pre_verify(attempt=0, final_response=_WRITE_TURN_TAIL, changed_paths=["a.py"])
    assert out == {}  # 期望: 不重复执法
    assert judge_stub.calls == []  # 期望: 写入回合根本不送判


def test_b4_done_report_exempt(judge_stub):
    """B4：带 commit 哈希+完成动词的报告性收尾——不送判直接放行。"""
    judge_stub(True)  # 桩判 True 也不该被咨询
    assert bg._is_analyst_signature(_DONE_REPORT_TAIL) is False  # 期望: 完成证据豁免
    assert judge_stub.calls == []  # 期望: 机械事实快速通道，零 judge 预算


def test_b5_attempt_cap_and_plain_close(judge_stub):
    """B5：attempt 达上限放行；正常陈述收尾（judge False）放行。"""
    judge_stub(True)
    assert bg._on_pre_verify(attempt=1, final_response=_VIOLATING_TAIL, changed_paths=[]) == {}  # 期望: 超限不续跑
    judge_stub(False)
    assert bg._on_pre_verify(attempt=0, final_response=_PLAIN_CLOSE, changed_paths=[]) == {}  # 期望: 正常收尾放行


def test_zero_edit_optin_declared():
    """契约面：零写入回合咨询声明存在（缺失则零写入回合结构性绕过）。"""
    assert getattr(bg._on_pre_verify, "pre_verify_zero_edit", False) is True  # 期望: opt-in 声明在位
