"""finish_guard v2.1 推迟式外推（defer pushout）回归测试。

事故形态（1015 用户批评原话）：「你为啥最近总是接口"不列入本轮"、"单独立项处理"，
而不是发现问题就立即处理好？」——收尾以陈述句宣布把顺带发现的问题外推出当轮
（「判定为独立战线暂不混入」「单独立项处理」「留待后续」）后停轮。旧判定矩阵
只有两类：请示式（问句）/宣言式假继续（说要继续）——推迟式是陈述句且明确说不做，
两类都不命中，零检测穿透。

修法：_is_defer_pushout 第三类判定 + L2/L3 挂点接入。合法豁免沿用
_LEGIT_ASK_RE（不可逆/凭据/付费/拍板/必须先确认）。

本文件锁三件事：D1 推迟句拦、D2 合法豁免放、D3 完成态边界话术不误伤。
"""
from __future__ import annotations

from plugins import finish_guard as fg

_DEFER_CLOSE = "诊断已完成。认证链三处矛盾属独立战线，单独立项处理，本轮先收尾。"
_DEFER_LEGIT = "该操作不可逆（生产库覆盖），必须先确认：是否执行。其余发现已全部修复。"
_DONE_CLOSE = "任务全部完成，四个模块已推送，commit 3260f7155。如需扩展另行指示。"


def test_defer_sentence_triggers_nudge():
    """D1：推迟式收尾触发 L2 强制续跑。"""
    out = fg._on_pre_verify(attempt=0, final_response=_DEFER_CLOSE, changed_paths=["a.py"])
    assert out.get("action") == "continue"  # 期望: 推迟式收尾被拦
    assert "单独立项" in out.get("message", "")  # 期望: 消息点名推迟形态


def test_defer_pushout_detection_unit():
    """D1 单元面：各推迟措辞变体全命中。"""
    for phrase in ("不列入本轮", "另行处理", "留待后续", "暂不混入", "独立战线", "后续迭代", "后续跟进"):
        assert fg._is_defer_pushout(f"发现X问题，{phrase}。") is True  # 期望: 变体命中
    assert fg._is_defer_pushout("") is False  # 期望: 空串不拦
    assert fg._is_defer_pushout("任务全部完成。") is False  # 期望: 正常收尾不拦


def test_legit_irreversible_defer_exempt():
    """D2：不可逆/需拍板级推迟放行（真豁免）。

    判据=不进 defer 拦截分支（消息不含推迟式外推语义）；changed_paths 触发的
    commit 证据分支是独立规则，不在本测断言面。"""
    assert fg._is_defer_pushout(_DEFER_LEGIT) is False  # 期望: 命中不可逆豁免词
    out = fg._on_pre_verify(attempt=0, final_response=_DEFER_LEGIT, changed_paths=["a.py"])
    msg = out.get("message") or ""
    assert "推迟式外推" not in msg  # 期望: 未被 defer 判定拦截（commit 证据分支可独立触发）


def test_done_close_not_deferred():
    """D3：完成态收尾（含「另行指示」边界话术）不误伤。"""
    assert fg._is_defer_pushout(_DONE_CLOSE) is False  # 期望: 「另行指示」不在推迟词表
    out = fg._transform_llm_output(response_text=_DONE_CLOSE)
    assert "[finish_guard]" not in out  # 期望: 完成态不打标


def test_l3_marker_covers_defer():
    """D3：L3 用户可见标记覆盖推迟式漏网。"""
    out = fg._transform_llm_output(response_text=_DEFER_CLOSE)
    assert "[finish_guard]" in out  # 期望: 推迟式收尾被打标
    assert "外推" in out  # 期望: 标记含禁止外推语义


def test_old_two_classes_still_work():
    """回归面：旧两类判定不受影响。"""
    assert fg._is_pushback("要不要继续？") is True  # 期望: 请示式仍拦
    assert fg._is_false_continue("继续下一批。") is True  # 期望: 假继续仍拦
    assert fg._is_defer_pushout("要不要继续？") is False  # 期望: 判定类不互相越界
