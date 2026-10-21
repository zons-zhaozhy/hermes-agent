"""finish_guard v2.2 根因隐匿（root cause hidden）回归测试。

事故形态（1021 用户指令原话）：「当你找到问题根因的时候要在回复里准确、
清晰地说明清楚，不能不告知、不回复、不显示」——收尾宣称「已根治/已定位
根因」，但通篇不展示根因内容（机制、证据、file:line）。旧判定矩阵四类
（请示/假继续/推迟/无commit证据）全部不覆盖：它有完成宣言，缺的是陈述本身。

修法：_is_rootcause_hidden 第五类判定（全文判定面）+ L2 pre_verify 强制补述
+ L3 transform_llm_output 用户可见标记，落地 .hermes-rules.md B6。

本文件锁四件事：R1 宣称无证据拦、R2 宣称+证据并存放行、R3 正常收尾不误伤、
R4 旧四类判定不越界。
"""
from __future__ import annotations

from plugins import finish_guard as fg

# 宣称根治但零根因内容（事故形态）
_HIDDEN_CLOSE = "问题已彻底根治，服务恢复正常，本轮收尾。"
# 宣称+证据并存（正常交付形态）
_SHOWN_CLOSE = (
    "根因是 retry 装饰器把 402 当 transient 重试到底（client.py:87 的 "
    "should_retry 未排除 payment 分支），已加分支白名单修复，402 立即失败。"
)
# 无根因宣称的正常收尾
_PLAIN_CLOSE = "部署完成，commit 3260f7155。"

_SENTENCES_UNIT_CLAIM = (
    "已根治",
    "已定位根因并修复",
    "找到根因了",
    "根因已明确",
    "根因已查明",
)
_SENTENCES_UNIT_EVIDENCE = (
    "根因是装饰器误吞 402",
    "根因：链路缺超时",
    "原因在于序列化分支",
    "定位到 client.py:87 的判断",
    "异常位于 scheduler.py 第 214 行",
)


def test_hidden_rootcause_triggers_nudge():
    """R1：宣称根治无根因陈述触发 L2 强制补述。"""
    out = fg._on_pre_verify(attempt=0, final_response=_HIDDEN_CLOSE, changed_paths=["a.py"])
    assert out.get("action") == "continue"  # 期望: 根因隐匿收尾被拦
    assert "根因必答" in out.get("message", "")  # 期望: 消息点名根因必答纪律


def test_rootcause_hidden_detection_unit():
    """R1 单元面：各宣称措辞变体在无证据语境全命中，有证据语境全放行。"""
    for claim in _SENTENCES_UNIT_CLAIM:
        assert fg._is_rootcause_hidden(f"问题{claim}，本轮收尾。") is True  # 期望: 宣称无证据命中
    for evidence in _SENTENCES_UNIT_EVIDENCE:
        assert fg._is_rootcause_hidden(f"已根治。{evidence}，修复完成。") is False  # 期望: 有展开放行
    assert fg._is_rootcause_hidden("") is False  # 期望: 空串不拦


def test_shown_rootcause_not_flagged():
    """R2：宣称+file:line 证据并存的正常收尾不拦。"""
    assert fg._is_rootcause_hidden(_SHOWN_CLOSE) is False  # 期望: 完整交付形态放行
    out = fg._transform_llm_output(response_text=_SHOWN_CLOSE)
    assert "[finish_guard]" not in out  # 期望: 不打标


def test_plain_close_not_flagged():
    """R3：无根因宣称的正常收尾不误伤。"""
    assert fg._is_rootcause_hidden(_PLAIN_CLOSE) is False  # 期望: 无宣称不拦
    out = fg._transform_llm_output(response_text=_PLAIN_CLOSE)
    assert "[finish_guard]" not in out  # 期望: 不打标


def test_l3_marker_covers_hidden_rootcause():
    """R3：L3 用户可见标记覆盖根因隐匿漏网。"""
    out = fg._transform_llm_output(response_text=_HIDDEN_CLOSE)
    assert "[finish_guard]" in out  # 期望: 根因隐匿收尾被打标
    assert "根因必答" in out  # 期望: 标记含根因必答语义


def test_old_four_classes_still_work():
    """R4 回归面：旧四类判定不受影响，判定类不互相越界。"""
    assert fg._is_pushback("要不要继续？") is True  # 期望: 请示式仍拦
    assert fg._is_false_continue("继续下一批。") is True  # 期望: 假继续仍拦
    assert fg._is_defer_pushout("留待后续处理") is True  # 期望: 推迟式仍拦
    assert fg._is_rootcause_hidden("要不要继续？") is False  # 期望: 根因判定不吃请示句
