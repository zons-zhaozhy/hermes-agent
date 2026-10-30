"""finish_guard 授权范围闸门回归测试。

事故形态：收尾宣布「下一步：建 MV0 骨架」后停轮，L2 续跑消息只以动作可逆性为门槛
（「可逆决策标注假设后直接连做剩余步骤」），授权范围维缺失——把「论证可行性」当成
建造授权，越权新建应用骨架。修法：三层消息统一先划范围（当轮授权范围 = 用户当轮
原话明确要求产出的东西；分析/设计/评审/调研类请求不等于建造授权），范围外的宣布
改列「必须先确认：」项而非执行。

本文件锁两件事：判定口径不变（I1/I2），三条消息都携带范围闸门（I5）。
"""
from __future__ import annotations

from plugins import finish_guard as fg

_DECLARED = "进度正常，下一步：建 MV0 骨架。"


def test_declared_next_step_still_nudged():
    """宣言式假继续仍被强制续跑（I1 不变量）。"""
    out = fg._on_pre_verify(attempt=0, final_response=_DECLARED, changed_paths=["a.py"])
    assert out.get("action") == "continue"  # 期望: 宣言式收尾仍触发续跑


def test_l2_message_carries_scope_gate():
    """L2 续跑消息携带授权范围闸门（I5）。"""
    out = fg._on_pre_verify(attempt=0, final_response=_DECLARED, changed_paths=["a.py"])
    msg = out.get("message") or ""
    assert "当轮授权范围" in msg  # 期望: 先给出范围定义
    assert "不等于建造授权" in msg  # 期望: 分析类请求不构成建造授权
    assert "不要执行它" in msg  # 期望: 范围外分支=撤回宣布而非执行


def test_l1_message_carries_scope_gate():
    """clarify 拦截消息携带授权范围闸门。"""
    out = fg._on_pre_tool_call(
        tool_name="clarify",
        args={"questions": [{"question": "要不要我继续把骨架建起来？"}]},
    )
    assert out.get("action") == "block"  # 期望: 请示式提问被拦
    assert "不等于建造授权" in out.get("message", "")  # 期望: 拦截消息先划范围


def test_l3_marker_carries_scope_gate():
    """用户可见标记携带授权范围闸门。"""
    out = fg._transform_llm_output(response_text=_DECLARED)
    assert "[finish_guard]" in out  # 期望: 漏网收尾被打标
    assert "不等于建造授权" in out  # 期望: 标记含范围定义


def test_messages_carry_companion_file_carveout():
    """边界收敛不变量：三条消息须同时携带两侧口径——越权线=新能力面
    （新应用/新依赖/新开目录树，与 scope_gate 机械闸门对齐）；既有目录树内
    为执行当轮任务所需的伴生文件不属越权。防「新文件」枚举回归漂移。"""
    l1 = fg._on_pre_tool_call(
        tool_name="clarify",
        args={"questions": [{"question": "要不要我继续把骨架建起来？"}]},
    ).get("message", "")
    l2 = fg._on_pre_verify(
        attempt=0, final_response=_DECLARED, changed_paths=["a.py"]
    ).get("message", "")
    l3 = fg._transform_llm_output(response_text=_DECLARED) or ""
    for name, msg in (("L1", l1), ("L2", l2), ("L3", l3)):
        assert "新开目录树" in msg, name  # 期望: 越权线=新能力面（机械闸门口径）
        assert "伴生" in msg, name  # 期望: 既有树内伴生文件豁免随消息下发


def test_legit_ask_exempt_still_passes():
    """合法请示豁免不受影响（I2 不变量）。"""
    out = fg._on_pre_tool_call(
        tool_name="clarify",
        args={"questions": [{"question": "要不要继续覆盖生产库？"}]},
    )
    assert out == {}  # 期望: 命中生产库/覆盖豁免，放行不拦
