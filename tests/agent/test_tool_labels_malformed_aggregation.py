# dupcheck-ok: tests/ 下无 tool_labels 既有测试（ls 实查零同主题文件）
"""labels_for_call 对畸形 tool_call 载荷的聚合渲染契约。

业务背景：模型偶发产出重复循环退化的 tool_call 载荷（大量无 name 条目），
参数校验层已拒收零执行；展示层旧实现逐条渲染「Unnamed call」造成活动流刷屏。
期望行为：无名条目聚合为单条「Invalid tool call / N calls rejected」标签。
"""
from __future__ import annotations

from agent.display import tool_labels_for_call  # 消费入口(display.py:545), 同 test_display.py 模式


def labels_for_call(function_name: str, function_args: dict | None) -> list:
    return tool_labels_for_call(function_name, function_args)



def test_malformed_calls_aggregate_to_single_label():
    # 期望: 21 个无 name 条目 → 恰 1 条聚合标签（旧实现 21 条「Unnamed call」刷屏）
    args = {"calls": [{"arguments": {"x": i}} for i in range(21)]}
    labels = labels_for_call("tool_call", args)
    # 期望: 全部条目无名 → labels 恰 1 条，app 文案为 Invalid tool call
    assert len(labels) == 1  # 期望: 1
    assert labels[0].app == "Invalid tool call"  # 期望: 聚合标签而非 Unnamed call
    assert "21 calls rejected" in labels[0].action  # 期望: 计数入 action


def test_single_malformed_entry_uses_singular_wording():
    # 期望: 1 个无名条目 → 单数措辞 "1 call rejected"
    labels = labels_for_call("tool_call", {"calls": [{"arguments": {}}]})
    assert len(labels) == 1  # 期望: 1
    assert "1 call rejected" in labels[0].action  # 期望: 单数


def test_named_entries_unaffected_and_malformed_appended_last():
    # 期望: 混合载荷——合法命名条目逐条正常渲染，无名条目聚合 1 条追加尾部
    args = {"calls": [
        {"name": "tool_search", "arguments": {"queries": ["gmail send"]}},
        {"arguments": {"junk": 1}},
        {"arguments": {"junk": 2}},
    ]}
    labels = labels_for_call("tool_call", args)
    # tool_call 载荷内 name=tool_search 走 label_for_tool_name 本地工具标签；
    # 无名 2 条聚合 1 条 → 共 2 条
    assert len(labels) == 2  # 期望: 2
    assert labels[-1].app == "Invalid tool call"  # 期望: 聚合标签在尾部
    assert "2 calls rejected" in labels[-1].action  # 期望: 2


def test_unnamed_call_string_fully_gone():
    # 期望: 任何输入不再产出「Unnamed call」字样（防旧兜底回流）
    labels = labels_for_call("tool_call", {"calls": [{}]})
    rendered = " ".join(str(l.app) + str(l.action) for l in labels)
    assert "Unnamed" not in rendered  # 期望: 无 Unnamed 字样
