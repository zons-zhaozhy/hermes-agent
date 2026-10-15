"""Tests for plugins/skill_diff_echo: skill_manage result diff echo."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from plugins.skill_diff_echo import transform_tool_result


def _ok_result() -> str:
    return json.dumps({"success": True, "operations_applied": 1})


def test_patch_op_appends_minus_plus_lines():
    args = {"action": "patch", "name": "my-skill", "old_string": "alpha\nbeta", "new_string": "alpha\ngamma"}
    out = transform_tool_result("skill_manage", args, _ok_result())
    assert out is not None and "--- my-skill (patch)" in out  # 期望: 标签=技能名+action
    assert "-beta" in out and "+gamma" in out  # 期望: 旧行-前缀、新行+前缀（diff 语义）


def test_batch_ops_count_in_header():
    args = {"operations": [
        {"action": "patch", "name": "s1", "old_string": "a", "new_string": "b"},
        {"action": "write_file", "name": "s1", "file_path": "references/x.md", "file_content": "h1\nh2"},
    ]}
    out = transform_tool_result("skill_manage", args, _ok_result())
    assert out is not None and out.startswith(_ok_result())  # 期望: 原结果保留，diff 追加在后
    assert "[skill_diff] 2 op(s):" in out  # 期望: len(sections)=2，计数即操作数
    assert "+h2" in out  # 期望: 全量写操作逐行渲染


def test_failed_result_untouched():
    bad = json.dumps({"success": False, "error": "nope"})
    out = transform_tool_result("skill_manage", {"action": "patch", "name": "x"}, bad)
    assert out is None  # 期望: 失败结果不附 diff


def test_other_tool_passthrough():
    out = transform_tool_result("write_file", {"path": "/tmp/x", "content": "y"}, _ok_result())
    assert out is None  # 期望: 仅 skill_manage 触发，其他工具透传


def test_long_full_file_head_tail():
    lines = [f"line{i}" for i in range(1, 16)]  # 15 行：头 line1-5，尾 line11-15
    args = {"action": "create", "name": "big", "content": "\n".join(lines)}
    out = transform_tool_result("skill_manage", args, _ok_result())
    assert out is not None and "+line1" in out and "+line15" in out  # 期望: 头尾行均在
    assert "(15 lines total)" in out and "15 lines, new/overwritten" in out  # 期望: 总数 15=入参行数
    assert "+line6\n" not in out  # 期望: >10 行走头5+计数+尾5，中间行不逐行输出


def test_remove_file_marker():
    args = {"action": "remove_file", "name": "s1", "file_path": "references/old.md"}
    out = transform_tool_result("skill_manage", args, _ok_result())
    assert out is not None and "(remove_file, file removed)" in out  # 期望: 删除以 file removed 标注
