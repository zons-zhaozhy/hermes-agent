"""Tests for plugins/skill_diff_echo v2: unified diff from disk before/after."""

import json
import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from plugins.skill_diff_echo import _snapshots, on_pre_tool_call, transform_tool_result


def _ok_result() -> str:
    return json.dumps({"success": True, "operations_applied": 1})


def test_snapshot_unified_diff(tmp_path):
    skill = tmp_path / "my-skill"
    skill.mkdir()
    target = skill / "SKILL.md"
    target.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    args = {"action": "patch", "name": "my-skill", "old_string": "beta", "new_string": "BETA"}
    with mock.patch("tools.skill_manager_tool._find_skill", return_value={"path": skill}):
        on_pre_tool_call(tool_name="skill_manage", args=args, session_id="s1")
        target.write_text("alpha\nBETA\ngamma\n", encoding="utf-8")  # 模拟工具已写入
        out = transform_tool_result("skill_manage", args, _ok_result())
    assert out is not None and "@@" in out  # 期望: unified diff 带 @@ hunk 头
    assert "-beta" in out and "+BETA" in out and " gamma" in out  # 期望: 上下文行 gamma 空格前缀
    assert (0, str(target)) not in _snapshots  # 期望: 快照一次性消费


def test_no_snapshot_falls_back_to_args_echo():
    args = {"action": "patch", "name": "ghost", "old_string": "a", "new_string": "b"}
    out = transform_tool_result("skill_manage", args, _ok_result())
    assert out is not None and "fallback args echo" in out  # 期望: 无快照降级 v1 形态
    assert "-a" in out and "+b" in out  # 期望: 降级仍保留 -old/+new 对照


def test_batch_ops_count_in_header():
    args = {"operations": [
        {"action": "patch", "name": "s1", "old_string": "a", "new_string": "b"},
        {"action": "write_file", "name": "s1", "file_path": "references/x.md", "file_content": "h1\nh2"},
    ]}
    out = transform_tool_result("skill_manage", args, json.dumps({"success": True}))
    assert out is not None and out.startswith(json.dumps({"success": True}))  # 期望: 原结果保留
    assert "[skill_diff] 2 op(s):" in out  # 期望: 计数=操作数
    assert "+h2" in out  # 期望: 全量写逐行渲染


def test_failed_result_untouched():
    bad = json.dumps({"success": False, "error": "nope"})
    out = transform_tool_result("skill_manage", {"action": "patch", "name": "x"}, bad)
    assert out is None  # 期望: 失败结果不附 diff


def test_other_tool_passthrough():
    out = transform_tool_result("write_file", {"path": "/tmp/x", "content": "y"}, _ok_result())
    assert out is None  # 期望: 仅 skill_manage 触发


def test_pre_hook_ignores_other_tools():
    before = dict(_snapshots)
    on_pre_tool_call(tool_name="write_file", args={"path": "/tmp/zzz"}, session_id="s")
    assert _snapshots == before  # 期望: 非 skill_manage 不产生快照


def test_no_textual_change_marker(tmp_path):
    skill = tmp_path / "s"
    skill.mkdir()
    target = skill / "SKILL.md"
    target.write_text("same\n", encoding="utf-8")
    _snapshots[(0, str(target))] = "same\n"
    args = {"action": "patch", "name": "s", "old_string": "x", "new_string": "y"}
    with mock.patch("tools.skill_manager_tool._find_skill", return_value={"path": skill}):
        out = transform_tool_result("skill_manage", args, _ok_result())
    assert out is not None and "no textual change" in out  # 期望: 磁盘无变化显式标注


def test_remove_file_marker():
    args = {"action": "remove_file", "name": "s1", "file_path": "references/old.md"}
    out = transform_tool_result("skill_manage", args, _ok_result())
    assert out is not None and "file removed" in out  # 期望: 删除操作有标注
