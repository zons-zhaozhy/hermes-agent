"""skill_diff_echo 负例契约测试——skill_manage 写操作必须回显真实 diff（违规必红）。

Contract:
    Preconditions: plugins/skill_diff_echo/__init__.py 存在
    Postconditions: 有快照的成功写操作追加 [skill_diff] 段；非 skill_manage 不处理
"""
import importlib.util
import json
from pathlib import Path
from types import ModuleType


def _load_guard() -> ModuleType:
    repo = Path(__file__).resolve().parents[2]
    path = repo / "plugins" / "skill_diff_echo" / "__init__.py"
    spec = importlib.util.spec_from_file_location("skill_diff_echo_under_test", path)
    assert spec is not None and spec.loader is not None  # 期望: 插件文件可定位
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_negative_write_result_gains_diff_section(tmp_path: Path):
    guard = _load_guard()
    target = tmp_path / "SKILL.md"
    target.write_text("old line\n", encoding="utf-8")
    guard._snapshots.clear()
    guard._snapshots[(0, str(target))] = "old line\n"
    target.write_text("old line\nnew line\n", encoding="utf-8")
    args = {"action": "patch", "name": "demo", "old_string": "old line",
            "new_string": "old line\nnew line"}
    result = json.dumps({"success": True})
    out = guard.transform_tool_result("skill_manage", args, result)
    assert out is not None and "[skill_diff]" in out  # 期望: 写操作结果必附 diff 回显（欠账拦截）
    assert "+new line" in out  # 期望: diff 含真实磁盘变化


def test_negative_non_skill_manage_untouched():
    guard = _load_guard()
    out = guard.transform_tool_result("terminal", {"command": "ls"}, "ok")
    assert out is None  # 期望: None=非 skill_manage 不处理


def test_negative_failed_result_not_echoed():
    guard = _load_guard()
    args = {"action": "patch", "name": "demo", "old_string": "x", "new_string": "y"}
    out = guard.transform_tool_result("skill_manage", args, json.dumps({"success": False}))
    assert out is None  # 期望: None=失败结果不追加（仅成功写须回显）
