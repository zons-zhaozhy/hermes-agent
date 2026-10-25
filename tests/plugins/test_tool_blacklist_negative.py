"""tool_blacklist 负例契约测试——必败工具黑名单必须真拦（违规必红）。

Contract:
    Preconditions: plugins/guards/tool_blacklist.py 存在
    Postconditions: 三条黑名单规则对违规输入均返回 {"action": "block"}
"""
import importlib.util
from pathlib import Path
from types import ModuleType


def _load_guard() -> ModuleType:
    repo = Path(__file__).resolve().parents[2]
    path = repo / "plugins" / "guards" / "tool_blacklist.py"
    spec = importlib.util.spec_from_file_location("tool_blacklist_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_negative_vision_analyze_blocked():
    guard = _load_guard()
    out = guard._on_pre_tool_call(tool_name="vision_analyze", args={})
    assert out.get("action") == "block"  # 期望: block=B1 黑名单必拦


def test_negative_memory_batch_operations_blocked():
    guard = _load_guard()
    out = guard._on_pre_tool_call(
        tool_name="memory",
        args={"operations": [{"action": "add"}, {"action": "add"}]},
    )
    assert out.get("action") == "block"  # 期望: block=B3 批量 operations 必拦


def test_negative_codegraph_outside_hermes_repo_blocked(monkeypatch):
    guard = _load_guard()
    monkeypatch.setattr(guard.os, "getcwd", lambda: "/tmp/other-project")
    out = guard._on_pre_tool_call(tool_name="mcp__codegraph__context", args={})
    assert out.get("action") == "block"  # 期望: block=B2 非 hermes 仓必拦


def test_negative_clean_tool_passes():
    guard = _load_guard()
    out = guard._on_pre_tool_call(tool_name="terminal", args={"command": "ls"})
    assert out == {}  # 期望: {}=非黑名单工具放行
