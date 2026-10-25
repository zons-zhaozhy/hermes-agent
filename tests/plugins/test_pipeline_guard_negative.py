"""pipeline_guard 负例契约测试——管线阶段 1-3 写工具必须真拦（违规必红）。

Contract:
    Preconditions: plugins/pipeline-guard/__init__.py 存在
    Postconditions: 活跃管线阶段<4 时写工具被 block；只读 terminal 与非管线会话放行
"""
import importlib.util
from pathlib import Path
from types import ModuleType


def _load_guard() -> ModuleType:
    repo = Path(__file__).resolve().parents[2]
    path = repo / "plugins" / "pipeline-guard" / "__init__.py"
    spec = importlib.util.spec_from_file_location("pipeline_guard_under_test", path)
    assert spec is not None and spec.loader is not None  # 期望: 插件文件可定位
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_negative_write_tool_blocked_in_early_phase(monkeypatch):
    guard = _load_guard()
    monkeypatch.setattr(guard, "_read_phase_state",
                        lambda sid: {"active": True, "phase": 2})
    out = guard.on_pre_tool_call(tool_name="write_file", args={"path": "x"}, session_id="s1")
    assert out is not None and out.get("action") == "block"  # 期望: block=阶段2 写工具必拦


def test_negative_write_terminal_command_blocked_in_early_phase(monkeypatch):
    guard = _load_guard()
    monkeypatch.setattr(guard, "_read_phase_state",
                        lambda sid: {"active": True, "phase": 1})
    out = guard.on_pre_tool_call(tool_name="terminal",
                                 args={"command": "echo hi > f.txt"}, session_id="s1")
    assert out is not None and out.get("action") == "block"  # 期望: block=阶段1 写命令必拦


def test_negative_readonly_terminal_allowed_in_early_phase(monkeypatch):
    guard = _load_guard()
    monkeypatch.setattr(guard, "_read_phase_state",
                        lambda sid: {"active": True, "phase": 1})
    out = guard.on_pre_tool_call(tool_name="terminal",
                                 args={"command": "scripts/run_tests.sh tests/"}, session_id="s1")
    assert out is None  # 期望: None=只读测试命令放行


def test_negative_inactive_pipeline_allows_writes(monkeypatch):
    guard = _load_guard()
    monkeypatch.setattr(guard, "_read_phase_state", lambda sid: None)
    out = guard.on_pre_tool_call(tool_name="write_file", args={"path": "x"}, session_id="s1")
    assert out is None  # 期望: None=非管线会话放行
