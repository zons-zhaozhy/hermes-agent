"""engine_invariants 负例契约测试——引擎不变量检测必须真抓违规（违规必红）。

Contract:
    Preconditions: plugins/engine-invariants/__init__.py 存在
    Postconditions: 孤儿工具结果/悬挂调用/连续 user 消息均产生非空 violations
"""
import importlib.util
from pathlib import Path
from types import ModuleType


def _load_guard() -> ModuleType:
    repo = Path(__file__).resolve().parents[2]
    path = repo / "plugins" / "engine-invariants" / "__init__.py"
    spec = importlib.util.spec_from_file_location("engine_invariants_under_test", path)
    assert spec is not None and spec.loader is not None  # 期望: 插件文件可定位
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_negative_orphan_tool_result_detected():
    guard = _load_guard()
    msgs = [{"role": "tool", "tool_call_id": "call-404", "content": "x"}]
    out = guard.run_all_invariants(msgs)
    assert any("orphan_tool_result" in v for v in out["tool_pairing"])  # 期望: 孤儿结果被拦


def test_negative_unanswered_tool_call_detected():
    guard = _load_guard()
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call-1", "type": "function", "function": {"name": "t", "arguments": "{}"}}
        ]},
    ]
    out = guard.run_all_invariants(msgs)
    assert any("unanswered_tool_call" in v for v in out["tool_pairing"])  # 期望: 悬挂调用被拦


def test_negative_consecutive_user_detected():
    guard = _load_guard()
    msgs = [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]
    out = guard.run_all_invariants(msgs)
    assert len(out["role_alternation"]) > 0  # 期望: 连续 user 违反交替不变量被拦


def test_negative_clean_history_passes():
    guard = _load_guard()
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call-1", "type": "function", "function": {"name": "t", "arguments": "{}"}}
        ]},
        {"role": "tool", "tool_call_id": "call-1", "content": "ok"},
        {"role": "assistant", "content": "done"},
    ]
    out = guard.run_all_invariants(msgs)
    assert out == {"tool_pairing": [], "role_alternation": []}  # 期望: 合法历史零违反
