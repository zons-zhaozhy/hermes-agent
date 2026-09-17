"""persistence-enforcer 回归测试。

行为契约（独立推导）：
- 解锁凭据只认真实生效的调用：status=blocked/error 的 write_file/todo
  不得置 persist_called/todo_called（自锁自溃防线）
- 状态按 session 隔离：A 会话计数不得影响 B 会话的拦截判定
- 计满阈值 + 无 todo/persist → block write_file；真 todo（status=ok）→ 解锁
"""
import importlib
from typing import Any, Generator

import pytest

SID_A = "sess-a"
SID_B = "sess-b"


@pytest.fixture(autouse=True)
def _clean_states() -> Generator[None, None, None]:
    plugin = importlib.import_module("plugins.persistence-enforcer")
    plugin._states.clear()
    yield
    plugin._states.clear()


@pytest.fixture
def plugin() -> Any:
    return importlib.import_module("plugins.persistence-enforcer")


def _post(plugin: Any, tool_name: str, sid: str, status: str = "ok") -> None:
    plugin._on_post_tool_call(
        session_id=sid, tool_name=tool_name, args={}, status=status, result={}
    )


def test_blocked_write_does_not_lift_gate(plugin: Any) -> None:
    # 期望: 被拦下的 write_file（status=blocked）不解锁拦截门
    for i in range(plugin.BLOCK_THRESHOLD + 2):
        _post(plugin, "read_file", SID_A)
    _post(plugin, "write_file", SID_A, status="blocked")
    out = plugin._on_pre_tool_call(
        session_id=SID_A, tool_name="write_file", args={"path": "/tmp/x.py"}
    )
    assert out.get("action") == "block"  # 期望: 拦截仍在


def test_ok_write_lifts_gate(plugin: Any) -> None:
    for i in range(plugin.BLOCK_THRESHOLD + 2):
        _post(plugin, "read_file", SID_A)
    _post(plugin, "write_file", SID_A, status="ok")
    out = plugin._on_pre_tool_call(
        session_id=SID_A, tool_name="write_file", args={"path": "/tmp/x.py"}
    )
    assert out == {}  # 期望: 真实生效的写入解锁


def test_error_write_does_not_lift_gate(plugin: Any) -> None:
    for i in range(plugin.BLOCK_THRESHOLD + 2):
        _post(plugin, "read_file", SID_A)
    _post(plugin, "write_file", SID_A, status="error")
    out = plugin._on_pre_tool_call(
        session_id=SID_A, tool_name="write_file", args={"path": "/tmp/x.py"}
    )
    assert out.get("action") == "block"  # 期望: 失败的写入不算凭据


def test_ok_todo_lifts_gate(plugin: Any) -> None:
    for i in range(plugin.BLOCK_THRESHOLD + 2):
        _post(plugin, "read_file", SID_A)
    _post(plugin, "todo", SID_A, status="ok")
    out = plugin._on_pre_tool_call(
        session_id=SID_A, tool_name="write_file", args={"path": "/tmp/x.py"}
    )
    assert out == {}  # 期望: 真实 TODO 解锁


def test_session_isolation(plugin: Any) -> None:
    # 期望: A 会话的计数不影响 B 会话——并发会话互踩防线
    for i in range(plugin.BLOCK_THRESHOLD + 2):
        _post(plugin, "read_file", SID_A)
    out_b = plugin._on_pre_tool_call(
        session_id=SID_B, tool_name="write_file", args={"path": "/tmp/y.py"}
    )
    assert out_b == {}  # 期望: B 会话计数为 0，放行


def test_blocked_todo_does_not_lift(plugin: Any) -> None:
    for i in range(plugin.BLOCK_THRESHOLD + 2):
        _post(plugin, "read_file", SID_A)
    _post(plugin, "todo", SID_A, status="blocked")
    out = plugin._on_pre_tool_call(
        session_id=SID_A, tool_name="write_file", args={"path": "/tmp/x.py"}
    )
    assert out.get("action") == "block"  # 期望: 被拦的 todo 不算凭据
