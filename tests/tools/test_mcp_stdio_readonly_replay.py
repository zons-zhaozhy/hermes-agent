"""Contract: stdio-child-exit replay safety for read-only MCP tools.

Regression for the 2026-10-27 glm-vision incident: a read-only analyze_image call lost a full
turn when a gateway-restart window killed its stdio child mid-call and the recoverer refused
to replay unconditionally. Read-only calls (server readOnlyHint=True or the user's
``read_only_tools`` assertion) must respawn and replay once; write-capable calls keep the
outcome-uncertain block; malformed ``read_only_tools`` skips the server at load.
"""

from types import SimpleNamespace
from typing import Any, Callable

import pytest

from tools import mcp_tool_config as mcp_config
from tools import mcp_tool_handlers as handlers
from tools import mcp_tool_registration as registration
from tools import mcp_tool as mcp_core


@pytest.fixture(autouse=True)
def _clean_hint_state() -> None:
    yield
    mcp_core._tool_read_only_hints.pop("k-test", None)


def test_read_only_tools_assertion_marks_tools_read_only() -> None:
    tools = [SimpleNamespace(name="analyze_image", annotations=None),
             SimpleNamespace(name="write_thing", annotations=None)]
    registration._record_tool_trust_metadata(
        "srv-x", {"read_only_tools": ["analyze_image"]}, tools, key="k-test")
    hints = mcp_core._tool_read_only_hints["k-test"]
    assert hints["analyze_image"] is True  # 期望: 用户断言合入只读表
    assert hints["write_thing"] is not True  # 期望: 未断言工具不受影响


def test_read_only_tools_missing_or_malformed() -> None:
    registration._record_tool_trust_metadata(
        "srv-x", {}, [SimpleNamespace(name="t", annotations=None)], key="k-test")
    assert mcp_core._tool_read_only_hints["k-test"] == {"t": False}  # 期望: 无键零变化
    with pytest.raises(ValueError):  # 期望: 非法值响亮报错
        registration._config_read_only_tools({"read_only_tools": "analyze_image"})


def test_malformed_read_only_tools_skips_server() -> None:
    servers = {
        "bad": {"command": "/bin/true", "read_only_tools": [123]},
        "good": {"command": "/bin/true", "read_only_tools": ["analyze_image"]},
    }
    kept = mcp_config._filter_suspicious_mcp_servers(servers)
    assert list(kept) == ["good"]  # 期望: 非法断言致 server 整体跳过，合法断言保留


def _patch_recover_env(monkeypatch: pytest.MonkeyPatch, retry_result: str = "replayed-ok"
                       ) -> tuple[Callable[[], str], dict[str, int]]:
    calls: dict[str, int] = {"retry": 0, "respawn": 0}

    def _fake_retry() -> str:
        calls["retry"] += 1
        return retry_result

    monkeypatch.setattr(handlers, "_lookup_reconnectable_server",
                        lambda name, require_loop=False: SimpleNamespace())
    monkeypatch.setattr(handlers, "_mcp_loop_running", lambda: True)
    monkeypatch.setattr(handlers, "_loop", SimpleNamespace(
        _signal_reconnect=lambda srv: calls.__setitem__("respawn", calls["respawn"] + 1),
        _signal_reconnect_and_wait=lambda name, srv, op_description="", timeout=0:
            (calls.__setitem__("respawn", calls["respawn"] + 1), True)[1]))
    monkeypatch.setattr(handlers, "_record_call_outcome", lambda name, result: result)
    return _fake_retry, calls


def test_midflight_exit_write_tool_blocks_without_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    retry, calls = _patch_recover_env(monkeypatch)
    exc = handlers._StdioChildExited("child gone", in_flight=True)
    out = handlers._handle_stdio_child_exited_and_retry(
        "srv-x", exc, retry, "tools/call write_thing", call_may_have_side_effects=True)
    assert "operation may have completed" in out  # 期望: 写工具 mid-call 死亡不重放(outcome-uncertain 响亮返回)
    assert calls["retry"] == 0  # 期望: retry_call 从未执行
    assert calls["respawn"] == 1  # 期望: 重连照常为后续调用恢复


def test_midflight_exit_read_only_replays_once(monkeypatch: pytest.MonkeyPatch) -> None:
    retry, calls = _patch_recover_env(monkeypatch)
    exc = handlers._StdioChildExited("child gone", in_flight=True)
    out = handlers._handle_stdio_child_exited_and_retry(
        "srv-x", exc, retry, "tools/call analyze_image", call_may_have_side_effects=False)
    assert out == "replayed-ok"  # 期望: 只读工具 mid-call 死亡重放一次成功
    assert calls["retry"] == 1  # 期望: 恰好重放一次
    assert calls["respawn"] == 1  # 期望: 重连先于重放


def test_non_stdio_exit_not_handled(monkeypatch: pytest.MonkeyPatch) -> None:
    _, calls = _patch_recover_env(monkeypatch)
    out: Any = handlers._handle_stdio_child_exited_and_retry(
        "srv-x", RuntimeError("other"), lambda: "x", "tools/call t")
    assert out is None  # 期望: 非本类错误不接管
    assert calls == {"retry": 0, "respawn": 0}  # 期望: 零副作用
