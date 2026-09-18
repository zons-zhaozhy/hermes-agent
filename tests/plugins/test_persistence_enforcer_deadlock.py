"""persistence-enforcer 死锁解除回归测试。

行为契约（独立推导）：
- cron/one-shot 会话工具集无 todo_list/skill_manage/memory 时，解锁凭据
  物理不可得——连续 DEBLOCK_AFTER 次 block 后第 3 次 write_file 必须放行，
  否则交付无解（2026-09-18 cron 实测死锁）。
- 放行非静默：deblock_pending 置位，下一次 pre_llm_call 注入一次性警告。
- 放行非永久豁免：consecutive_blocks 归零，后续无凭据 block 重新计数。
"""
import importlib
from typing import Any, Generator

import pytest

SID = "sess-deadlock"


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


def _pre_write(plugin: Any, sid: str) -> dict:
    return plugin._on_pre_tool_call(
        session_id=sid, tool_name="write_file", args={"path": "/tmp/x.py"}
    )


def test_deadlock_breaks_after_consecutive_blocks(plugin: Any) -> None:
    # 期望: 前 DEBLOCK_AFTER 次 block，第 3 次放行（unlock 工具不可得场景）
    for _ in range(plugin.BLOCK_THRESHOLD + 2):
        _post(plugin, "read_file", SID)
    for i in range(plugin.DEBLOCK_AFTER):
        assert _pre_write(plugin, SID).get("action") == "block", f"第{i+1}次应拦"
    assert _pre_write(plugin, SID) == {}  # 期望: 第 3 次放行（死锁解除）


def test_deblock_injects_one_shot_warning(plugin: Any) -> None:
    # 期望: 死锁解除后首次 pre_llm_call 注入警告，第二次不再注入
    for _ in range(plugin.BLOCK_THRESHOLD + 2):
        _post(plugin, "read_file", SID)
    for _ in range(plugin.DEBLOCK_AFTER):
        _pre_write(plugin, SID)
    _pre_write(plugin, SID)  # 触发放行，deblock_pending 置位
    out1 = plugin._on_pre_llm_call(session_id=SID, conversation_history=[{"role": "user", "content": "x"}])
    assert "死锁解除" in out1.get("context", "")  # 期望: 一次性警告注入
    out2 = plugin._on_pre_llm_call(session_id=SID, conversation_history=[{"role": "user", "content": "x"}])
    assert "死锁解除" not in out2.get("context", "")  # 期望: 不重复注入


def test_deblock_not_permanent_exemption(plugin: Any) -> None:
    # 期望: 放行后计数归零——若无凭据继续写，重新 block 重新计数（非永久豁免）
    for _ in range(plugin.BLOCK_THRESHOLD + 2):
        _post(plugin, "read_file", SID)
    for _ in range(plugin.DEBLOCK_AFTER):
        _pre_write(plugin, SID)
    assert _pre_write(plugin, SID) == {}  # 死锁解除放行
    # 模拟放行的 write_file 未成功（仍 status=blocked/无凭据），继续探测
    assert _pre_write(plugin, SID).get("action") == "block"  # 期望: 重新拦截


def test_real_persist_still_short_circuits(plugin: Any) -> Any:
    # 期望: 真实生效的 write_file（status=ok）仍立即解锁，不进入死锁逻辑
    for _ in range(plugin.BLOCK_THRESHOLD + 2):
        _post(plugin, "read_file", SID)
    _post(plugin, "write_file", SID, status="ok")
    assert _pre_write(plugin, SID) == {}
