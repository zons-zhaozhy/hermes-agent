"""pre_write 已读证据落盘探针验证（红绿测试）。

规则来源（独立期望）：先读后写门禁的证据在 pre/post 钩子分属不同进程时必须仍可
传递——post 侧记账双写磁盘 marker，pre 侧内存或磁盘任一命中即放行；marker 缺失、
超 TTL（24h）、格式错一律 fail-closed（仍拦）。期望值从上述规则推导，不从实现反推。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from plugins.guards import pre_write

SID = "test-sid-pw-probe"


@pytest.fixture(autouse=True)
def _probe_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """隔离 marker 缓存目录——防测试触碰真实 ~/.hermes。"""
    monkeypatch.setenv("PW_READ_PROBE_CACHE", str(tmp_path / "probe_cache"))


def _post_read(path: str) -> None:
    pre_write.on_post_tool_call(session_id=SID, tool_name="read_file", args={"path": path})


def _pre_patch(path: str) -> Optional[dict[str, Any]]:
    return pre_write.on_pre_tool_call(session_id=SID, tool_name="patch", args={"path": path})


def test_marker_survives_memory_reset(tmp_path: Path) -> None:
    """post 记账后清空内存态（模拟 pre/post 进程分裂），patch 仍放行。"""
    target = tmp_path / "app.py"
    target.write_text("x = 1\n", encoding="utf-8")
    _post_read(str(target))
    pre_write._get_read_files(SID).clear()
    result = _pre_patch(str(target))
    assert result is None  # 期望: 内存清空后磁盘 marker 命中即放行


def test_stale_marker_fails_closed(tmp_path: Path) -> None:
    """marker 超 TTL（伪造 25h 前）时 fail-closed，patch 仍拦。"""
    target = tmp_path / "app.py"
    target.write_text("x = 1\n", encoding="utf-8")
    marker = pre_write._read_marker(SID)
    marker.parent.mkdir(parents=True, exist_ok=True)
    stale = json.dumps({"sid": SID, "ts": time.time() - 90000, "paths": [str(target)]})
    marker.write_text(stale, encoding="utf-8")
    pre_write._get_read_files(SID).clear()
    result = _pre_patch(str(target))
    assert result is not None and result["action"] == "block"  # 期望: 超 TTL 仍拦


def test_no_evidence_blocks(tmp_path: Path) -> None:
    """零证据（无内存无 marker）时 patch 仍拦——门禁基线不变。"""
    target = tmp_path / "app.py"
    target.write_text("x = 1\n", encoding="utf-8")
    pre_write._get_read_files(SID).clear()
    result = _pre_patch(str(target))
    assert result is not None and result["action"] == "block"  # 期望: 零证据仍拦
