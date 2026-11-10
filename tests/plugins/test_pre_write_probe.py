"""pre_write 已读证据落盘探针验证（红绿测试）。

规则来源（独立期望）：先读后写门禁的证据在 pre/post 钩子分属不同进程时必须仍可
传递——post 侧记账双写磁盘 marker，pre 侧内存或磁盘任一命中即放行；marker 缺失、
超 TTL（24h）、格式错一律 fail-closed（仍拦）。期望值从上述规则推导，不从实现反推。
"""

from __future__ import annotations

import json
import os
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


def test_stale_markers_are_pruned_on_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """过期 marker 会被清掉——一个 home 不能无限攒死文件。

    规则来源：marker 超 TTL 即失效，失效文件没有读取价值，留着只拖慢目录扫描
    （实测 ~/.hermes/cache 攒到 2233 个）。清扫只在写路径上发生，且必须留活口：
    本次刚写的 marker、以及仍在 TTL 内的其它 session 的 marker 都不能被误删。
    """
    cache = tmp_path / "probe_cache"
    cache.mkdir()
    old = cache / "pw_read_probe_00000000deadbeef.json"
    old.write_text(json.dumps({"sid": "gone", "ts": 0, "paths": []}), encoding="utf-8")
    stale_mtime = time.time() - (pre_write._PW_PROBE_TTL + 3600)
    os.utime(old, (stale_mtime, stale_mtime))
    fresh = cache / "pw_read_probe_11111111feedface.json"
    fresh.write_text(json.dumps({"sid": "live", "ts": time.time(), "paths": []}), encoding="utf-8")
    monkeypatch.setattr(pre_write, "_pw_last_prune", 0.0)  # 绕过进程内节流

    _post_read(str(tmp_path / "app.py"))

    assert not old.exists(), "超过 TTL 的 marker 应被删除"
    assert fresh.exists(), "仍有效的 marker 必须保留"
    assert pre_write._read_marker(SID).exists(), "本次写入的 marker 必须保留"
