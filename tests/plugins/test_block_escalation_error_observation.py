"""block_escalation error 观测层回归（反方审查 D1）。

缺陷: _on_post_tool_call 只处理 blocked/ok 两分支——status=error 的调用
（读工具放行后的无进展失败循环）零观测零升级；且 _intent_fingerprint 对
read_file/search_files 恒返回 ()，读通道结构性失明。

期望值独立推导（源自「读工具零拦截，失败循环须可观测」不变量）：
- 同一 read_file 路径连续 error → streak 计数（DB 落一行 count 递增）
- 该路径随后成功（ok）→ streak 清零
- read_file 指纹非空元组（参与计数）
"""

from __future__ import annotations

import logging
import pathlib
import sqlite3

import pytest

import plugins.block_escalation as be

logger = logging.getLogger(__name__)


def _setup(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(be, "_db_path", tmp_path / "be.db")
    monkeypatch.setattr(be, "_escalated", {})
    conn = sqlite3.connect(str(tmp_path / "be.db"))
    try:
        be._ensure_schema(conn)
    finally:
        try:
            conn.close()
        except Exception as close_err:  # pragma: no cover
            logger.warning("close 失败: %s", close_err, exc_info=True)


def _count(tmp_path: pathlib.Path, fingerprint: str) -> int:
    conn = sqlite3.connect(str(tmp_path / "be.db"))
    try:
        row = conn.execute(
            "SELECT count FROM block_streaks WHERE fingerprint = ?",
            (fingerprint,)).fetchone()
        return row[0] if row else 0
    finally:
        try:
            conn.close()
        except Exception as close_err:  # pragma: no cover
            logger.warning("close 失败: %s", close_err, exc_info=True)


def test_read_file_fingerprint_is_nonempty(tmp_path: pathlib.Path) -> None:
    # 期望: read_file 指纹=path 键控元组，不再恒 ()
    target = str(tmp_path / "sample.txt")
    fp = be._intent_fingerprint("read_file", {"path": target})
    assert fp == (f"read_file:{target}",)  # 期望: 工具名+路径的稳定指纹


def test_error_status_counts_into_streak(tmp_path: pathlib.Path,
                                         monkeypatch: pytest.MonkeyPatch) -> None:
    # 期望: 同一 read_file 连续 error 两次 → count=2（观测层捕获失败循环）
    _setup(tmp_path, monkeypatch)
    target = str(tmp_path / "protected.txt")
    for _ in range(2):
        be._on_post_tool_call(
            tool_name="read_file", status="error",
            args={"path": target}, session_id="s1")
    assert _count(tmp_path, f"read_file:{target}") == 2  # 期望: error 循环被计数


def test_ok_clears_error_streak(tmp_path: pathlib.Path,
                                monkeypatch: pytest.MonkeyPatch) -> None:
    # 期望: 失败后同指纹成功 → streak 清零（成功落地作废）
    _setup(tmp_path, monkeypatch)
    target = str(tmp_path / "ok.txt")
    be._on_post_tool_call(
        tool_name="read_file", status="error",
        args={"path": target}, session_id="s1")
    be._on_post_tool_call(
        tool_name="read_file", status="ok",
        args={"path": target}, session_id="s1")
    assert _count(tmp_path, f"read_file:{target}") == 0  # 期望: 成功清账
