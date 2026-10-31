"""block_escalation error/blocked 车道分流回归（10-31 deputy-patrol 误杀链）。

缺陷: _on_post_tool_call 把 status=error（执行失败）与 status=blocked（护栏
拦截）同权计入跨会话升级指纹——周期 cron 任务（deputy-patrol 每 10 分钟）
重跑同一巡检命令，命令环境性报错跨班叠加被误判「拦截循环」，班次被升级
终止（10-31 04:13/10:02/10:49 三杀，block_escalation.db sample 实证=纯读
git 查询）。且 terminal 工具体护栏拒绝（_blocked_json 含 error 键）被
model_tools observer 复播为 error——按 status 字面分流会漏判真拦截。

期望值独立推导（源自「杀班权仅限护栏拦截循环」不变量）:
- 纯执行失败 ×2 → count 计数但 _escalated 空（观测告警不终止）
- status=error 但包体 status=blocked（terminal 护栏拒绝形态）×2 → 升级
- status=blocked ×2 → 照旧升级（原行为不回退）
- 旧 schema 库迁移幂等，历史行清空，二次执行不抛
- git rev-parse 复合查询判只读豁免
- 同指纹 blocked 史不因后续 error 降级
"""

from __future__ import annotations

import json
import logging
import pathlib
import sqlite3

import pytest

import plugins.block_escalation as be

logger = logging.getLogger(__name__)


def _close(conn: sqlite3.Connection) -> None:
    try:
        conn.close()
    except Exception as close_err:  # pragma: no cover
        logger.warning("close 失败: %s", close_err, exc_info=True)


def _setup(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(be, "_db_path", tmp_path / "be.db")
    monkeypatch.setattr(be, "_escalated", {})
    conn = sqlite3.connect(str(tmp_path / "be.db"))
    try:
        be._ensure_schema(conn)
    finally:
        _close(conn)


def _row_count_kind(tmp_path: pathlib.Path, fingerprint: str) -> tuple[int, str] | None:
    conn = sqlite3.connect(str(tmp_path / "be.db"))
    try:
        row = conn.execute(
            "SELECT count, kind FROM block_streaks WHERE fingerprint = ?",
            (fingerprint,)).fetchone()
        return (row[0], row[1]) if row else None
    finally:
        _close(conn)


_WRITE_ARGS = {"path": "/tmp/be_lane_target.md", "content": "x"}


def _err_call(result_json: str) -> None:
    be._on_post_tool_call(
        tool_name="write_file", status="error", error_type="tool_error",
        args=dict(_WRITE_ARGS), session_id="s-lane", result=result_json)


def test_error_lane_counts_without_escalation(tmp_path: pathlib.Path,
                                              monkeypatch: pytest.MonkeyPatch) -> None:
    _setup(tmp_path, monkeypatch)
    _err_call(json.dumps({"output": "", "exit_code": 1, "error": "boom"}))
    _err_call(json.dumps({"output": "", "exit_code": 1, "error": "boom again"}))
    fp_key = be._fp_key(be._intent_fingerprint("write_file", _WRITE_ARGS))
    assert _row_count_kind(tmp_path, fp_key) == (2, "error")  # 期望: (2,'error') — 执行失败计数但落 error 车道
    assert not be._escalated.get("s-lane")  # 期望: 空 — error 车道不置升级标记(deputy-patrol 误杀链根除)


def test_body_blocked_goes_blocked_lane(tmp_path: pathlib.Path,
                                        monkeypatch: pytest.MonkeyPatch) -> None:
    _setup(tmp_path, monkeypatch)
    blocked_envelope = json.dumps(
        {"output": "", "exit_code": 1, "error": "self_repo_block", "status": "blocked"})
    _err_call(blocked_envelope)
    _err_call(blocked_envelope)
    assert be._escalated.get("s-lane")  # 期望: 非空 — terminal 工具体护栏拒绝(包体 status=blocked)不得因复播 status=error 逃逸升级


def test_status_blocked_still_escalates(tmp_path: pathlib.Path,
                                        monkeypatch: pytest.MonkeyPatch) -> None:
    _setup(tmp_path, monkeypatch)
    for _ in range(2):
        be._on_post_tool_call(
            tool_name="write_file", status="blocked", error_type="guardrail_block",
            args=dict(_WRITE_ARGS), session_id="s-lane", result="{}")
    assert be._escalated.get("s-lane")  # 期望: 非空 — 显式 blocked 复播照旧升级(原行为不回退)


def test_kind_migration_idempotent_and_clears_history(
        tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db = str(tmp_path / "be.db")
    conn = sqlite3.connect(db)
    try:
        conn.execute(
            "CREATE TABLE block_streaks (fingerprint TEXT PRIMARY KEY, count INTEGER NOT NULL"
            " DEFAULT 0, last_ts REAL NOT NULL, tools TEXT NOT NULL DEFAULT '', sample TEXT)")
        conn.execute(
            "INSERT INTO block_streaks (fingerprint, count, last_ts, tools, sample)"
            " VALUES ('legacy_fp', 3, 9999999999.0, 'write_file', 'legacy')")
        conn.commit()
    finally:
        _close(conn)
    monkeypatch.setattr(be, "_db_path", tmp_path / "be.db")
    for _ in range(2):  # 迁移须幂等：第二次不抛 duplicate column
        conn = sqlite3.connect(db)
        try:
            be._ensure_schema(conn)
        finally:
            _close(conn)
    conn = sqlite3.connect(db)
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(block_streaks)")}
        remaining = conn.execute("SELECT COUNT(*) FROM block_streaks").fetchone()[0]
    finally:
        _close(conn)
    assert "kind" in cols  # 期望: 含 kind — 迁移加列
    assert remaining == 0  # 期望: 0 — kind 不可判定的历史行清空(旧行保有升级权=误杀残留)


def test_rev_parse_compound_query_is_readonly() -> None:
    cmd = ("cd /Users/stan/code/ai/cnb/ssd_test && git log --oneline -1 f167ebc"
           " && git rev-parse HEAD origin/main && git status --short")
    assert be._command_is_readonly(cmd)  # 期望: True — 10-48 波被计入的复合查询全段只读应豁免


def test_blocked_kind_not_downgraded_by_error(tmp_path: pathlib.Path,
                                              monkeypatch: pytest.MonkeyPatch) -> None:
    _setup(tmp_path, monkeypatch)
    fp_key = be._fp_key(be._intent_fingerprint("write_file", _WRITE_ARGS))
    be._note_blocked("write_file", dict(_WRITE_ARGS), session_id="s-lane", kind="blocked")
    be._note_blocked("write_file", dict(_WRITE_ARGS), session_id="s-lane", kind="error")
    assert _row_count_kind(tmp_path, fp_key) == (2, "blocked")  # 期望: (2,'blocked') — 同指纹已有拦截史,error 挣扎不弱化升级权
