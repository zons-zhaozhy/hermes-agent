"""block_escalation 指纹计数双缺陷回归（2026-10-05 v181 报告 #9）。

缺陷①跨工具清不掉：terminal 被拦指纹=命令中多路径元组，write_file/patch
成功指纹=(path,) 单元组——键不同，成功落地后 terminal streak 残留，30 分钟
窗内持续威胁假升级。实测 block_escalation.db 残留 count=4（patch,write_file）。

缺陷②只读意图计 streak：纯 SELECT/stat 只读命令被安全护栏拦截（如 DBSafety
前置 schema 检查）时也被计为写意图指纹——读意图无硬闯危害，计数即噪声。

期望值独立推导：
- 只读命令（sqlite3 "SELECT ..."、ls、stat）被拦 → streak 行数不变（不计数）
- 写命令（cat > file、rm、git push）被拦 → 计数
- 同一目标路径经任意工具成功落地 → 与该路径有交集的所有 streak 行清零
"""

import logging
import pathlib
import sqlite3

import pytest

import plugins.block_escalation as be


def _setup(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(be, "_db_path", tmp_path / "be.db")
    monkeypatch.setattr(be, "_escalated", set())
    conn = sqlite3.connect(str(tmp_path / "be.db"))
    try:
        be._ensure_schema(conn)
    finally:
        try:
            conn.close()
        except Exception as close_err:  # pragma: no cover - 关闭失败不掩原始异常
            logging.getLogger(__name__).warning("close 失败: %s", close_err)


def _rows(tmp_path: pathlib.Path) -> list:
    conn = sqlite3.connect(str(tmp_path / "be.db"))
    try:
        return conn.execute(
            "SELECT fingerprint, count FROM block_streaks").fetchall()
    finally:
        conn.close()


def test_readonly_select_block_not_counted(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    be._note_blocked("terminal", {
        "command": 'sqlite3 ~/.hermes/cron/executions.db "SELECT id, job_id '
                   'FROM executions WHERE job_id = \'af1\'" 2>&1'})
    # 期望: 纯 SELECT 只读被拦不计入升级计数（读意图零硬闯危害）
    assert _rows(tmp_path) == []


def test_readonly_stat_ls_not_counted(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    be._note_blocked("terminal", {
        "command": "stat -f '%m %N' ~/Documents/aml-bid/PIPELINE-STATE.md && "
                   "ls /tmp && find ~/code -name '*.py'"})
    # 期望: stat/ls/find 只读组合被拦不计数
    assert _rows(tmp_path) == []


def test_write_command_block_counted(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    be._note_blocked("terminal", {
        "command": "cat > /tmp/x/report.md << 'EOF'\nhello\nEOF"})
    # 期望: 写重定向命令被拦 → 计数=1（防线保持）
    assert [(r[1]) for r in _rows(tmp_path)] == [1]


def test_cross_tool_success_clears_streak(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    # 写意图经 terminal 被拦（指纹=命令中多路径元组）
    be._note_blocked("terminal", {
        "command": "cat > /Users/stan/KB/report.md << 'EOF'\nhello\nEOF"
                   " && sqlite3 ~/.hermes/cron/executions.db 'SELECT 1'"})
    assert len(_rows(tmp_path)) == 1
    # 同一目标路径经 write_file 成功落地（指纹=(path,) 单元组，键不同）
    be._on_post_tool_call(tool_name="write_file", status="ok", args={
        "path": "/Users/stan/KB/report.md", "content": "hello"})
    # 期望: 意图已成功落地 → 跨工具 streak 清零（惩罚作废）
    assert _rows(tmp_path) == []


def test_unrelated_success_keeps_streak(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    be._note_blocked("terminal", {
        "command": "cat > /tmp/other/target.md << 'EOF'\nhi\nEOF"})
    be._on_post_tool_call(tool_name="write_file", status="ok", args={
        "path": "/tmp/different/file.md", "content": "x"})
    # 期望: 无路径交集的成功不清该 streak（不同意图各自计数）
    assert len(_rows(tmp_path)) == 1 and _rows(tmp_path)[0][1] == 1


def test_escalated_flag_cleared_on_cross_tool_success(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch)
    be._note_blocked("terminal", {
        "command": "cat > /tmp/x/a.md << 'EOF'\n1\nEOF"})
    be._note_blocked("terminal", {
        "command": "echo hi > /tmp/x/a.md"})
    # 期望: 同意图第 2 次被拦 → 升级标记置位
    assert any("/tmp/x/a.md" in fp for fp in be._escalated)
    be._on_post_tool_call(tool_name="write_file", status="ok", args={
        "path": "/tmp/x/a.md", "content": "done"})
    # 期望: 成功落地撤销升级标记（跨工具同样生效）
    assert not be._escalated
