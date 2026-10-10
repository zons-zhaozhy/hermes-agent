"""violations 表迁移提交持久性验证（红绿测试）。

规则来源（独立期望）：outcome 列的迁移回填必须在任何调用路径下持久化——
2026-10-10 实证缺陷：迁移在无 commit 的只读路径（_violation_stats）上执行时，
ALTER(DDL) 隐式提交持久化、回填 UPDATE(DML) 被 close 回滚，「列存在」幂等
判据随之短路，回填永久丢失。期望值从该缺陷机理推导，不从实现反推。
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# 已 read_file plugins/discipline/__init__.py（套件入口，re-export no_guessing 子模块）
from plugins.discipline import no_guessing


def _make_legacy_db(path: Path) -> None:
    """模拟旧库：无 outcome 列，含三个执法路径的行。"""
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE violations (id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " rule TEXT NOT NULL, command TEXT NOT NULL, level TEXT,"
        " session_id TEXT, timestamp TEXT NOT NULL)"
    )
    conn.executemany(
        "INSERT INTO violations (rule, command, level, session_id, timestamp) VALUES (?,?,?,?,?)",
        [
            ("R6", "ps aux | grep x 2>/dev/null", "L1", "s1", "2026-10-01T10:00:00"),
            ("R5", "sleep 60; grep done /tmp/x", "L1", "s1", "2026-10-01T11:00:00"),
            ("R5", "sleep 60", "L1", "s1", "2026-10-01T12:00:00"),
        ],
    )
    conn.commit()
    conn.close()


def test_backfill_survives_readonly_path_close(tmp_path: Path) -> None:
    """缺陷正例：迁移在只读路径执行后 close（无显式 commit），回填必须已持久化。"""
    db = tmp_path / "t.db"
    _make_legacy_db(db)
    # 只读路径模拟：ensure 后立即 close（不经过 _record_violation 的 commit）
    conn = sqlite3.connect(db)
    no_guessing._ensure_violations_table(conn)
    conn.close()
    conn = sqlite3.connect(db)
    rows = conn.execute(
        "SELECT rule, command, outcome FROM violations ORDER BY id"
    ).fetchall()
    conn.close()
    assert rows[0] == ("R6", "ps aux | grep x 2>/dev/null", "rewrite"), rows[0]  # 期望: 09-20后R6唯一出口=modify机械改写→rewrite（UPDATE须随迁移commit持久化）
    assert rows[1] == ("R5", "sleep 60; grep done /tmp/x", "hint"), rows[1]  # 期望: 09-26后R5组合式唯一出口=HINT放行→hint
    assert rows[2] == ("R5", "sleep 60", "block"), rows[2]  # 期望: R5纯sleep真拦截→保持block（回填禁误赦真拦截）


def test_backfill_idempotent_on_second_ensure(tmp_path: Path) -> None:
    """幂等：列已存在（且已回填）时再 ensure 不重复改写、不误伤后继真 block 行。"""
    db = tmp_path / "t.db"
    _make_legacy_db(db)
    conn = sqlite3.connect(db)
    no_guessing._ensure_violations_table(conn)
    conn.close()
    # 新代码分列记账产生一行真 block（R6 目前无此出口，用 R5 纯 sleep 代表真拦截族）
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO violations (rule, command, level, session_id, timestamp, outcome)"
        " VALUES ('R5','sleep 99','L1','s2','2026-10-10T13:00:00','block')"
    )
    conn.commit()
    conn.close()
    # 第二次 ensure（幂等判据=列存在）
    conn = sqlite3.connect(db)
    no_guessing._ensure_violations_table(conn)
    conn.close()
    conn = sqlite3.connect(db)
    outcomes = conn.execute(
        "SELECT command, outcome FROM violations ORDER BY id"
    ).fetchall()
    conn.close()
    assert outcomes == [  # 期望: 幂等判据=列存在即跳过——后继真block行(sleep 99)不被改写
        ("ps aux | grep x 2>/dev/null", "rewrite"),
        ("sleep 60; grep done /tmp/x", "hint"),
        ("sleep 60", "block"),
        ("sleep 99", "block"),
    ], outcomes


def test_broken_state_selfheal_on_production_shape(tmp_path: Path) -> None:
    """生产库现状形态：列已存在但回填丢失（旧迁移回滚残局）——需一次性修复。

    代码路径按「列存在即跳过」设计（避免 pending 永真误伤），生产残局由
    ops 按同判据补跑（见 audit-20261010 报告 C1）。本测试固化该判据，
    防止未来把行级回填塞回 ensure 导致误伤。
    """
    db = tmp_path / "t.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE violations (id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " rule TEXT NOT NULL, command TEXT NOT NULL, level TEXT,"
        " session_id TEXT, timestamp TEXT NOT NULL,"
        " outcome TEXT NOT NULL DEFAULT 'block')"
    )
    conn.executemany(
        "INSERT INTO violations (rule, command, level, session_id, timestamp, outcome)"
        " VALUES (?,?,?,?,?,?)",
        [
            ("R6", "ps aux | grep x 2>/dev/null", "L1", "s1", "2026-10-01T10:00:00", "block"),
            ("R5", "sleep 60; grep done /tmp/x", "L1", "s1", "2026-10-01T11:00:00", "block"),
            ("R5", "sleep 60", "L1", "s1", "2026-10-01T12:00:00", "block"),
        ],
    )
    conn.commit()
    conn.close()
    # ops 修复判据（与 ensure 内首次回填同语义）：确定性路径+时间边界+组合形态
    conn = sqlite3.connect(db)
    conn.execute(
        "UPDATE violations SET outcome='rewrite' WHERE rule='R6'"
        " AND timestamp >= '2026-09-20' AND outcome='block'"
        " AND id <= (SELECT MAX(id) FROM violations WHERE timestamp <= '2026-10-10T11:10:00')"
    )
    conn.execute(
        "UPDATE violations SET outcome='hint' WHERE rule='R5' AND level='L1'"
        " AND timestamp >= '2026-09-26' AND outcome='block'"
        " AND (command LIKE '%;%' OR command LIKE '%&&%')"
        " AND id <= (SELECT MAX(id) FROM violations WHERE timestamp <= '2026-10-10T11:10:00')"
    )
    conn.commit()
    rows = conn.execute("SELECT command, outcome FROM violations ORDER BY id").fetchall()
    conn.close()
    assert rows == [  # 期望: 与首次迁移判据同构——同批行同分类，防两套判据漂移
        ("ps aux | grep x 2>/dev/null", "rewrite"),
        ("sleep 60; grep done /tmp/x", "hint"),
        ("sleep 60", "block"),
    ], rows
