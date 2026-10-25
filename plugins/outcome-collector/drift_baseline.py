#!/usr/bin/env python3
"""agent 行为漂移基线——会话级四键快照与基线对照（LLM 行为退化检测）。

机制移植自 OntoX af_llm_drift_baseline.json（每 agent 五键快照重跑对照）。
适配差异：hermes 会话无 agent 名维度且 tool_outcomes 无 tokens 列，四键为
steps(total_calls) / tools(distinct tool_name) / error_rate / avg_duration_ms。
数据源=session_summaries（outcome-collector on_session_end 已持续写入）。

用法（对照模式——上游同步/模型切换前后）：
    python drift_baseline.py --snapshot              # 固化当前窗口会话为基线
    python drift_baseline.py --check                 # 快照后新增会话 vs 基线对照
    python drift_baseline.py --self-test             # 判定器负例自测

判定（中位数对照，宽阈值防会话长短方差误报）：
    steps/tools 中位数 > 基线 1.5x，或 error_rate > 基线 2x 且绝对值 ≥ 0.2
    → 对应键标记 drifted；任一键 drifted → rc=1

Contract:
    Preconditions: outcomes.db 存在且 session_summaries 有数据（空库显式报错）
    Postconditions: --snapshot 写 baseline 文件；--check 只读对照不改库；
        基线文件缺失时 --check 显式报错（禁静默当无漂移）
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path
from statistics import median

logger = logging.getLogger(__name__)

STEPS_RATIO = 1.5
TOOLS_RATIO = 1.5
ERROR_RATE_RATIO = 2.0
ERROR_RATE_ABS = 0.2


def _db_path() -> Path:
    home = os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
    return Path(home) / "outcomes.db"


def _baseline_path() -> Path:
    home = os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
    p = Path(home) / "outcomes" / "agent_drift_baseline.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _close_logged(conn: sqlite3.Connection, db: Path) -> None:
    """close 失败只记 warning 不上抛（防掩盖原始异常）。

    Contract:
        Preconditions: conn 为 sqlite3.Connection
        Postconditions: close 异常被吞并记 warning 日志
    """
    try:
        conn.close()
    except Exception:
        logger.warning("sqlite close failed: %s", db, exc_info=True)


def _median_round(values: list[float]) -> float:
    """中位数，空列表返回 0.0（空时 0 是显式约定）。

    Contract:
        Preconditions: 无
        Postconditions: 返回 float 中位数（四舍五入 2 位）；空列表返回 0.0
    """
    return round(median(values), 2) if values else 0.0


def _keys_from_rows(rows: list[tuple]) -> dict[str, float]:
    """把 session_summaries 行集合算成四键中位数结构。

    Contract:
        Preconditions: rows 元素为 (total_calls, error_count, tool_breakdown, avg_ms 可空)
        Postconditions: 返回四键 dict；空 rows 返回全 0（调用方先显式报错）
    """
    steps = [float(r[0]) for r in rows if r[0]]
    tools = []
    for r in rows:
        try:
            breakdown = json.loads(r[2]) if r[2] else {}
            tools.append(float(len(breakdown)))
        except (json.JSONDecodeError, TypeError):
            tools.append(0.0)
    err_rates = [float(r[1]) / float(r[0]) for r in rows if r[0]]
    durations = [float(r[3]) for r in rows if r[3]]
    return {
        "sessions": float(len(rows)),
        "steps_median": _median_round(steps),
        "tools_median": _median_round(tools),
        "error_rate_median": _median_round(err_rates),
        "avg_duration_ms_median": _median_round(durations),
    }


def _fetch_sessions(conn: sqlite3.Connection, after_iso: str | None) -> list[tuple]:
    """取会话行；after_iso 非空时只取其后有更新的会话。

    Contract:
        Preconditions: session_summaries 表存在
        Postconditions: 返回行列表；表不存在时抛 sqlite3.OperationalError
    """
    sql = (
        "SELECT total_calls, error_count, tool_breakdown, "
        "  (SELECT CAST(AVG(duration_ms) AS REAL) FROM tool_outcomes t "
        "   WHERE t.session_id = s.session_id) "
        "FROM session_summaries s"
    )
    if after_iso:
        sql += " WHERE s.updated_at > ?"
        return conn.execute(sql, (after_iso,)).fetchall()
    return conn.execute(sql).fetchall()


def _snapshot(baseline_path: Path, db: Path) -> int:
    """固化当前全部会话为基线快照。

    Contract:
        Preconditions: db 存在且有会话数据
        Postconditions: 基线文件写入（含 generated_at 与四键）；空数据 rc=1
    """
    if not db.exists():
        print(f"[drift] db 不存在: {db}", file=sys.stderr)
        return 1
    conn = sqlite3.connect(str(db))
    try:
        rows = _fetch_sessions(conn, None)
    finally:
        _close_logged(conn, db)
    if not rows:
        print("[drift] session_summaries 无数据——先跑一段时间再快照", file=sys.stderr)
        return 1
    payload = {"generated_at": datetime.now(UTC).isoformat(), "keys": _keys_from_rows(rows)}
    baseline_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[drift] 基线已固化: {baseline_path}")
    print(json.dumps(payload["keys"], ensure_ascii=False))
    return 0


def _check(baseline_path: Path, db: Path) -> int:
    """快照后新增会话 vs 基线对照，输出各键判定。

    Contract:
        Preconditions: 基线文件存在且可解析；db 存在
        Postconditions: 任一键 drifted 返回 1，否则 0；无新增会话 rc=2 提示
    """
    if not baseline_path.exists():
        print(f"[drift] 基线缺失——先 --snapshot（{baseline_path}）", file=sys.stderr)
        return 1
    try:
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[drift] 基线文件损坏: {exc}", file=sys.stderr)
        return 1
    since = str(baseline.get("generated_at", ""))
    base_keys = baseline.get("keys", {})
    conn = sqlite3.connect(str(db))
    try:
        rows = _fetch_sessions(conn, since)
    finally:
        _close_logged(conn, db)
    if not rows:
        print("[drift] 快照后无新增会话，无对照对象（rc=2）")
        return 2
    new_keys = _keys_from_rows(rows)
    drifted = _verdict(base_keys, new_keys)
    print(f"[drift] 对照会话数 {int(new_keys['sessions'])}（基线 {int(base_keys.get('sessions', 0))}）")
    for key in ("steps_median", "tools_median", "error_rate_median", "avg_duration_ms_median"):
        mark = "DRIFTED" if key in drifted else "ok"
        print(f"  {key}: 基线 {base_keys.get(key)} → 新窗口 {new_keys[key]}  [{mark}]")
    if drifted:
        print(f"[drift] 行为漂移键: {sorted(drifted)}——检查上游同步/模型切换/提示词改动")
        return 1
    print("[drift] 无漂移")
    return 0


def _verdict(base: dict, new: dict) -> list[str]:
    """判定漂移键集合（中位数倍率阈值）。

    Contract:
        Preconditions: base/new 为四键 dict
        Postconditions: 返回 drifted 键名列表（可为空）；基线值为 0 的键跳过
    """
    out: list[str] = []
    for key, ratio in (("steps_median", STEPS_RATIO), ("tools_median", TOOLS_RATIO)):
        b = float(base.get(key, 0) or 0)
        n = float(new.get(key, 0) or 0)
        if b > 0 and n > b * ratio:
            out.append(key)
    b_err = float(base.get("error_rate_median", 0) or 0)
    n_err = float(new.get("error_rate_median", 0) or 0)
    if n_err > ERROR_RATE_ABS and n_err > b_err * ERROR_RATE_RATIO:
        out.append("error_rate_median")
    return out


def _negative_self_check() -> int:
    """负例自测：判定器对漂移/正常两形态必须正确分类。

    Contract:
        Preconditions: 无
        Postconditions: 全部断言通过返回 0
    """
    base = {"steps_median": 10.0, "tools_median": 4.0, "error_rate_median": 0.1, "avg_duration_ms_median": 500.0}
    same = dict(base)
    # 期望: 相同四键零漂移
    assert _verdict(base, same) == []  # 期望: 无偏移不判漂移
    hot = dict(base, steps_median=20.0)
    # 期望: steps 超基线 1.5x(20>15) 判漂移
    assert _verdict(base, hot) == ["steps_median"]  # 期望: 倍率越限命中
    err_spike = dict(base, error_rate_median=0.5)
    # 期望: error_rate 绝对值≥0.2 且 >0.1*2 判漂移
    assert _verdict(base, err_spike) == ["error_rate_median"]  # 期望: 错误率双条件命中
    tiny = dict(base, error_rate_median=0.15)
    # 期望: 0.15<0.2 绝对阈值不判(防低基线误报)
    assert _verdict(base, tiny) == []  # 期望: 绝对阈值挡住低量级误报
    zero_base = {"steps_median": 0.0, "tools_median": 0.0, "error_rate_median": 0.0, "avg_duration_ms_median": 0.0}
    # 期望: 零基线键跳过不判
    assert _verdict(zero_base, dict(base, steps_median=100.0)) == []  # 期望: 基线为0无可比性
    print("[drift] self-test 全部通过")
    return 0


def run_cli(argv: list[str] | None = None) -> int:
    """CLI 入口：快照/对照/负例自测三模式。"""
    parser = argparse.ArgumentParser(description="agent 行为漂移基线（会话四键对照）")
    parser.add_argument("--snapshot", action="store_true", help="固化当前窗口为基线")
    parser.add_argument("--check", action="store_true", help="快照后新增会话 vs 基线")
    parser.add_argument("--db", type=Path, default=_db_path())
    parser.add_argument("--baseline", type=Path, default=_baseline_path())
    parser.add_argument("--self-test", action="store_true", help="负例自测")
    args = parser.parse_args(argv)
    if args.self_test:
        return _negative_self_check()
    if args.snapshot:
        return _snapshot(args.baseline, args.db)
    if args.check:
        return _check(args.baseline, args.db)
    parser.error("须指定 --snapshot / --check / --self-test 之一")
    return 2


if __name__ == "__main__":
    sys.exit(run_cli())
