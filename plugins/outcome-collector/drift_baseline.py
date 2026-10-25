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

判定 v2（两层，反方审查推翻纯中位数倍率方案后的重设计）：
    第一层 中位数对照：steps/tools > 基线 1.5x、error_rate > 2x 且绝对值 ≥ 0.2、
    avg_duration_ms > 2x → 对应键 drifted
    第二层 会话级离群：窗口内单会话 steps 超基线 steps_p95 的会话占比 > 20%
    → steps 键 drifted（附离群会话证据）——消除短窗口中位数抽样波动误报
    最小样本门：窗口 < 20 会话 → rc=3「样本不足」，与 rc=0「无漂移」显式区分

Contract:
    Preconditions: outcomes.db 存在且 session_summaries 有数据（空库显式报错）
    Postconditions: --snapshot 写 v2 基线文件；--check 只读对照不改库；
        基线缺失/损坏/格式过旧时 --check 显式报错（禁静默当无漂移）
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
from statistics import median, quantiles

logger = logging.getLogger(__name__)

FORMAT_VERSION = 2
STEPS_RATIO = 1.5
TOOLS_RATIO = 1.5
ERROR_RATE_RATIO = 2.0
ERROR_RATE_ABS = 0.2
DURATION_RATIO = 2.0
MIN_WINDOW_SESSIONS = 20
OUTLIER_FRAC = 0.2
KEY_ORDER = ("steps_median", "tools_median", "error_rate_median", "avg_duration_ms_median")


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


def _steps_percentile(steps: list[float], pct: int) -> float:
    """会话 steps 分位数（基线分布刻画，会话级离群判定输入）。

    Contract:
        Preconditions: pct in (95, 99)
        Postconditions: 样本 ≥2 时返回分位值；样本不足时退化为 max（保守上界）
    """
    if len(steps) < 2:
        return round(max(steps), 2) if steps else 0.0
    qs = quantiles(steps, n=100)
    idx = min(pct, 99) - 1  # quantiles(n=100) 产出 99 个切点，p95→idx94
    return round(qs[idx], 2)


def _keys_from_rows(rows: list[tuple]) -> dict:
    """把 session_summaries 行集合算成四键结构（含逐会话 steps 与解析失败数）。

    Contract:
        Preconditions: rows 元素为 (total_calls, error_count, tool_breakdown, avg_ms 可空)
        Postconditions: 返回含四键中位数/steps 序列/parse_failures 的 dict；
            tool_breakdown 解析失败计数并记 warning（禁静默按 0）
    """
    steps = [float(r[0]) for r in rows if r[0]]
    tools: list[float] = []
    parse_failures = 0
    for r in rows:
        try:
            breakdown = json.loads(r[2]) if r[2] else {}
            tools.append(float(len(breakdown)))
        except (json.JSONDecodeError, TypeError):
            parse_failures += 1
            logger.warning("tool_breakdown 解析失败按缺失计: %r", r[2])
    err_rates = [float(r[1]) / float(r[0]) for r in rows if r[0]]
    durations = [float(r[3]) for r in rows if r[3]]
    return {
        "sessions": float(len(rows)),
        "steps_median": _median_round(steps),
        "tools_median": _median_round(tools),
        "error_rate_median": _median_round(err_rates),
        "avg_duration_ms_median": _median_round(durations),
        "steps_series": steps,
        "parse_failures": parse_failures,
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
    """固化当前全部会话为 v2 基线快照（含 steps 分位数）。

    Contract:
        Preconditions: db 存在且有会话数据
        Postconditions: 基线文件写入（format_version=2 + 四键 + p95/p99 + 会话数）；
            空数据 rc=1；parse_failures>0 打印警示行
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
    keys = _keys_from_rows(rows)
    steps = keys.pop("steps_series")
    parse_failures = keys.pop("parse_failures")
    payload = {
        "format_version": FORMAT_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "keys": keys,
        "steps_p95": _steps_percentile(steps, 95),
        "steps_p99": _steps_percentile(steps, 99),
        "baseline_sessions": len(rows),
    }
    baseline_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[drift] 基线已固化: {baseline_path}（v2，会话 {len(rows)}）")
    print(json.dumps({k: payload[k] for k in ("keys", "steps_p95", "steps_p99")}, ensure_ascii=False))
    if parse_failures:
        print(f"[drift] 警示: {parse_failures} 个会话 tool_breakdown 解析失败按缺失计", file=sys.stderr)
    return 0


def _load_baseline(baseline_path: Path) -> dict | None:
    """读基线并做格式版本守卫（v1/损坏文件显式报错禁静默）。

    Contract:
        Preconditions: baseline_path 存在
        Postconditions: 合法 v2 返回 dict；否则打印原因返回 None
    """
    try:
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[drift] 基线文件损坏: {exc}", file=sys.stderr)
        return None
    if baseline.get("format_version") != FORMAT_VERSION:
        print(
            f"[drift] 基线格式过旧（v{baseline.get('format_version', 1)}≠v{FORMAT_VERSION}）"
            f"——重跑 --snapshot 生成 v2（含分位数），禁沿用旧格式静默对照",
            file=sys.stderr,
        )
        return None
    return baseline


def _outlier_evidence(steps_series: list[float], steps_p95: float) -> tuple[float, int]:
    """会话级离群：窗口内 steps 超基线 p95 的会话数与占比。

    Contract:
        Preconditions: steps_p95 ≥ 0
        Postconditions: 返回 (占比, 离群会话数)；p95=0（基线退化）时占比 0（不判）
    """
    if steps_p95 <= 0:
        return 0.0, 0
    outliers = sum(1 for s in steps_series if s > steps_p95)
    frac = outliers / len(steps_series) if steps_series else 0.0
    return frac, outliers


def _report(base_keys: dict, new_keys: dict, drifted: list[str], frac: float, outliers: int, n: int) -> int:
    """打印四键对照结果并返回 rc。

    Contract:
        Preconditions: drifted ⊆ KEY_ORDER；n ≥ MIN_WINDOW_SESSIONS
        Postconditions: drifted 非空返回 1，否则 0
    """
    session_level = (
        f"（会话级证据: {outliers}/{n}={frac:.0%} 超基线 p95，占比>{OUTLIER_FRAC:.0%}）"
        if "steps_median" in drifted and frac > OUTLIER_FRAC
        else ""
    )
    for key in KEY_ORDER:
        mark = "DRIFTED" if key in drifted else "ok"
        extra = session_level if key == "steps_median" and session_level else ""
        print(f"  {key}: 基线 {base_keys.get(key)} → 新窗口 {new_keys[key]}  [{mark}]{extra}")
    if drifted:
        print(f"[drift] 行为漂移键: {sorted(drifted)}——检查上游同步/模型切换/提示词改动")
        return 1
    print(f"[drift] 无漂移（离群会话 {outliers}/{n}={frac:.0%}，阈 {OUTLIER_FRAC:.0%}）")
    return 0


def _check(baseline_path: Path, db: Path) -> int:
    """快照后新增会话 vs 基线两层对照。

    Contract:
        Preconditions: 基线文件存在且为 v2；db 存在
        Postconditions: 任一键 drifted 返回 1；无新增 rc=2；
            窗口 < MIN_WINDOW_SESSIONS rc=3（样本不足与无漂移显式区分）
    """
    if not baseline_path.exists():
        print(f"[drift] 基线缺失——先 --snapshot（{baseline_path}）", file=sys.stderr)
        return 1
    baseline = _load_baseline(baseline_path)
    if baseline is None:
        return 1
    base_keys = dict(baseline.get("keys", {}))
    steps_p95 = float(baseline.get("steps_p95", 0) or 0)
    if not db.exists():
        print(f"[drift] db 不存在: {db}", file=sys.stderr)
        return 1
    conn = sqlite3.connect(str(db))
    try:
        rows = _fetch_sessions(conn, str(baseline.get("generated_at", "")))
    finally:
        _close_logged(conn, db)
    if not rows:
        print("[drift] 快照后无新增会话，无对照对象（rc=2）")
        return 2
    new_keys = _keys_from_rows(rows)
    steps_series = new_keys.pop("steps_series")
    parse_failures = new_keys.pop("parse_failures")
    n = int(new_keys["sessions"])
    print(f"[drift] 对照会话数 {n}（基线 {int(baseline.get('baseline_sessions', 0))}）")
    if parse_failures:
        print(f"[drift] 警示: {parse_failures} 个会话 tool_breakdown 解析失败按缺失计", file=sys.stderr)
    if n < MIN_WINDOW_SESSIONS:
        print(
            f"[drift] 样本不足（n={n}<{MIN_WINDOW_SESSIONS}）——短窗口中位数抽样波动大，"
            f"禁据此刻判漂移或宣称无漂移（rc=3）"
        )
        return 3
    drifted = _verdict(base_keys, new_keys)
    frac, outliers = _outlier_evidence(steps_series, steps_p95)
    if frac > OUTLIER_FRAC:
        drifted.append("steps_median")
    return _report(base_keys, new_keys, drifted, frac, outliers, n)


def _verdict(base: dict, new: dict) -> list[str]:
    """判定漂移键集合（中位数倍率阈值 + duration 第四键）。

    Contract:
        Preconditions: base/new 为四键 dict
        Postconditions: 返回 drifted 键名列表（可为空）；基线值为 0 的键跳过
    """
    out: list[str] = []
    for key, ratio in (
        ("steps_median", STEPS_RATIO),
        ("tools_median", TOOLS_RATIO),
        ("avg_duration_ms_median", DURATION_RATIO),
    ):
        b = float(base.get(key, 0) or 0)
        n = float(new.get(key, 0) or 0)
        if b > 0 and n > b * ratio:
            out.append(key)
    b_err = float(base.get("error_rate_median", 0) or 0)
    n_err = float(new.get("error_rate_median", 0) or 0)
    if n_err > ERROR_RATE_ABS and n_err > b_err * ERROR_RATE_RATIO:
        out.append("error_rate_median")
    return out


def _assert_verdict_cases() -> None:
    """中位数判定层负例（漂移命中/阈值内/零基线三形态）。"""
    base = {"steps_median": 10.0, "tools_median": 4.0, "error_rate_median": 0.1, "avg_duration_ms_median": 500.0}
    assert _verdict(base, dict(base)) == []  # 期望: 无偏移不判漂移
    assert _verdict(base, dict(base, steps_median=20.0)) == ["steps_median"]  # 期望: steps 超 1.5x 命中
    assert _verdict(base, dict(base, error_rate_median=0.5)) == ["error_rate_median"]  # 期望: 错误率双条件命中
    assert _verdict(base, dict(base, error_rate_median=0.15)) == []  # 期望: 0.15<0.2 绝对阈值挡住误报
    assert _verdict(base, dict(base, avg_duration_ms_median=1200.0)) == ["avg_duration_ms_median"]  # 期望: 第四键命中
    assert _verdict(base, dict(base, avg_duration_ms_median=900.0)) == []  # 期望: 900<1000 阈内不判
    zero = {"steps_median": 0.0, "tools_median": 0.0, "error_rate_median": 0.0, "avg_duration_ms_median": 0.0}
    assert _verdict(zero, dict(base, steps_median=100.0)) == []  # 期望: 基线为 0 无可比性


def _assert_outlier_cases() -> None:
    """会话级离群与分位数层负例（长尾/退化/正常三形态）。"""
    assert _steps_percentile([1.0, 2.0, 3.0, 100.0], 95) > 90  # 期望: 长尾单会话抬高 p95
    frac, n_out = _outlier_evidence([5.0, 6.0, 5.5, 200.0], 10.0)
    assert n_out == 1 and frac == 0.25  # 期望: 单离群会话 1/4 占比正确
    assert _outlier_evidence([5.0, 200.0], 0.0) == (0.0, 0)  # 期望: 基线 p95=0 退化时不判
    assert _outlier_evidence([5.0] * 8, 10.0) == (0.0, 0)  # 期望: 全正常零离群


def _negative_self_check() -> int:
    """负例自测：判定器对漂移/正常/样本不足三形态必须正确分类。

    Contract:
        Preconditions: 无
        Postconditions: 全部断言通过返回 0
    """
    _assert_verdict_cases()
    _assert_outlier_cases()
    print("[drift] self-test 全部通过")
    return 0


def run_cli(argv: list[str] | None = None) -> int:
    """CLI 入口：快照/对照/负例自测三模式。"""
    parser = argparse.ArgumentParser(description="agent 行为漂移基线（会话四键两层对照 v2）")
    parser.add_argument("--snapshot", action="store_true", help="固化当前窗口为 v2 基线")
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
