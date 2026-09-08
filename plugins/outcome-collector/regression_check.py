#!/usr/bin/env python3
"""discipline-regression — 纪律写入后回归验收（outcome feedback flywheel Layer 2.5）.

回答一个问题：从 violations/face_slaps 固化进 memory 的纪律条目，
写入之后该类违规是否真的下降了？

    纪律写入 memory ──→ 后续会话行为改变 ──→ violations 下降 ?

判定方法（纯 SQL，无 LLM）：
- 每条被跟踪纪律有一个生效日期（discipline_dates.json，键=违规规则名）。
- 基线窗口 = 生效日前 7 天；验收窗口 = 生效日后 7 天（须已完整过去才判定）。
- 指标 = 日均违规数 / 日均工具调用量（密度归一，消除会话活动量波动的误判：
  违规绝对数下降可能只是当天用得少）。
- 判定：验收密度 < 基线密度的 70% → effective；
  > 基线的 100% → regressed（纪律无效，候选出清）；之间 → inconclusive。

USAGE:
    python plugins/outcome-collector/regression_check.py [--db PATH] [--dates PATH]
    OUTCOME_COLLECTOR_DISABLE=1  → no-op

Preconditions:
    - outcomes.db 存在且含 violations / tool_outcomes 表（由本插件 Layer 0 创建）。
    - discipline_dates.json 为 JSON dict：{"R6": "2026-09-07", ...}；缺失则用空集（报告提示）。
Postconditions:
    - stdout 输出人类可读判定；--json 输出结构化结果；
      --write 落盘 ~/.hermes/skill_suggestions/regression-YYYYMMDD.md。
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

WINDOW_DAYS = 7
# 验收窗口末尾距今至少留 N 天，避免"今天刚生效"就下判定
MIN_DAYS_AFTER = 7

EFFECTIVE_RATIO = 0.70     # 验收/基线密度 ≤ 0.70 → effective
REGRESSED_RATIO = 1.00     # 验收/基线密度 > 1.00 → regressed


def _disabled() -> bool:
    return os.environ.get("OUTCOME_COLLECTOR_DISABLE", "").lower() in {"1", "true", "yes", "on"}


def _default_db_path() -> Path:
    home = os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
    return Path(home) / "outcomes.db"


def _default_dates_path() -> Path:
    home = os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes")
    return Path(home) / "outcomes" / "discipline_dates.json"


def load_discipline_dates(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {str(k): str(v) for k, v in data.items() if isinstance(v, str)}
    except (json.JSONDecodeError, OSError):
        return {}


def _daily_counts(conn: sqlite3.Connection, table: str, ts_col: str,
                  extra_where: str = "", params: tuple = ()) -> Dict[str, int]:
    """date(YYYY-MM-DD) → row count for the given table/filters."""
    sql = (
        f"SELECT date({ts_col}) AS d, COUNT(*) AS n FROM {table} "
        f"WHERE {ts_col} IS NOT NULL {extra_where} GROUP BY d"
    )
    return {r[0]: r[1] for r in conn.execute(sql, params).fetchall() if r[0]}


def _window_stats(daily: Dict[str, int], start: str, end: str) -> Optional[Dict[str, float]]:
    """Average daily count over [start, end] inclusive. None if window empty."""
    vals = [daily[d] for d in _dates_between(start, end) if d in daily]
    if not vals:
        return None
    return {"days": len(vals), "total": sum(vals), "avg": sum(vals) / len(vals)}


def _dates_between(start: str, end: str) -> List[str]:
    from datetime import date, timedelta
    s = date.fromisoformat(start)
    e = date.fromisoformat(end)
    out = []
    while s <= e:
        out.append(s.isoformat())
        s += timedelta(days=1)
    return out


def check_rule(conn: sqlite3.Connection, rule: str, effective: str,
               today: str) -> Optional[Dict[str, Any]]:
    """Compare violation density before/after the discipline's effective date.

    Contract:
      Preconditions: effective is ISO date str; violations table exists.
      Postconditions: returns dict with verdict in
        {effective, regressed, inconclusive, pending} or None (no data).
    """
    viol = _daily_counts(
        conn, "violations", "timestamp", " AND rule = ?", (rule,)
    )
    calls = _daily_counts(conn, "tool_outcomes", "timestamp")
    if not viol:
        return None

    from datetime import date, timedelta
    eff = date.fromisoformat(effective)
    base_start = (eff - timedelta(days=WINDOW_DAYS)).isoformat()
    base_end = (eff - timedelta(days=1)).isoformat()
    after_start = (eff + timedelta(days=1)).isoformat()
    after_end = (eff + timedelta(days=WINDOW_DAYS)).isoformat()

    if after_end > today:
        return {"rule": rule, "effective": effective, "verdict": "pending",
                "reason": f"验收窗口 {after_start}~{after_end} 尚未完整过去"}

    def density(start: str, end: str) -> Optional[float]:
        c = _window_stats(calls, start, end)
        if not c or c["total"] == 0:
            return None
        # 0违规但当天有工具调用 = 有效数据（密度0），不能当"无数据"跳过
        v = _window_stats(viol, start, end)
        viol_total = v["total"] if v else 0
        # 每千次工具调用的违规数
        return viol_total / c["total"] * 1000.0

    base_d = density(base_start, base_end)
    after_d = density(after_start, after_end)
    if base_d is None or after_d is None:
        return {"rule": rule, "effective": effective, "verdict": "inconclusive",
                "reason": "窗口内无数据，无法计算密度"}

    ratio = after_d / base_d if base_d > 0 else (0.0 if after_d == 0 else float("inf"))
    if ratio <= EFFECTIVE_RATIO:
        verdict = "effective"
    elif ratio > REGRESSED_RATIO:
        verdict = "regressed"
    else:
        verdict = "inconclusive"

    return {
        "rule": rule,
        "effective": effective,
        "verdict": verdict,
        "baseline_window": f"{base_start}~{base_end}",
        "after_window": f"{after_start}~{after_end}",
        "baseline_density_per_1k_calls": round(base_d, 2),
        "after_density_per_1k_calls": round(after_d, 2),
        "ratio": round(ratio, 3),
    }


def run_regression_check(db_path: Path, dates_path: Path) -> Dict[str, Any]:
    dates = load_discipline_dates(dates_path)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    results: List[Dict[str, Any]] = []
    if not db_path.exists():
        return {"error": f"db not found: {db_path}"}
    conn = sqlite3.connect(str(db_path))
    try:
        for rule, eff in sorted(dates.items()):
            r = check_rule(conn, rule, eff, today)
            if r:
                results.append(r)
        # 附带：当前 DB 里实际出现过的规则，提示未跟踪的
        tracked = set(dates)
        present = {r[0] for r in conn.execute("SELECT DISTINCT rule FROM violations")}
        untracked = sorted(present - tracked)
    finally:
        conn.close()
    return {"generated_at": today, "results": results, "untracked_rules": untracked,
            "dates_file": str(dates_path)}


def format_as_text(report: Dict[str, Any]) -> str:
    lines = [f"=== Discipline Regression Check ({report['generated_at']}) ===", ""]
    verdict_cn = {"effective": "生效✅", "regressed": "回退❌", "inconclusive": "不显著",
                  "pending": "窗口未满"}
    for r in report.get("results", []):
        v = verdict_cn.get(r["verdict"], r["verdict"])
        lines.append(f"[{v}] {r['rule']} (生效 {r['effective']})")
        if all(k in r for k in ("baseline_window", "baseline_density_per_1k_calls")):
            lines.append(
                f"    基线 {r['baseline_window']}: {r['baseline_density_per_1k_calls']}/千调用  →  "
                f"验收 {r['after_window']}: {r['after_density_per_1k_calls']}/千调用  "
                f"(ratio={r['ratio']})"
            )
        else:
            lines.append(f"    {r.get('reason', '')}")
    untracked = report.get("untracked_rules", [])
    if untracked:
        lines.append("")
        lines.append(
            f"未跟踪生效日期的规则: {', '.join(untracked)} "
            f"— 在 {report['dates_file']} 补条目后可纳入验收"
        )
    return "\n".join(lines)


def write_report_file(report: Dict[str, Any]) -> Path:
    home = Path(os.environ.get("HERMES_HOME") or str(Path.home() / ".hermes"))
    out_dir = home / "skill_suggestions"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"regression-{report['generated_at'].replace('-', '')}.md"
    body = [
        f"# 纪律回归验收报告 {report['generated_at']}",
        "",
        "指标：每千次工具调用的违规数（密度归一）。effective=验收≤基线70%；"
        "regressed=验收>基线（纪律无效，候选出清或改写）。",
        "",
        "```",
        format_as_text(report),
        "```",
    ]
    out.write_text("\n".join(body), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="纪律写入后回归验收")
    parser.add_argument("--db", type=Path, default=_default_db_path())
    parser.add_argument("--dates", type=Path, default=_default_dates_path())
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    if _disabled():
        print("outcome-collector disabled (OUTCOME_COLLECTOR_DISABLE)")
        return

    report = run_regression_check(args.db, args.dates)
    if "error" in report:
        print(report["error"])
        sys.exit(1)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(format_as_text(report))
    if args.write:
        p = write_report_file(report)
        print(f"\n--- report written: {p}")


if __name__ == "__main__":
    main()
