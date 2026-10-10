#!/usr/bin/env python3
"""RSI 前沿度量——自改进系统的可达前沿由验证能力决定（机器度量 + 棘轮）。

度量三件事（全部机器事实，无关键词判断、无自述）：
  1. 闸门生产率：outcomes.db 里每个工具的 blocked/rejected 命中数，以及命中之后
     同工具是否转 ok（retry_then_success）＝该拦截是否产生了修复动作 → 有效拦截率
  2. 红证覆盖：guards/discipline 下每个插件在 tests/ 下是否有引用它的测试文件
     （没有红证的门禁＝把「我以为它会拦」当门禁；红证=它失效时会变红的那条测试）
  3. 棘轮：与 frontier_baseline.json 比较，红证率/有效拦截率下降即 regressed

输出：~/.hermes/outcomes/frontier-<date>.md 与 frontier_alerts.json（regressed 时），
由 outcome-collector 首轮注入带给每个新会话。

Contract:
  Preconditions: 无——db/基线/目录都可能不存在，缺即按零值起步
  Postconditions: 纯读 + 落盘报告；任何 IO 失败只 logger.warning，绝不 raise
"""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_HOME_FALLBACK = Path.home() / ".hermes"


def _hermes_home() -> Path:
    """活动 HERMES_HOME（含 env 覆盖），无 Hermes 运行时时退回 ~/.hermes。

    Contract:
      Preconditions: 无
      Postconditions: 返回 home 路径；无法导入 hermes_constants 时告警并回退
    """
    try:
        from hermes_constants import get_hermes_home

        return Path(get_hermes_home())
    except ImportError as exc:
        import os

        logger.warning("frontier: hermes_constants 不可用，按环境变量/默认 home 解析: %s", exc)
        return Path(os.environ.get("HERMES_HOME") or _HOME_FALLBACK)


def _python_modules(plugin_dir: Path) -> list[str]:
    """列出插件目录下的模块名（排除 __init__/preflight 等非规则件）。

    Preconditions: plugin_dir 可能不存在。
    Postconditions: 目录缺失返回空列表；返回按名排序的 .py 模块名（不含后缀）。
    """
    if not plugin_dir.is_dir():
        return []
    skip = {"__init__", "preflight"}
    return sorted(p.stem for p in plugin_dir.glob("*.py") if p.stem not in skip)


def _red_proof_markers(name: str) -> tuple[str, ...]:
    """「测试真正引用该规则件」的判定标记（插件路径 / from-import / 模块属性三种形态）。

    Contract:
      Preconditions: name 为非空模块名。
      Postconditions: 返回标记元组；命中任一即视为该测试真的在验这个规则件。
    """
    return (
        f"plugins/guards/{name}.py",
        f"plugins/discipline/{name}.py",
        f"guards import {name}",
        f"discipline import {name}",
        f'"guards" / "{name}.py"',
        f'"discipline" / "{name}.py"',
        f"guards.{name}",
        f"discipline.{name}",
    )


def _has_red_proof(name: str, tests_dir: Path) -> bool:
    """该插件在 tests/ 下是否存在「真正引用它」的测试（＝失效时会变红的证据）。

    Contract:
      Preconditions: tests_dir 可能不存在。
      Postconditions: 命中任一引用标记即 True；裸子串撞名（临时文件名/函数名）不算，
        否则度量会被无关字面量污染（实测：自测文件的临时文件名曾把覆盖率从 4/19 抬到 7/19）。
    """
    if not tests_dir.is_dir():
        return False
    markers = _red_proof_markers(name)
    for f in sorted(tests_dir.rglob("test_*.py")):
        try:
            text = f.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("frontier: 测试文件读取失败 %s: %s", f, exc)
            continue
        if any(marker in text for marker in markers):
            return True
    return False


def _fetch_status_rows(db: Path, days: int) -> list[tuple[str, str, str, str]]:
    """取窗口内的 (session_id, turn_id, tool_name, status) 行。

    Preconditions: db 可能不存在或表缺失。
    Postconditions: 缺库/缺表返回空列表；按 id 升序，供序列判定。
    """
    if not db.exists():
        return []
    since = (datetime.now(UTC) - timedelta(days=days)).isoformat()
    try:
        with closing(sqlite3.connect(str(db), timeout=5)) as conn:
            rows = conn.execute(
                """SELECT session_id, COALESCE(turn_id,''), tool_name, status
                   FROM tool_outcomes
                   WHERE timestamp >= ? AND tool_name != '_turn_outcome'
                   ORDER BY id""",
                (since,),
            ).fetchall()
        return [(str(r[0]), str(r[1]), str(r[2]), str(r[3])) for r in rows]
    except sqlite3.Error as exc:
        logger.warning("frontier: 读取 tool_outcomes 失败（库可能未建/表缺失）: %s", exc)
        return []


def _rule_hits(db: Path, days: int) -> dict[str, int]:
    """L3 纪律命中账：violations 按 rule×outcome 计数（block/rewrite/hint 分列）。

    Contract:
      Preconditions: violations 表 schema 由 plugins/discipline/no_guessing
        的 _ensure_violations_table 定义，必含 outcome 列。
      Postconditions: 返回 {"R<N>:<outcome>": count}；库不存在或表未建
        =零数据合法态返回 {}；表存在但缺 outcome 列=契约违约，raise
        RuntimeError（错误原文不遮盖）。
    """
    if not db.exists():
        return {}
    since = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")
    try:
        with closing(sqlite3.connect(str(db), timeout=5)) as conn:
            rows = conn.execute(
                "SELECT rule, outcome, COUNT(*) FROM violations "
                "WHERE timestamp >= ? GROUP BY rule, outcome",
                (since,),
            ).fetchall()
        return {f"{r[0]}:{r[1]}": int(r[2]) for r in rows}
    except sqlite3.OperationalError as exc:
        if "no such table" in str(exc):
            return {}
        raise RuntimeError(
            f"frontier: violations 表违反契约 schema（期望含 outcome 列）: {exc}"
        ) from exc


def _declared_rule_ids(plugin_dirs: list[Path]) -> list[str]:
    """规则件源码里声明的规则编号（str 方法提取，不用正则）。

    Contract:
      Preconditions: 目录可能不存在。
      Postconditions: 返回排序去重的 R+数字编号；目录缺失按跳过处理。
    """
    found: set[str] = set()
    for d in plugin_dirs:
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.py")):
            try:
                src = p.read_text(encoding="utf-8")
            except OSError as exc:
                logger.warning("frontier: 规则件读取失败 %s: %s", p, exc)
                continue
            for raw in src.replace('"', " ").replace("'", " ").replace("(", " ").split():
                token = raw.strip(",:[].{}|=")
                if len(token) >= 2 and token[0] == "R" and token[1:].isdigit():
                    found.add(token)
    return sorted(found)


def compute_gate_stats(db: Path, days: int = 30) -> dict[str, dict[str, Any]]:
    """按工具统计闸门生产率。

    Preconditions: db 为 outcomes.db 路径（可不存在）。
    Postconditions: {tool: {blocks, effective, ineffective}}；effective=该工具的
      拦截之后（同会话同回合内）出现过 ok；判据只看序列，不看文案。
    """
    per_tool: dict[str, dict[str, Any]] = {}
    pending: dict[tuple[str, str], set[str]] = {}
    for sid, tid, tool, status in _fetch_status_rows(db, days):
        stat = per_tool.setdefault(tool, {"blocks": 0, "effective": 0, "ineffective": 0})
        key = (sid, tid)
        if status in ("blocked", "rejected"):
            stat["blocks"] += 1
            pending.setdefault(key, set()).add(tool)
        elif status == "ok" and tool in pending.get(key, set()):
            stat["effective"] += 1
            pending[key].discard(tool)
    for stat in per_tool.values():
        stat["ineffective"] = stat["blocks"] - stat["effective"]
    return per_tool


def compute_frontier(db: Path, guards_dir: Path, discipline_dir: Path, tests_dir: Path,
                     days: int = 30) -> dict[str, Any]:
    """汇总前沿报告（闸门生产率 + 红证覆盖）。

    Preconditions: 四个路径都可不存在。
    Postconditions: 返回 {generated_at, window_days, gates, rule_hits, red_proof, summary}；
      summary 含 red_proof_ratio / effective_ratio / zero_red_proof / zero_hit_rules。
      两本账分列：工具账（tool_outcomes）与规则账（violations）口径不同，不混算。
    """
    modules = _python_modules(guards_dir) + _python_modules(discipline_dir)
    red = {name: _has_red_proof(name, tests_dir) for name in modules}
    stats = compute_gate_stats(db, days)
    rule_hits = _rule_hits(db, days)
    declared = _declared_rule_ids([guards_dir, discipline_dir])

    total_blocks = sum(s["blocks"] for s in stats.values())
    total_effective = sum(s["effective"] for s in stats.values())
    zero_red = sorted(name for name, has in red.items() if not has)
    # rule_hits 键已带 outcome 后缀（R6:rewrite）——零命中判定按裸规则码前缀聚合
    hit_codes = {k.split(":", 1)[0] for k in rule_hits}
    zero_hit_rules = [r for r in declared if r not in hit_codes]
    red_ratio = (sum(1 for v in red.values() if v) / len(red)) if red else 1.0
    eff_ratio = (total_effective / total_blocks) if total_blocks else 1.0
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "window_days": days,
        "gates": stats,
        "rule_hits": rule_hits,
        "red_proof": red,
        "summary": {
            "modules": len(modules),
            "red_proof_ratio": round(red_ratio, 4),
            "blocks": total_blocks,
            "effective": total_effective,
            "effective_ratio": round(eff_ratio, 4),
            "zero_red_proof": zero_red,
            "zero_hit_rules": zero_hit_rules,
        },
    }


def compare_with_baseline(report: dict[str, Any], baseline_path: Path) -> list[str]:
    """棘轮：前沿指标只增不减，下降即 regressed。

    Preconditions: baseline_path 可不存在（首次运行＝建立基线）。
    Postconditions: 返回回退项人话列表（空=无回退）；基线文件损坏视为无基线并告警。
    """
    if not baseline_path.exists():
        return []
    try:
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("frontier: 基线读取失败（按无基线段处理）: %s", exc)
        return []
    if not isinstance(baseline, dict):
        return []
    now = report["summary"]
    regressions: list[str] = []
    for metric, label in (("red_proof_ratio", "红证覆盖率"),
                          ("effective_ratio", "有效拦截率")):
        old = baseline.get(metric)
        if isinstance(old, (int, float)) and now[metric] < old - 1e-9:
            regressions.append(f"{label} 回退：{old:.4f} → {now[metric]:.4f}")
    return regressions


def _filter_dispositioned(regressions: list[str], home: Path) -> list[str]:
    """已登记处置的回退不再告警（dispositions.json 键形如 'frontier:红证覆盖率'）。

    Contract:
      Preconditions: home 为 HERMES_HOME；regressions 为回退项人话列表。
      Postconditions: 返回未处置的回退；处置文件缺失/损坏按全部未处置处理（宁告警不漏警）。
    """
    if not regressions:
        return []
    path = home / "outcomes" / "dispositions.json"
    if not path.exists():
        return regressions
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("frontier: dispositions 读取失败（按全部未处置处理）: %s", exc)
        return regressions
    if not isinstance(data, dict):
        return regressions
    kept: list[str] = []
    for item in regressions:
        dispositioned = any(
            key.startswith("frontier:") and key.split(":", 1)[1] in item for key in data)
        if not dispositioned:
            kept.append(item)
    return kept


def render_markdown(report: dict[str, Any], regressions: list[str]) -> str:
    """人读报告（供人/agent 复查；不参与判据计算）。

    Preconditions: report 为 compute_frontier 输出。
    Postconditions: 返回 markdown 文本，含汇总/工具账/规则账/缺口/棘轮回退五段。
    """
    s = report["summary"]
    lines = [
        f"# RSI 前沿报告（{report['generated_at'][:10]}，窗口 {report['window_days']} 天）",
        "",
        f"- 红证覆盖率：{s['red_proof_ratio']:.4f}（"
        f"{sum(1 for v in report['red_proof'].values() if v)}/{s['modules']} 个规则件有红证）",
        f"- 工具账拦截：{s['blocks']} 次，其中 {s['effective']} 次同回合转 ok（有效拦截率 "
        f"{s['effective_ratio']:.4f}）",
        f"- 规则账命中：{sum(report['rule_hits'].values())} 次"
        f"（{len(report['rule_hits'])} 个规则码）",
        f"- 无红证的规则件：{len(s['zero_red_proof'])} 个；"
        f"账上零命中的声明规则码：{len(s['zero_hit_rules'])} 个",
        "",
        "## 工具账（tool_outcomes）",
        "",
        "| 工具 | 命中 | 有效 | 无效 |",
        "|---|---|---|---|",
    ]
    for tool, stat in sorted(report["gates"].items()):
        lines.append(
            f"| {tool} | {stat['blocks']} | {stat['effective']} | {stat['ineffective']} |")
    lines += ["", "## 纪律规则账（violations）", "", "| 规则码 | 命中 |", "|---|---|"]
    for rule, hits in sorted(report["rule_hits"].items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"| {rule} | {hits} |")
    lines += ["", "## 缺口", ""]
    lines += [f"- 无红证：{n}" for n in s["zero_red_proof"]] or ["- 无红证规则件"]
    lines += [f"- 零命中：{r}" for r in s["zero_hit_rules"]] or ["- 无零命中规则码"]
    lines += ["", "## 棘轮回退", ""]
    lines += [f"- {r}" for r in regressions] if regressions else ["- 无"]
    return "\n".join(lines) + "\n"


def _write_outputs(report: dict[str, Any], regressions: list[str], home: Path) -> Path:
    """落盘报告与（regressed 时的）警讯文件。

    Preconditions: home 为 HERMES_HOME。
    Postconditions: 写 outcomes/frontier-<date>.md；有回退时另写 frontier_alerts.json
      （无回退则删除旧警讯）；返回报告路径。
    """
    out_dir = home / "outcomes"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = report["generated_at"][:10]
    report_path = out_dir / f"frontier-{stamp}.md"
    report_path.write_text(render_markdown(report, regressions), encoding="utf-8")
    alerts_path = out_dir / "frontier_alerts.json"
    if regressions:
        alerts_path.write_text(
            json.dumps({"alerts": regressions, "date": stamp}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    elif alerts_path.exists():
        alerts_path.unlink()
    return report_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="RSI 前沿度量（验证能力账本 + 棘轮）")
    home = _hermes_home()
    plugin_root = Path(__file__).resolve().parent.parent
    ap.add_argument("--db", default=str(home / "outcomes.db"))
    ap.add_argument("--guards", default=str(plugin_root / "guards"))
    ap.add_argument("--discipline", default=str(plugin_root / "discipline"))
    ap.add_argument("--tests", default=str(Path(__file__).resolve().parents[2] / "tests"))
    ap.add_argument("--baseline", default=str(home / "outcomes" / "frontier_baseline.json"))
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--no-write", action="store_true", help="只算不落盘（CI/测试用）")
    ap.add_argument("--regen-baseline", action="store_true", help="把当前指标写成新基线")
    args = ap.parse_args(argv)

    report = compute_frontier(
        Path(args.db), Path(args.guards), Path(args.discipline), Path(args.tests), args.days)
    baseline_path = Path(args.baseline)
    if args.regen_baseline:
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(
            json.dumps(report["summary"], ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"基线已更新: {baseline_path}")
        return 0

    regressions = _filter_dispositioned(compare_with_baseline(report, baseline_path), home)
    if args.json:
        print(json.dumps({**report, "regressions": regressions},
                         ensure_ascii=False, indent=1))
        return 1 if regressions else 0
    if not args.no_write:
        print(f"报告已落盘: {_write_outputs(report, regressions, home)}")
    print(render_markdown(report, regressions))
    return 1 if regressions else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    cli_argv = sys.argv[1:]
    raise SystemExit(main(cli_argv))
