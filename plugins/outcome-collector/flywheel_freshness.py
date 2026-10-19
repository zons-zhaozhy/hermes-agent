#!/usr/bin/env python3
"""flywheel-freshness — 飞轮产出时效自检（监测者的监测）。

背景（2026-09-26 实测断点B）：每日审计断流 8 天（09-17~09-23）零告警——
heartbeat-watchdog 只盯 retro/metrics，不盯审计产出；审计 job 失效后
整个自我监督层静默消失。本脚本补齐：检查飞轮关键产出的时效，过期即
以非零退出码 + stderr 告警（cron/watchdog 调用方转为可见信号）。

检查项（产出 → 最大静默时长）：
  1. skill_suggestions/audit-YYYYMMDD.md     — 每日审计，>26h 未见新文件即告警
     （每日 21:00 产出 → 次日 23:00 前未更新即告警）
  2. outcomes/regression_alerts.json         — 每日 07:00 cron，>26h 未更新即告警
  3. outcomes/findings.md                    — 每 6h 跑，>12h 未更新即告警

用法:
    python plugins/outcome-collector/flywheel_freshness.py [--home PATH] [--json]
    退出码: 0=全部新鲜; 1=有产出过期（stderr 逐条列出）; 2=环境错误

Contract:
  Preconditions: HERMES_HOME 可解析（参数 > 环境变量 > 默认 ~/.hermes）
  Postconditions: 永不 raise；产出缺失/过期时 exit 1 且 stderr 含文件路径与账龄

已查重：plugins/outcome-collector/ 内 regression_check=纪律回归判定（密度对比），
analyze=工具错误模式分析，本文件=飞轮产出时效哨兵，职责互斥无等价实现。
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional


def _resolve_home(explicit: Optional[str]) -> Path:
    """Contract: 显式参数 > HERMES_HOME 环境变量 > ~/.hermes，返回 Path。"""
    if explicit:
        return Path(explicit)
    env = os.environ.get("HERMES_HOME")
    if env:
        return Path(env)
    return Path.home() / ".hermes"


def _latest_audit(home: Path) -> Optional[Path]:
    """找 skill_suggestions/ 下最新的 audit-YYYYMMDD.md（命名约定=产出通道契约）。"""
    d = home / "skill_suggestions"
    if not d.is_dir():
        return None
    cands = sorted(p for p in d.glob("audit-*.md") if len(p.stem) == len("audit-YYYYMMDD"))
    return cands[-1] if cands else None


def _age_check(name: str, path: Optional[Path], max_age_hours: float) -> dict[str, Any]:
    """单产出时效检查。缺失=最老（账龄 None，reason=missing）。

    Contract:
      Preconditions: path 为 Path 或 None
      Postconditions: 返回含 name/ok/path/age_hours/reason 的 dict，永不 raise
    """
    if path is None or not path.exists():
        return {
            "name": name, "ok": False, "path": str(path) if path else "<none>",
            "age_hours": None, "reason": "missing",
        }
    age_h = (time.time() - path.stat().st_mtime) / 3600
    return {
        "name": name, "ok": age_h <= max_age_hours, "path": str(path),
        "age_hours": round(age_h, 1),
        "reason": None if age_h <= max_age_hours else "stale",
    }


def _latest_mtime(directory: Path, pattern: str = "*.md") -> Optional[float]:
    """取目录下匹配文件的最新 mtime（无匹配返回 None）。

    Contract:
      Preconditions: directory 为 Path（可能不存在）
      Postconditions: 返回最新 mtime epoch 秒或 None，永不 raise
    """
    if not directory.is_dir():
        return None
    mtimes = [p.stat().st_mtime for p in directory.glob(pattern) if p.is_file()]
    return max(mtimes) if mtimes else None


def _age_check_dir(name: str, directory: Path, max_age_hours: float) -> dict[str, Any]:
    """目录级产出时效检查（按最新文件 mtime）。

    Contract:
      Preconditions: directory 为 Path（可能不存在——按全缺失处理）
      Postconditions: 返回含 name/ok/path/age_hours/reason 的 dict，永不 raise
    """
    latest = _latest_mtime(directory)
    if latest is None:
        return {
            "name": name, "ok": False, "path": str(directory),
            "age_hours": None, "reason": "missing",
        }
    age_h = (time.time() - latest) / 3600
    return {
        "name": name, "ok": age_h <= max_age_hours, "path": str(directory),
        "age_hours": round(age_h, 1),
        "reason": None if age_h <= max_age_hours else "stale",
    }


# arXiv 收割链判活参数。文件 mtime 不能做该源的唯一判活依据：arXiv OAI-PMH
# 周六/周日零 datestamp 记录，周五晨收割→下周二晨收割的合法产出间隙≈72h，
# 恒大于旧 50h 阈值——每逢完整周末后的周一结构性误报（2026-10-14/10-19 两案
# 定性）。主信号=收割 job 自身执行结果；内容 mtime 降为兜底地板（80h=最长
# 合法间隙 72h + 8h 余量），地板击穿且机器不活才告警。
ARXIV_PAPERS_DIR = Path("/Users/stan/Knowledge-Base/ai-research/papers")
ARXIV_HARVEST_JOB_ID = "504c034bbdba"  # 「AI专题知识OAI增量采集」每日 07:10
ARXIV_CONTENT_FLOOR_HOURS = 80.0
ARXIV_MACHINERY_WINDOW_HOURS = 48.0

_LOGGER = logging.getLogger(__name__)


def _harvest_job_alive(home: Path, job_id: str,
                       window_hours: float = ARXIV_MACHINERY_WINDOW_HOURS) -> Optional[bool]:
    """收割 job 近 window_hours 内是否有 completed 执行。

    Contract:
      Preconditions: executions.db 位于 home/cron/executions.db（可能不存在）
      Postconditions: True/False=判定结果；None=无法判定（db 缺失/查询异常，
        调用方走内容地板兜底）；永不 raise
    """
    import sqlite3
    db = home / "cron" / "executions.db"
    if not db.is_file():
        return None
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            row = con.execute(
                "select count(*) from executions where job_id=? and status='completed' "
                "and julianday(finished_at)>julianday('now', ?)",
                (job_id, f"-{window_hours:g} hours"),
            ).fetchone()
        finally:
            try:
                con.close()
            except sqlite3.Error:  # close 失败不得掩盖查询原始异常
                _LOGGER.warning("harvest liveness: close executions.db failed", exc_info=True)
    except sqlite3.Error as e:
        _LOGGER.warning("harvest liveness: query executions.db failed: %s", e)
        return None
    return bool(row and row[0] > 0)


def _age_check_arxiv(home: Path) -> dict[str, Any]:
    """arxiv-papers 判活：内容地板 + 收割机器信号双门。

    告警条件=内容真空超地板 且 收割 job 48h 内无 completed 执行（双失才报，
    周末上游零增量不报——机器活着即链路健康）。job 自身失败由每日 cron 失败
    审计独立覆盖，不在此重复。

    Contract:
      Preconditions: home 为 Path
      Postconditions: 返回含 name/ok/path/age_hours/reason 的 dict，永不 raise
    """
    content = _age_check_dir("arxiv-papers", ARXIV_PAPERS_DIR, ARXIV_CONTENT_FLOOR_HOURS)
    if content["ok"]:
        return content
    if _harvest_job_alive(home, ARXIV_HARVEST_JOB_ID):
        content["ok"] = True
        content["reason"] = "upstream-quiet-harvest-ok"
    return content


def check_freshness(home: Path) -> dict[str, Any]:
    """逐项检查飞轮产出时效。

    Contract:
      Preconditions: home 为 Path（可能不存在——按全缺失处理）
      Postconditions: 返回 {"home","alerts","checked"}，永不 raise
    """
    checks: list[dict[str, Any]] = []
    checks.append(_age_check("daily-audit", _latest_audit(home), max_age_hours=26))
    checks.append(_age_check("regression-alerts", home / "outcomes" / "regression_alerts.json", 26))
    checks.append(_age_check("analyzer-findings", home / "outcomes" / "findings.md", 12))
    # 知识采集层：arXiv 收割链判活（内容地板 80h + 收割 job 48h 双门，
    # 2026-10-19 根治周末结构性误报——详见 _age_check_arxiv 注释）
    checks.append(_age_check_arxiv(home))

    stale = [c for c in checks if not c["ok"]]
    return {"home": str(home), "alerts": stale, "checked": checks}


def main() -> int:
    parser = argparse.ArgumentParser(description="飞轮产出时效自检")
    parser.add_argument("--home", default=None, help="HERMES_HOME 覆盖")
    parser.add_argument("--json", action="store_true", help="JSON 输出")
    args = parser.parse_args()

    result = check_freshness(_resolve_home(args.home))

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        for c in result["checked"]:
            mark = "OK " if c["ok"] else "⚠ "
            age = c["age_hours"] if c["age_hours"] is not None else "∞"
            print(f"{mark}{c['name']}: age={age}h path={c['path']}", file=sys.stderr)
        if result["alerts"]:
            print(
                "FLYWHEEL STALE: " + ", ".join(a["name"] for a in result["alerts"]),
                file=sys.stderr,
            )
    return 1 if result["alerts"] else 0


if __name__ == "__main__":
    sys.exit(main())
