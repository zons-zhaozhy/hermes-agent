"""outcome-collector 飞轮末环测试（Layer 4 处置闭环）。

背景（2026-09-26 实测三断点）：
  A. regression 判 regressed 后报告只写 skill_suggestions/ 死目录，无处置通道——
     R5/R6 回退 9 天无人处置。
  B. 审计产出（audit-*.md / regression-*.md）断流无人察觉（09-17~09-23 断 8 天）。
  C. R6 机械改写每次发生时 agent 不可见——反馈只到行动层未回认知层。

本测试锁定闭合后的行为契约：
  - regression 报告生成时同步写 alerts 文件（regressed 判定未处置）；
  - alerts 存在时 findings 注入链把它带给每个新会话首 turn；
  - 处置登记（dispositions.json）后对应 alert 消失。

Contract:
  Preconditions: 回归模块可用 importlib 从插件目录加载（不依赖 hermes 运行时）
  Postconditions: 三个断言族全绿——alerts 落盘、注入文本含处置指引、处置后消警
"""

import importlib.util
import json
import logging
import os
import sqlite3
import types
from datetime import datetime, timedelta
from pathlib import Path

import pytest

logger = logging.getLogger(__name__)


def _load(name: str, rel: str) -> types.ModuleType:
    """Contract: 从插件目录按相对路径加载独立脚本模块，失败即 raise。"""
    path = Path(__file__).resolve().parents[2] / "plugins" / "outcome-collector" / rel
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def rc() -> types.ModuleType:
    return _load("rc_under_test", "regression_check.py")


def _close_quietly(conn: sqlite3.Connection) -> None:
    """Contract: 关闭连接,失败仅告警不上抛(清理路径不得掩盖原始异常)。"""
    try:
        conn.close()
    except Exception:
        logger.warning("sqlite close failed during cleanup", exc_info=True)


def _seed_regressed_db(db: Path) -> Path:
    """构造 regressed 形态:R6 生效 2026-09-01。
    密度推导:基线窗每天 10 调用×1 违规=100/千;
    验收窗每天 1 调用×1 违规=1000/千;ratio=10>1.00→regressed。"""
    conn = sqlite3.connect(str(db))
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS violations (id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " rule TEXT, command TEXT, level TEXT, session_id TEXT, timestamp TEXT)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_viol_rule_ts ON violations(rule, timestamp)"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS tool_outcomes (id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " session_id TEXT, tool_name TEXT, status TEXT, timestamp TEXT)"
        )
        for day in range(25, 32):
            d = f"2026-08-{day:02d}"
            for _ in range(10):
                conn.execute(
                    "INSERT INTO tool_outcomes(session_id, tool_name, status, timestamp)"
                    " VALUES ('s','t','ok',:ts)", {"ts": f"{d}T12:00:00"} )
            conn.execute(
                "INSERT INTO violations(rule, command, level, session_id, timestamp)"
                " VALUES ('R6','x','L1','s',:ts)", {"ts": f"{d}T12:00:00"} )
        for day in range(2, 9):
            d = f"2026-09-{day:02d}"
            conn.execute(
                "INSERT INTO tool_outcomes(session_id, tool_name, status, timestamp)"
                " VALUES ('s','t','ok',:ts)", {"ts": f"{d}T12:00:00"} )
            conn.execute(
                "INSERT INTO violations(rule, command, level, session_id, timestamp)"
                " VALUES ('R6','x','L1','s',:ts)", {"ts": f"{d}T12:00:00"} )
        conn.commit()
    finally:
        _close_quietly(conn)
    return db


@pytest.fixture()
def populated_db(tmp_path: Path) -> Path:
    return _seed_regressed_db(tmp_path / "outcomes.db")


@pytest.fixture()
def dates_file(tmp_path: Path) -> Path:
    p = tmp_path / "discipline_dates.json"
    p.write_text(json.dumps({"R6": "2026-09-01"}), encoding="utf-8")
    return p


def test_regressed_rule_writes_alert_file(rc, populated_db, dates_file, tmp_path, monkeypatch):
    """regressed 判定必须同步落 alerts 文件——不落=报告进死目录无人处置。"""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    report = rc.run_regression_check(populated_db, dates_file)
    verdicts = {r["rule"]: r["verdict"] for r in report["results"]}
    assert verdicts.get("R6") == "regressed"  # 期望: 基线100/千<验收1000/千,ratio=10>1.00→regressed
    alerts_path = tmp_path / "outcomes" / "regression_alerts.json"
    assert alerts_path.exists()  # 期望: regressed 判定同步落 alerts 文件
    data = json.loads(alerts_path.read_text(encoding="utf-8"))
    assert any(a["rule"] == "R6" and a["verdict"] == "regressed" for a in data["alerts"])  # 期望: alerts 含 R6 regressed 条目
    _ = data


def test_disposition_clears_alert(rc, populated_db, dates_file, tmp_path, monkeypatch):
    """处置登记后对应 alert 必须消失——处置无反馈=闭环又是死环。"""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    rc.run_regression_check(populated_db, dates_file)
    disp = tmp_path / "outcomes" / "dispositions.json"
    disp.write_text(json.dumps({"R6": {"action": "rewrote_as_mechanical_fix",
                                       "date": "2026-09-10"}}), encoding="utf-8")
    rc.run_regression_check(populated_db, dates_file)
    alerts_path = tmp_path / "outcomes" / "regression_alerts.json"
    data = json.loads(alerts_path.read_text(encoding="utf-8"))
    assert not any(a["rule"] == "R6" for a in data["alerts"])  # 期望: dispositions 登记处置→R6 消警


def test_injection_carries_alert(rc, populated_db, dates_file, tmp_path, monkeypatch):
    """alerts 非空时,Layer4 注入函数必须产出回退警讯文本（带处置指引）。"""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    rc.run_regression_check(populated_db, dates_file)
    alerts_path = tmp_path / "outcomes" / "regression_alerts.json"
    assert alerts_path.exists()  # 期望: 前置——alerts 已落盘
    init = _load("oc_under_test", "__init__.py")
    ctx = init._regression_alerts_context()
    assert ctx is not None  # 期望: 未处置警讯存在→注入非 None
    assert "R6" in ctx  # 期望: 注入文本指名回退规则
    assert "处置" in ctx  # 期望: 注入文本携带处置指引


def test_no_alerts_when_effective(rc, tmp_path, monkeypatch):
    """无数据/不显著时不得产生噪音警讯。"""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = tmp_path / "empty.db"
    conn = sqlite3.connect(str(db))
    try:
        conn.execute("CREATE TABLE violations (rule TEXT, command TEXT, level TEXT,"
                     " session_id TEXT, timestamp TEXT)")
        conn.execute("CREATE TABLE tool_outcomes (session_id TEXT, tool_name TEXT,"
                     " status TEXT, timestamp TEXT)")
        conn.commit()
    finally:
        _close_quietly(conn)
    p = tmp_path / "dates.json"
    p.write_text(json.dumps({"R1": "2026-09-01"}), encoding="utf-8")
    rc.run_regression_check(db, p)
    alerts_path = tmp_path / "outcomes" / "regression_alerts.json"
    data = json.loads(alerts_path.read_text(encoding="utf-8"))
    assert not any(a["rule"] == "R1" for a in data["alerts"])  # 期望: 零数据→inconclusive→不挂警


def test_rewrite_history_in_findings(tmp_path, monkeypatch):
    """R6 近7天改写次数必须出现在分析报告的行为修正段。"""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = tmp_path / "o.db"
    conn = sqlite3.connect(str(db))
    try:
        conn.execute("CREATE TABLE violations (rule TEXT, command TEXT, level TEXT,"
                     " session_id TEXT, timestamp TEXT,"
                     " outcome TEXT NOT NULL DEFAULT 'rewrite')")
        # 造数必须相对「当下」：硬编码日期会随日历滑出 days=7 窗口，把测试变成日期炸弹
        recent = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%dT10:00:00")
        for _ in range(14):
            conn.execute(
                "INSERT INTO violations(rule, command, level, session_id, timestamp)"
                " VALUES ('R6','x','L1','s',:ts)",
                {"ts": recent})
        conn.execute("CREATE TABLE tool_outcomes (id INTEGER PRIMARY KEY AUTOINCREMENT,"
                     " session_id TEXT, turn_id TEXT,"
                     " tool_call_id TEXT, tool_name TEXT, status TEXT,"
                     " error_type TEXT, error_message TEXT,"
                     " duration_ms INTEGER, timestamp TEXT)")
        conn.commit()
    finally:
        _close_quietly(conn)
    analyze = _load("analyze_under_test", "analyze.py")
    report = analyze.run_analysis(db, days=7)
    section = report.get("behavior_rewrites", {})
    assert section.get("R6_rewrite_last7d") == 14  # 期望: 插入14条R6改写→统计数=14


def test_watchdog_covers_audit_freshness():
    """repo 侧 freshness 自检脚本必须含审计产出检查——没有=监测者自身无监测。

    watchdog 部署于 ~/.hermes/scripts/（home_io_guard 禁测试直读）,其 repo 侧
    真相源是 scripts/check_flywheel_freshness.py——本测试钉死该真相源存在且
    覆盖 audit-*/regression_* 两类产出。
    """
    checker = Path(__file__).resolve().parents[2] / "plugins" / "outcome-collector" / "flywheel_freshness.py"
    assert checker.exists()  # 期望: freshness 自检脚本存在于插件目录
    body = checker.read_text(encoding="utf-8")
    assert "regression_alerts" in body or "regression-" in body  # 期望: 覆盖 regression 产出
    assert "audit-" in body  # 期望: 覆盖每日审计产出


def _seed_stale_flywheel_home(home: Path) -> None:
    """构造三项产出全部过期(48h>26h/12h 阈值)的假 home。"""
    old = datetime.now().timestamp() - 48 * 3600
    sug = home / "skill_suggestions"
    sug.mkdir(parents=True)
    audit = sug / "audit-20260101.md"
    audit.write_text("stale", encoding="utf-8")
    os.utime(audit, (old, old))
    out = home / "outcomes"
    out.mkdir(parents=True)
    for name in ("regression_alerts.json", "findings.md"):
        p = out / name
        p.write_text("stale", encoding="utf-8")
        os.utime(p, (old, old))


def _write_jobs(home: Path, enabled_map: dict) -> None:
    """Contract: 写出仅含 enabled_map 条目的 jobs.json 供停用门判定。"""
    cron = home / "cron"
    cron.mkdir(parents=True, exist_ok=True)
    (cron / "jobs.json").write_text(
        json.dumps({"jobs": [{"id": jid, "enabled": en} for jid, en in enabled_map.items()]}),
        encoding="utf-8",
    )


_PRODUCER_IDS = ("0ee8fec495c8", "a7c3f9d2e1b4", "72e85e6646fe")
_STALE_NAMES = ("daily-audit", "regression-alerts", "analyzer-findings")


def test_stale_but_all_producers_disabled_not_alerted(tmp_path: Path):
    """生产 job 全部显式停用时,产出静默属预期——三项过期不得进 alerts。"""
    _seed_stale_flywheel_home(tmp_path)
    _write_jobs(tmp_path, {jid: False for jid in _PRODUCER_IDS})
    ff = _load("ff_under_test", "flywheel_freshness.py")
    result = ff.check_freshness(tmp_path)
    alerted = {a["name"] for a in result["alerts"]}
    for name in _STALE_NAMES:
        assert name not in alerted  # 期望: 生产者停用→判活预期收敛→不告警
    reasons = {c["name"]: c["reason"] for c in result["checked"]}
    assert all(reasons.get(n) == "producer-disabled" for n in _STALE_NAMES)  # 期望: 停用门留下可审计标记


def test_stale_and_producer_enabled_still_alerted(tmp_path: Path):
    """任一生产 job 启用时,该产出必须照常告警——停用门不得吞真断流。

    门按产出逐项判定:仅 daily-audit 的生产者启用时,daily-audit 告警,
    其余两项(生产者仍停用)消警。
    """
    _seed_stale_flywheel_home(tmp_path)
    enabled_map = {jid: False for jid in _PRODUCER_IDS}
    enabled_map["0ee8fec495c8"] = True
    _write_jobs(tmp_path, enabled_map)
    ff = _load("ff_under_test", "flywheel_freshness.py")
    result = ff.check_freshness(tmp_path)
    alerted = {a["name"] for a in result["alerts"]}
    assert "daily-audit" in alerted  # 期望: 该项生产者启用→过期照常告警
    assert "regression-alerts" not in alerted  # 期望: 该项生产者停用→消警
    assert "analyzer-findings" not in alerted  # 期望: 该项生产者停用→消警


def test_stale_and_jobs_json_unresolvable_still_alerted(tmp_path: Path):
    """jobs.json 缺失(不可判定)时必须保持原告警——fail-loud 保守侧。"""
    _seed_stale_flywheel_home(tmp_path)
    ff = _load("ff_under_test", "flywheel_freshness.py")
    result = ff.check_freshness(tmp_path)
    alerted = {a["name"] for a in result["alerts"]}
    assert set(_STALE_NAMES).issubset(alerted)  # 期望: 无法判定停用→照常告警
