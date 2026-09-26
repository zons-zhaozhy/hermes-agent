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
import sqlite3
import types
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
                     " session_id TEXT, timestamp TEXT)")
        for i in range(14):
            conn.execute(
                "INSERT INTO violations(rule, command, level, session_id, timestamp)"
                " VALUES ('R6','x','L1','s',:ts)",
                {"ts": f"2026-09-{20 + i // 2:02d}T10:00:00"})
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
    assert section.get("R6_last7d") == 14  # 期望: 插入14条R6→统计数=14


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
