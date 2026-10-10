"""RSI 前沿度量测试（plugins/outcome-collector/frontier_report.py）。

锁定三条机器判据（禁自述、禁关键词）：
  1. 闸门生产率＝拦截命中后「同会话同回合」是否转 ok（跨回合 ok 不算修复）；
  2. 红证覆盖＝规则件在 tests/ 下是否有引用它的测试（没红证的门禁＝假门禁）；
  3. 棘轮＝红证率/有效拦截率下降即 regressed（只增不减）。

Contract:
  Preconditions: 插件目录与模块可 importlib 加载（不依赖 hermes 运行时）
  Postconditions: 九个用例全绿；期望值由判据独立推导，不来自实现
"""

import importlib.util
import json
import sqlite3
import types
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

import pytest

_PLUGIN = Path(__file__).resolve().parents[2] / "plugins" / "outcome-collector"


def _load() -> types.ModuleType:
    """Contract: 按相对路径加载 frontier_report，失败即 raise（不静默跳过）。"""
    spec = importlib.util.spec_from_file_location(
        "frontier_under_test", _PLUGIN / "frontier_report.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载模块: {_PLUGIN / 'frontier_report.py'}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def fr() -> types.ModuleType:
    return _load()


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _seed_db(db: Path, rows: list[tuple[str, str, str, str]]) -> Path:
    """建真实 schema 的 outcomes.db 并写入 (session_id, turn_id, tool, status) 行。

    Contract:
      Preconditions: rows 中 status ∈ ok/error/blocked/rejected
      Postconditions: 库文件存在且含全部行，timestamp=当前 UTC（落在窗口内）
    """
    with closing(sqlite3.connect(str(db))) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS tool_outcomes (id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " session_id TEXT NOT NULL, turn_id TEXT, tool_call_id TEXT,"
            " tool_name TEXT NOT NULL, status TEXT NOT NULL, error_type TEXT,"
            " error_message TEXT, duration_ms INTEGER DEFAULT 0, args_summary TEXT,"
            " timestamp TEXT NOT NULL)"
        )
        for sid, tid, tool, status in rows:
            conn.execute(
                "INSERT INTO tool_outcomes(session_id, turn_id, tool_name, status, timestamp)"
                " VALUES (?, ?, ?, ?, ?)",
                (sid, tid, tool, status, _now()),
            )
        conn.commit()
    return db


def _local_now() -> str:
    """violations 表用的是本地朴素时间戳（与 tool_outcomes 的 UTC 带偏移不同格式）。

    Contract: Preconditions 无；Postconditions 返回 'YYYY-MM-DDTHH:MM:SS'。
    """
    return datetime.now().isoformat(timespec="seconds")


def _seed_violations(db: Path, rows: list[tuple[str, int]]) -> Path:
    """建 violations 表（契约 schema，含 outcome 列）并写入 (rule, 次数)。

    Contract:
      Preconditions: rows 为 (规则码, 次数) 列表。
      Postconditions: 库内恰好含 sum(次数) 行，时间戳=当前本地时刻（落在窗口内）；
        outcome 恒为 'block'（与生产者 no_guessing 的默认口径一致）。
    """
    with closing(sqlite3.connect(str(db))) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS violations (id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " rule TEXT, command TEXT, level TEXT, session_id TEXT, timestamp TEXT,"
            " outcome TEXT NOT NULL DEFAULT 'block')"
        )
        for rule, times in rows:
            for _ in range(times):
                conn.execute(
                    "INSERT INTO violations(rule, command, level, session_id, timestamp)"
                    " VALUES (?, 'x', 'L1', 's', ?)", (rule, _local_now()))
        conn.commit()
    return db


@pytest.fixture()
def world(tmp_path: Path) -> dict[str, Path]:
    """一棵最小世界：guards/discipline/tests 目录 + 一份空库 + 假 home。"""
    guards = tmp_path / "guards"
    discipline = tmp_path / "discipline"
    tests = tmp_path / "tests"
    for d in (guards, discipline, tests):
        d.mkdir(parents=True, exist_ok=True)
    return {"guards": guards, "discipline": discipline, "tests": tests,
            "db": tmp_path / "outcomes.db", "home": tmp_path / "home"}


def test_python_modules_skips_init_and_preflight(fr, world):
    (world["guards"] / "alpha.py").write_text("", encoding="utf-8")
    (world["guards"] / "__init__.py").write_text("", encoding="utf-8")
    (world["guards"] / "preflight.py").write_text("", encoding="utf-8")
    assert fr._python_modules(world["guards"]) == ["alpha"]  # 期望: 只留规则件,__init__/preflight 不当门禁
    assert fr._python_modules(world["guards"] / "nope") == []  # 期望: 目录不存在返回空表不抛


def test_has_red_proof_by_content_reference(fr, world):
    (world["tests"] / "test_alpha.py").write_text(
        "from plugins.guards import alpha\n", encoding="utf-8")
    assert fr._has_red_proof("alpha", world["tests"]) is True  # 期望: 测试正文引用该规则件=有红证
    assert fr._has_red_proof("beta", world["tests"]) is False  # 期望: 无引用=无红证
    assert fr._has_red_proof("alpha", world["tests"] / "nope") is False  # 期望: 测试目录缺失按无红证


def test_has_red_proof_ignores_incidental_mentions(fr, world):
    (world["tests"] / "test_other.py").write_text(
        'p = tmp["guards"] / "alpha.py"\np.write_text("")\n'
        'def test_refuses_removed_plugins_with_alpha(client):\n    pass\n',
        encoding="utf-8")
    assert fr._has_red_proof("alpha", world["tests"]) is False  # 期望: 临时文件名/函数名撞名不算红证


def test_gate_stats_counts_effective_retry_same_turn(fr, world):
    _seed_db(world["db"], [
        ("s1", "t1", "terminal", "blocked"),
        ("s1", "t1", "terminal", "ok"),
        ("s1", "t2", "patch", "blocked"),
        ("s1", "t2", "patch", "blocked"),
        ("s1", "t3", "read_file", "blocked"),
        ("s1", "t4", "read_file", "ok"),
    ])
    stats = fr.compute_gate_stats(world["db"], days=30)
    term = stats["terminal"]
    assert (term["blocks"], term["effective"], term["ineffective"]) == (1, 1, 0)  # 期望: 同回合拦截后转 ok=有效
    patch = stats["patch"]
    assert (patch["blocks"], patch["effective"], patch["ineffective"]) == (2, 0, 2)  # 期望: 命中后没修复=无效
    rf = stats["read_file"]
    assert (rf["blocks"], rf["effective"]) == (1, 0)  # 期望: 跨回合的 ok 不算修复动作


def test_frontier_lists_modules_without_red_proof(fr, world):
    (world["guards"] / "alpha.py").write_text("", encoding="utf-8")
    (world["guards"] / "beta.py").write_text("", encoding="utf-8")
    (world["tests"] / "test_alpha.py").write_text(
        "from plugins.guards import alpha\n", encoding="utf-8")
    report = fr.compute_frontier(world["db"], world["guards"], world["discipline"],
                                 world["tests"], days=30)
    s = report["summary"]
    assert s["zero_red_proof"] == ["beta"]  # 期望: 无红证的规则件逐个点名(beta),alpha 有红证不算
    assert s["red_proof_ratio"] == 0.5  # 期望: 2 个规则件里 1 个有红证
    assert s["modules"] == 2  # 期望: 两个目录合并计数


def test_rule_ledger_counts_hits_and_flags_zero(fr, world):
    (world["guards"] / "coding_standards.py").write_text(
        'RULE = "R013"\nHELP = "R022"\n', encoding="utf-8")
    (world["discipline"] / "no_guessing.py").write_text('RULE = "R6"\n', encoding="utf-8")
    _seed_violations(world["db"], [("R6", 3)])
    report = fr.compute_frontier(world["db"], world["guards"], world["discipline"],
                                 world["tests"], days=30)
    assert report["rule_hits"] == {"R6:block": 3}  # 期望: 规则账按 rule×outcome 分列,别的规则码不进账
    assert report["summary"]["zero_hit_rules"] == ["R013", "R022"]  # 期望: 声明了却零命中的编号被点名


def test_rule_ledger_missing_outcome_column_raises(fr, world, tmp_path):
    """violations 表缺 outcome 列=契约违约，必须响亮报错而非静默空账。"""
    db = tmp_path / "legacy.db"
    with closing(sqlite3.connect(str(db))) as conn:
        conn.execute(
            "CREATE TABLE violations (id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " rule TEXT, command TEXT, level TEXT, session_id TEXT, timestamp TEXT)")
        conn.execute(
            "INSERT INTO violations(rule, command, level, session_id, timestamp)"
            " VALUES ('R6','x','L1','s','2026-01-01T00:00:00')")
        conn.commit()
    with pytest.raises(RuntimeError, match="契约 schema"):
        fr.compute_frontier(db, world["guards"], world["discipline"],
                            world["tests"], days=30)  # 期望: 缺列 raise 不吞


def test_declared_rule_ids_unique_sorted(fr, world):
    (world["guards"] / "coding_standards.py").write_text(
        'A = "R022"\nB = "R013"\nC = "R013"\nD = "Rx"\n', encoding="utf-8")
    assert fr._declared_rule_ids([world["guards"]]) == ["R013", "R022"]  # 期望: 去重排序,非数字编号不收
    assert fr._declared_rule_ids([world["guards"] / "nope"]) == []  # 期望: 目录缺失返回空表不抛


def test_baseline_drop_is_regression(fr, world):
    baseline = world["home"] / "frontier_baseline.json"
    baseline.parent.mkdir(parents=True, exist_ok=True)
    baseline.write_text(json.dumps(
        {"red_proof_ratio": 1.0, "effective_ratio": 1.0}), encoding="utf-8")
    (world["guards"] / "alpha.py").write_text("", encoding="utf-8")
    (world["guards"] / "beta.py").write_text("", encoding="utf-8")
    (world["tests"] / "test_alpha.py").write_text("import alpha\n", encoding="utf-8")
    report = fr.compute_frontier(world["db"], world["guards"], world["discipline"],
                                 world["tests"], days=30)
    regressions = fr.compare_with_baseline(report, baseline)
    assert len(regressions) == 1  # 期望: 只有红证率下降;无命中时有效拦截率按 1.0 不虚报
    assert "红证覆盖率" in regressions[0]  # 期望: 回退项指明具体前沿指标


def test_baseline_missing_or_corrupt_is_quiet(fr, world):
    (world["guards"] / "alpha.py").write_text("", encoding="utf-8")
    report = fr.compute_frontier(world["db"], world["guards"], world["discipline"],
                                 world["tests"], days=30)
    missing = world["home"] / "frontier_baseline.json"
    assert fr.compare_with_baseline(report, missing) == []  # 期望: 首次运行无基线=不回退
    broken = world["home"] / "broken.json"
    broken.parent.mkdir(parents=True, exist_ok=True)
    broken.write_text("{not json", encoding="utf-8")
    assert fr.compare_with_baseline(report, broken) == []  # 期望: 基线损坏按无基线段处理不炸


def test_main_json_exit_code_and_report_write(fr, world):
    (world["guards"] / "alpha.py").write_text("", encoding="utf-8")
    (world["tests"] / "test_alpha.py").write_text("import alpha\n", encoding="utf-8")
    baseline = world["home"] / "frontier_baseline.json"
    baseline.parent.mkdir(parents=True, exist_ok=True)
    clean = fr.main(["--db", str(world["db"]), "--guards", str(world["guards"]),
                     "--discipline", str(world["discipline"]), "--tests", str(world["tests"]),
                     "--baseline", str(baseline), "--no-write"])
    assert clean == 0  # 期望: 无回退=退出码 0(可进 CI)
    baseline.write_text(json.dumps({"red_proof_ratio": 1.0}), encoding="utf-8")
    (world["tests"] / "test_alpha.py").unlink()
    regressed = fr.main(["--db", str(world["db"]), "--guards", str(world["guards"]),
                         "--discipline", str(world["discipline"]), "--tests", str(world["tests"]),
                         "--baseline", str(baseline), "--no-write"])
    assert regressed == 1  # 期望: 红证率 1.0→0.0=前沿回退,退出码 1


def _load_outcome_collector() -> types.ModuleType:
    """Contract: 加载插件入口模块（模块级不依赖 hermes 运行时）；失败即 raise。"""
    spec = importlib.util.spec_from_file_location(
        "outcome_collector_under_test", _PLUGIN / "__init__.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载插件入口: {_PLUGIN / '__init__.py'}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dispositioned_regression_is_silenced(fr, world):
    regressions = ["红证覆盖率 回退：1.0000 → 0.0000", "有效拦截率 回退：1.0000 → 0.5000"]
    dispositions = world["home"] / "outcomes" / "dispositions.json"
    dispositions.parent.mkdir(parents=True, exist_ok=True)
    assert fr._filter_dispositioned(regressions, world["home"]) == regressions  # 期望: 无处置文件全告警
    dispositions.write_text(json.dumps({
        "frontier:红证覆盖率": {"action": "接受当前值", "date": "2026-10-09"}}), encoding="utf-8")
    kept = fr._filter_dispositioned(regressions, world["home"])
    assert kept == ["有效拦截率 回退：1.0000 → 0.5000"]  # 期望: 已登记项消警,未登记项照告警
    dispositions.write_text("{not json", encoding="utf-8")
    assert fr._filter_dispositioned(regressions, world["home"]) == regressions  # 期望: 处置文件损坏宁告警不漏警


def test_frontier_alerts_context_injects_with_disposition_exit(fr, world, monkeypatch):
    oc = _load_outcome_collector()
    alerts = world["home"] / "outcomes" / "frontier_alerts.json"
    alerts.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(world["home"]))
    assert oc._frontier_alerts_context() is None  # 期望: 无警讯文件=不注入(未跑过度量属正常)
    alerts.write_text(json.dumps({"alerts": ["红证覆盖率 回退：1.0000 → 0.0000"]}),
                      encoding="utf-8")
    text = oc._frontier_alerts_context()
    assert text is not None and "红证覆盖率 回退：1.0000 → 0.0000" in text  # 期望: 注入回退原文
    assert "dispositions.json" in text  # 期望: 注入文本必须给出可执行的处置出口


def test_write_outputs_writes_report_and_clears_stale_alert(fr, world):
    (world["guards"] / "alpha.py").write_text("", encoding="utf-8")
    (world["tests"] / "test_alpha.py").write_text("import alpha\n", encoding="utf-8")
    report = fr.compute_frontier(world["db"], world["guards"], world["discipline"],
                                 world["tests"], days=30)
    alerts = world["home"] / "outcomes" / "frontier_alerts.json"
    alerts.parent.mkdir(parents=True, exist_ok=True)
    alerts.write_text(json.dumps({"alerts": ["旧回退"]}), encoding="utf-8")
    path = fr._write_outputs(report, [], world["home"])
    assert path.exists()  # 期望: 报告落盘 outcomes/frontier-<date>.md
    assert not alerts.exists()  # 期望: 无回退清旧警讯,不把历史回退当现状
    assert "工具账（tool_outcomes）" in path.read_text(encoding="utf-8")  # 期望: 报告含工具账段
    fr._write_outputs(report, ["红证覆盖率 回退：1.0000 → 0.0000"], world["home"])
    fresh = json.loads(alerts.read_text(encoding="utf-8"))
    assert fresh["alerts"] == ["红证覆盖率 回退：1.0000 → 0.0000"]  # 期望: 有回退写警讯供首轮注入
