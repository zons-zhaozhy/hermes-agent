"""dynamic_tips 动态提示段回归测试。

期望值来源: 双保险闭环业务语义——①排除平台不触库 ②聚合只收映射命中的
签名且带实测次数 ③未映射签名宁缺毋滥不进段 ④新鲜缓存直用不触库
⑤库缺失/无缓存 fail-open 空串 ⑥聚合失败旧缓存兜底。
预算截断分支（>600 字符）豁免测试: _TIP_LIMIT=5 与现有 _TIP_MAP 最长
条目（约 90 字符）联合保证 5 条总长 < 600,该分支为防御性兜底,当前
输入空间不可达。
"""

import json
import sqlite3
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from plugins.guards import dynamic_tips as dt


def _make_db(path: Path, rows: list[tuple[str, int]]) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE tool_outcomes (session_id TEXT NOT NULL, tool_name TEXT NOT NULL,"
        " status TEXT NOT NULL, error_type TEXT, error_message TEXT, timestamp TEXT NOT NULL)"
    )
    for msg, n in rows:
        for _ in range(n):
            conn.execute(
                "INSERT INTO tool_outcomes VALUES ('s','terminal','blocked',NULL,?,"
                "'2026-11-03T00:00:00+00:00')",
                (msg,),
            )
    conn.commit()
    conn.close()


def _wire(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, rows: list[tuple[str, int]]
) -> Path:
    db = tmp_path / "outcomes.db"
    _make_db(db, rows)
    cache = tmp_path / "guard_dynamic_tips.json"
    monkeypatch.setattr(dt, "_db_path", lambda: db)
    monkeypatch.setattr(dt, "_cache_path", lambda: cache)
    return cache


def test_render_platform_excluded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """排除平台直接空串,不触发聚合（subagent/batch 无写代码主循环）。"""
    called = []
    monkeypatch.setattr(dt, "_aggregate_tips", lambda: called.append(1) or "x")
    assert dt.render({"platform": "subagent"}) == ""  # 期望: 排除平台空串
    assert dt.render({"platform": "batch"}) == ""  # 期望: 排除平台空串
    assert called == []  # 期望: 聚合未被调用


def test_aggregate_maps_signatures_with_counts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """护栏前缀消息进段且带实测次数+消息原文;无前缀噪音不进段。"""
    _wire(
        monkeypatch,
        tmp_path,
        [("[SlowSearchGuard] terminal 中检测到无保护的递归 grep, 请限定子目录", 3),
         ("[DBSafety] SQL 查询前必须先确认 schema", 2),
         ("callback timed out 通道超时噪音", 5)],
    )
    out = dt.render({"platform": "cli"})
    assert "- SlowSearchGuard(3次):" in out  # 期望: 护栏标识+次数聚合
    assert "[SlowSearchGuard] terminal 中检测到无保护的递归 grep" in out  # 期望: 消息原文(修复指引)进段
    assert "- DBSafety(2次):" in out  # 期望: 第二护栏也进段
    assert "callback timed out" not in out  # 期望: 无 [护栏] 前缀的噪音不进段
    assert "[高频拦截教训" in out  # 期望: 带日期头


def test_unmapped_only_returns_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """全部为无前缀消息 → 空段（基础设施噪音不进提示段）。"""
    _wire(monkeypatch, tmp_path, [("callback timed out", 9)])
    assert dt.render({"platform": "cli"}) == ""  # 期望: 无护栏前缀空串


def test_fresh_cache_short_circuits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """新鲜缓存直接复用,不触发聚合（7 天周期内输出恒定）。"""
    cache = _wire(monkeypatch, tmp_path, [])
    cache.write_text(
        json.dumps({"generated": time.time() - 60, "text": "CACHED_TIP"}, ensure_ascii=False),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        dt, "_aggregate_tips", lambda: (_ for _ in ()).throw(AssertionError("不应触库"))
    )
    out = dt.render({"platform": "cli"})
    assert "CACHED_TIP" in out  # 期望: 缓存文本直用
    assert out.startswith("\n")  # 期望: 追加式段以换行衔接段本体


def test_db_missing_fail_open(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """库缺失且无缓存 → 空串（fail-open,段本体不受影响）。"""
    monkeypatch.setattr(dt, "_db_path", lambda: tmp_path / "nope.db")
    monkeypatch.setattr(dt, "_cache_path", lambda: tmp_path / "nope.json")
    assert dt.render({"platform": "cli"}) == ""  # 期望: 数据面故障空串


def test_stale_cache_used_when_aggregate_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """聚合失败（库锁死）时旧缓存兜底——宁旧勿空。"""
    cache = tmp_path / "guard_dynamic_tips.json"
    cache.write_text(
        json.dumps({"generated": time.time() - 999999, "text": "OLD_TIP"}, ensure_ascii=False),
        encoding="utf-8",
    )
    monkeypatch.setattr(dt, "_db_path", lambda: tmp_path / "nope.db")
    monkeypatch.setattr(dt, "_cache_path", lambda: cache)
    assert "OLD_TIP" in dt.render({"platform": "cli"})  # 期望: 旧缓存兜底


def test_reinjection_fingerprint_includes_dynamic_tips(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """动态段 7 天周期翻转 → 补注指纹变化 → 在跑会话收到新动态段。

    期望值来源: 双保险闭环必须同达新旧会话——动态段不在段库文件里,
    补注通道须单独并入其指纹,否则周期翻转只在下个新会话生效。
    """
    from plugins.discipline import section_reinjection as sr

    _wire(monkeypatch, tmp_path, [])
    sid = "test-dyn-reinject"
    sr._session_state.pop(sid, None)
    monkeypatch.setattr(dt, "peek", lambda: "DYN_V1")
    assert sr._on_pre_llm_call(session_id=sid) == {}  # 期望: 首轮登记不注入
    assert sr._on_pre_llm_call(session_id=sid) == {}  # 期望: 指纹未变不注入
    monkeypatch.setattr(dt, "peek", lambda: "DYN_V2")
    out = sr._on_pre_llm_call(session_id=sid)
    assert "DYN_V2" in out["context"]  # 期望: 动态段变化即对在跑会话补注
