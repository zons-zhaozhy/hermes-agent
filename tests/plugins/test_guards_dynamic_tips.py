"""dynamic_tips 动态提示段回归测试。

期望值来源: 双保险闭环业务语义——①排除平台不触库 ②聚合只输出已知护栏的
短正解并带实测次数 ③未映射签名与原始拦截文案不进段 ④最多 3 条且总量
≤300 字符 ⑤新鲜缓存直用不触库 ⑥库缺失 fail-open ⑦聚合失败用同版本
旧缓存兜底。
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
    """已知护栏输出短正解;原始拦截文案、未知护栏和噪音不进段。"""
    _wire(
        monkeypatch,
        tmp_path,
        [("[SlowSearchGuard] terminal 中检测到无保护的递归 grep, 请限定子目录", 5),
         ("[DBSafety] SQL 查询前必须先确认 schema", 2),
         ("[PreWriteGuard] write_file/patch/execute_code 前必须先读目标文件", 1),
         ("[UnknownGuard] 新护栏原文不进提示段", 9),
         ("callback timed out 通道超时噪音", 4)],
    )
    out = dt.render({"platform": "cli"})
    assert "- SlowSearchGuard(5次): 用 search_files；terminal grep 加 timeout 和排除目录。" in out
    assert "- DBSafety(2次): 先查表结构，再执行 SQL。" in out
    assert "- PreWriteGuard(1次): 写前先读目标文件。" in out
    assert "UnknownGuard" not in out
    assert "terminal 中检测到" not in out
    assert "callback timed out" not in out  # 期望: 无 [护栏] 前缀的噪音不进段
    assert "[近期高频拦截教训]" in out  # 期望: 带短日期头
    assert out.count("\n- ") <= dt._TIP_LIMIT  # 期望: 最多 3 条
    assert len(out.strip()) <= dt._BUDGET_CHARS  # 期望: 动态提示总量受控


def test_unmapped_only_returns_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """全部为无前缀消息 → 空段（基础设施噪音不进提示段）。"""
    _wire(monkeypatch, tmp_path, [("callback timed out", 9), ("[UnknownGuard] x", 8)])
    assert dt.render({"platform": "cli"}) == ""  # 期望: 未映射与噪音均不进段


def test_fresh_cache_short_circuits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """新鲜缓存直接复用,不触发聚合（7 天周期内输出恒定）。"""
    cache = _wire(monkeypatch, tmp_path, [])
    cache.write_text(
        json.dumps(
            {"generated": time.time() - 60, "text": "CACHED_TIP", "version": dt._CACHE_VERSION},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        dt, "_aggregate_tips", lambda: (_ for _ in ()).throw(AssertionError("不应触库"))
    )
    out = dt.render({"platform": "cli"})
    assert "CACHED_TIP" in out  # 期望: 缓存文本直用
    assert out.startswith("\n")  # 期望: 追加式段以换行衔接段本体


def test_utf8_bom_cache_short_circuits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Windows 工具写入的 BOM 缓存仍是新鲜缓存,直接复用而不是触库重建。"""
    cache = _wire(monkeypatch, tmp_path, [])
    cache.write_text(
        json.dumps(
            {"generated": time.time() - 60, "text": "BOM_TIP", "version": dt._CACHE_VERSION},
            ensure_ascii=False,
        ),
        encoding="utf-8-sig",
    )
    monkeypatch.setattr(
        dt, "_aggregate_tips", lambda: (_ for _ in ()).throw(AssertionError("不应触库"))
    )
    assert "BOM_TIP" in dt.render({"platform": "cli"})


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
    """聚合失败（库锁死）时同版本旧缓存兜底——宁旧勿空。"""
    cache = tmp_path / "guard_dynamic_tips.json"
    cache.write_text(
        json.dumps(
            {"generated": time.time() - 999999, "text": "OLD_TIP", "version": dt._CACHE_VERSION},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(dt, "_db_path", lambda: tmp_path / "nope.db")
    monkeypatch.setattr(dt, "_cache_path", lambda: cache)
    assert "OLD_TIP" in dt.render({"platform": "cli"})  # 期望: 旧缓存兜底


def test_legacy_cache_is_rebuilt_instead_of_replaying_raw_tips(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """旧版缓存没有版本号,不得把 600 字符原始拦截文案继续注入新会话。"""
    cache = _wire(monkeypatch, tmp_path, [("[DBSafety] SQL 查询前必须先确认 schema", 2)])
    cache.write_text(
        json.dumps({"generated": time.time() - 60, "text": "OLD_RAW_TIP"}, ensure_ascii=False),
        encoding="utf-8",
    )
    out = dt.render({"platform": "cli"})
    assert "OLD_RAW_TIP" not in out
    assert "- DBSafety(2次):" in out


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
    monkeypatch.setattr(sr, "_active_section_ids", lambda: frozenset({"guards_preflight"}))
    monkeypatch.setattr(dt, "peek", lambda: "DYN_V1")
    assert sr._on_pre_llm_call(session_id=sid) == {}  # 期望: 首轮登记不注入
    assert sr._on_pre_llm_call(session_id=sid) == {}  # 期望: 指纹未变不注入
    monkeypatch.setattr(dt, "peek", lambda: "DYN_V2")
    out = sr._on_pre_llm_call(session_id=sid)
    assert "DYN_V2" in out["context"]  # 期望: 动态段变化即对在跑会话补注
