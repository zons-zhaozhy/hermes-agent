"""terminology-guard 回归测试。

覆盖四条行为契约（期望值独立推导，非实现反推）：
1. 确定性别名检测：CJK 子串/ASCII 整词两形态，大小写不敏感；标识符不被拆词误伤
2. 注入预算与三态：空全空→None；术语行预算封顶；漂移行必放
3. 跨轮闭环：命中→pending→下轮注入纠正行并消费标记（一次即消）
4. fail-open：judge 挂 → 插件不 raise、不注入垃圾；禁用开关空转

Contract:
  Preconditions: judge 全部 monkeypatch（零真实 LLM 调用）；独立会话 state
  Postconditions: 全部断言通过 = 四契约成立
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PLUGIN_DIR = _REPO_ROOT / "plugins" / "terminology-guard"


def _load_plugin() -> Any:
    """按 PluginManager 命名约定加载插件 __init__.py。"""
    if "hermes_plugins" not in sys.modules:
        ns = types.ModuleType("hermes_plugins")
        ns.__path__ = []
        sys.modules["hermes_plugins"] = ns
    spec = importlib.util.spec_from_file_location(
        "hermes_plugins.terminology_guard",
        _PLUGIN_DIR / "__init__.py",
        submodule_search_locations=[str(_PLUGIN_DIR)],
    )
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "hermes_plugins.terminology_guard"
    mod.__path__ = [str(_PLUGIN_DIR)]
    sys.modules["hermes_plugins.terminology_guard"] = mod
    spec.loader.exec_module(mod)
    return mod


class _FakeCtx:
    """收集 register_hook 注册的钩子，供测试直接驱动。"""

    def __init__(self) -> None:
        self.hooks: dict[str, Any] = {}

    def register_hook(self, name: str, fn: Any) -> None:
        self.hooks[name] = fn


def _mk(mod: Any, judge_result: Any, sid: str = "s1") -> tuple[Any, str]:
    """加载插件并挂好钩子；judge 结果注入。"""
    ctx = _FakeCtx()
    mod.register(ctx)

    def fake_multi(task: str, system: str, text: str, keys: Any, **kw: Any) -> Any:
        return {k: judge_result for k in keys}

    mod.llm_judge_multi = fake_multi
    return ctx, sid


# ── 1. 确定性别名检测 ──────────────────────────────────────────────────────

def test_cjk_alias_substring() -> None:
    mod = _load_plugin()
    entries = [{"canonical": "DBChat", "aliases": ["dbchat 问数", "问数模块"]}]
    # 期望: CJK 别名走子串命中，返回 (别名, 规范名) 对
    got = mod.detect_alias_drift("我们用问数模块查表", entries)
    assert got == [("问数模块", "DBChat")]  # 期望: 表内别名「问数模块」子串命中
    got2 = mod.detect_alias_drift("正常句子", entries)
    assert got2 == []  # 期望: 无别名出现 → 零命中


def test_mixed_and_pure_ascii_alias() -> None:
    mod = _load_plugin()
    entries = [{"canonical": "OntoX", "aliases": ["ontoX 平台"]}]
    got = mod.detect_alias_drift("ontoX 平台已部署", entries)
    assert got == [("ontoX 平台", "OntoX")]  # 期望: 混合别名（含 CJK）走子串
    entries2 = [{"canonical": "PostgreSQL", "aliases": ["postgres"]}]
    got2 = mod.detect_alias_drift("连 Postgres 库", entries2)
    assert got2 == [("postgres", "PostgreSQL")]  # 期望: 纯 ASCII 别名整词命中，大小写不敏感


def test_identifier_not_split() -> None:
    mod = _load_plugin()
    entries = [{"canonical": "OntoX", "aliases": ["ontox"]}]
    got = mod.detect_alias_drift("import ontox_protocols.schemas", entries)
    assert got == []  # 期望: ontox_protocols 是整 token，"ontox" 整词比对不命中（_ 与 . 不拆词）


# ── 2. 注入预算与三态 ──────────────────────────────────────────────────────

def test_build_injection_empty() -> None:
    mod = _load_plugin()
    got = mod.build_injection([], [], False)
    assert got is None  # 期望: 术语/漂移/开放漂移三者皆空 → 不注入


def test_build_injection_budget() -> None:
    mod = _load_plugin()
    # 150 条 × ~20 字符/行 = 3000 > 预算 2000 → 必截断；50 条装得下 → 不截断
    many = [
        {"canonical": f"术语{i}", "aliases": [f"旧写法{i}"]} for i in range(150)
    ]
    out = mod.build_injection(many, [], False)
    assert out is not None and out.startswith(mod._INJECT_HEAD)  # 期望: 有表头
    count = out.count("- 术语")
    assert count < 150  # 期望: 150 条超预算 → 截断生效
    head_part = out.split("\n你上一条")[0]
    assert len(head_part) <= mod._INJECT_BUDGET + 60  # 期望: 表头+术语行 ≤ 预算+首行容差
    few = [
        {"canonical": f"词{i}", "aliases": [f"旧{i}"]} for i in range(50)
    ]
    out2 = mod.build_injection(few, [], False)
    assert out2.count("- 词") == 50  # 期望: 50 条 × ~12 字符 = 600 < 2000 → 全量注入零截断


def test_build_injection_drift_lines() -> None:
    mod = _load_plugin()
    out1 = mod.build_injection([], [("旧称", "规范称")], False)
    assert "「旧称」→ 用「规范称」" in out1  # 期望: 表漂移给具体纠正对
    out2 = mod.build_injection([], [], True)
    assert "统一用词" in out2  # 期望: 开放集漂移给泛化纠正行


# ── 3. 跨轮闭环 ────────────────────────────────────────────────────────────

def test_cross_turn_loop(monkeypatch: Any) -> None:
    mod = _load_plugin()
    ctx, sid = _mk(mod, judge_result=False)
    entries = [{"canonical": "名单筛查", "aliases": ["名单扫描"]}]
    monkeypatch.setattr(mod, "_load_glossary", lambda: entries)
    ret = ctx.hooks["transform_llm_output"]("做名单扫描后入库", session_id=sid)
    assert ret is None  # 期望: 用户可见回复零改动
    st = mod._state(sid)
    assert st.get("pending_drifts") == [("名单扫描", "名单筛查")]  # 期望: 别名命中已记账
    injected = ctx.hooks["pre_llm_call"](session_id=sid)
    assert injected is not None and "名单筛查" in injected["context"]  # 期望: 注入含规范名
    assert "「名单扫描」→ 用「名单筛查」" in injected["context"]  # 期望: 纠正行含具体对照
    again = ctx.hooks["pre_llm_call"](session_id=sid)
    assert again is not None and "上一条回复称谓漂移" not in again["context"]  # 期望: 标记一次即消费


def test_open_drift_loop(monkeypatch: Any) -> None:
    mod = _load_plugin()
    ctx, sid = _mk(mod, judge_result=True)
    monkeypatch.setattr(mod, "_load_glossary", list)
    ctx.hooks["transform_llm_output"]("我们先用查询器把全量名单过了一遍，随后又用检索工具查了第二遍，两轮口径不一致。", session_id=sid)
    injected = ctx.hooks["pre_llm_call"](session_id=sid)
    assert injected is not None and "统一用词" in injected["context"]  # 期望: judge=true → 泛化纠正注入


# ── 3.5 会话用词账本（L2 开放集根治件）────────────────────────────────────

def test_ledger_lock_and_inject(monkeypatch: Any) -> None:
    mod = _load_plugin()
    ctx = _FakeCtx()
    mod.register(ctx)
    monkeypatch.setattr(mod, "_load_glossary", list)
    monkeypatch.setattr(
        mod, "llm_judge_multi",
        lambda task, system, text, keys, **kw: {k: False for k in keys},
    )
    monkeypatch.setattr(
        mod, "llm_judge_json",
        lambda task, system, text, keys, **kw: {"key_terms": ["评分卡", "准入阈值"]},
    )
    long_reply = "本次评审采用评分卡口径，准入阈值按最新监管指引调整，全部结论已复核落库。" * 2
    ctx.hooks["transform_llm_output"](long_reply, session_id="sL")
    st = mod._state("sL")
    # 期望: 首次抽取的词锁进账本（顺序保持，无重复）
    assert st.get("ledger") == ["评分卡", "准入阈值"]  # 期望: judge_json 返回词全入账
    injected = ctx.hooks["pre_llm_call"](session_id="sL")
    assert injected is not None  # 期望: 账本非空 + 时态规则行 → 注入必非 None
    assert "评分卡" in injected["context"]  # 期望: 账本词进注入
    assert "已确立用词" in injected["context"]  # 期望: 账本行表头在
    assert "时态纪律" in injected["context"]  # 期望: 时态规则行恒注入


def test_ledger_cap(monkeypatch: Any) -> None:
    mod = _load_plugin()
    monkeypatch.setattr(
        mod, "llm_judge_json",
        lambda task, system, text, keys, **kw: {"key_terms": [f"词{i}" for i in range(40)]},
    )
    st = mod._state("sCap")
    mod._update_ledger("这是一段足够长的回复，用于触发账本抽取路径的长度门槛检验。" * 2, "sCap", st)
    # 期望: 40 词入账被截到 _LEDGER_CAP=24
    assert len(st.get("ledger") or []) == mod._LEDGER_CAP


def test_ledger_short_reply_skipped() -> None:
    mod = _load_plugin()
    st = mod._state("sShort")

    def no_call(task: str, system: str, text: str, keys: Any, **kw: Any) -> Any:
        raise AssertionError("短回复不应触发账本抽取调用")

    mod.llm_judge_json = no_call
    mod._update_ledger("太短", "sShort", st)
    # 期望: <40 字符早退，零 judge 调用，账本不建
    assert "ledger" not in st


# ── 5. 语义层固化（跨会话 admission，持久化）──────────────────────────────

def _fake_home(monkeypatch: Any, tmp_path: Any) -> Any:
    mod_dir = tmp_path / "hermes_home"
    mod_dir.mkdir()
    import types as _t
    fake_constants = _t.ModuleType("hermes_constants")
    fake_constants.get_hermes_home = lambda: mod_dir  # type: ignore[attr-defined]
    monkeypatch.setitem(__import__("sys").modules, "hermes_constants", fake_constants)
    return mod_dir


def test_consolidate_three_sessions_persistent(monkeypatch: Any, tmp_path: Any) -> None:
    mod = _load_plugin()
    home = _fake_home(monkeypatch, tmp_path)
    # 模拟进程重启：sessA/sessB 各自独立重载模块（内存态清零），计数只能来自文件
    for sid in ("sessA", "sessB"):
        mod2 = _load_plugin()
        mod2._consolidate(["评分卡"], sid)
    mod._consolidate(["评分卡"], "sessC")
    import yaml as _yaml
    raw = _yaml.safe_load((home / "terminology_consolidated.yaml").read_text(encoding="utf-8"))
    terms = raw.get("terms") or []
    # 期望: 第 3 会话触发固化（admission 计数跨"重启"持久），词入 terms
    assert [e["canonical"] for e in terms] == ["评分卡"]  # 期望: 3 会话达标即固化
    assert "固化" in terms[0]["note"]  # 期望: note 标注固化
    # 期望: 固化后候选计数清出 candidates 段
    assert "评分卡" not in (raw.get("candidates") or {})  # 期望: 晋升即出列
    # 第 4 会话再来同词 → 不重复追加
    mod._consolidate(["评分卡"], "sessD")
    raw2 = _yaml.safe_load((home / "terminology_consolidated.yaml").read_text(encoding="utf-8"))
    # 期望: terms 仍 1 条（已固化词去重）
    assert len(raw2.get("terms") or []) == 1  # 期望: known 集合去重


def test_consolidate_two_sessions_keeps_candidates(monkeypatch: Any, tmp_path: Any) -> None:
    mod = _load_plugin()
    home = _fake_home(monkeypatch, tmp_path)
    mod._consolidate(["临时词"], "sessA")
    mod._consolidate(["临时词"], "sessB")
    import yaml as _yaml
    raw = _yaml.safe_load((home / "terminology_consolidated.yaml").read_text(encoding="utf-8"))
    # 期望: 2 会话未达标 → terms 空，但 candidates 持久化计数在（重启不丢）
    assert (raw.get("terms") or []) == []  # 期望: 2 < 3 阈值
    assert "临时词" in (raw.get("candidates") or {})  # 期望: admission 计数已落盘
    # 期望: 同会话重复出现不重复计会话
    mod._consolidate(["临时词"], "sessA")
    raw2 = _yaml.safe_load((home / "terminology_consolidated.yaml").read_text(encoding="utf-8"))
    assert len(raw2["candidates"]["临时词"]["sessions"]) == 2  # 期望: 去重后仍 2


def test_consolidate_stale_candidate_skipped(monkeypatch: Any, tmp_path: Any) -> None:
    mod = _load_plugin()
    home = _fake_home(monkeypatch, tmp_path)
    mod._consolidate(["旧词"], "s1")
    import yaml as _yaml
    # 手工把 last_seen 改成 40 天前 → 老化命中，跳过晋升
    path = home / "terminology_consolidated.yaml"
    raw = _yaml.safe_load(path.read_text(encoding="utf-8"))
    from datetime import date, timedelta
    raw["candidates"]["旧词"]["last_seen"] = str(date.today() - timedelta(days=40))
    path.write_text(_yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    mod2 = _load_plugin()
    mod2._consolidate(["旧词"], "s2")
    mod2._consolidate(["旧词"], "s3")
    raw2 = _yaml.safe_load(path.read_text(encoding="utf-8"))
    # 期望: stale 候选不晋升（terms 空）
    assert (raw2.get("terms") or []) == []  # 期望: 老化候选禁入 terms


def test_consolidated_line_budget() -> None:
    mod = _load_plugin()
    words = [f"超长术语测试条目{i:03d}" for i in range(60)]
    # 期望: head≈26 + 每词条目 11+3 字符 × 60 ≈ 766 > 400 预算 → 必截断
    line = mod._consolidated_line(words)
    assert line is not None and "跨会话固化词" in line  # 期望: 预算截断非丢弃
    # 期望: 400 预算容纳 ≈(400-26)/14 ≈ 26 条 < 60 → 截断生效
    assert line.count("条目") < 60  # 期望: 未全量装入即截断
    assert len(line) <= 400  # 期望: 行长受预算封顶


# ── 4. fail-open ───────────────────────────────────────────────────────────

def test_fail_open_judge_crash(monkeypatch: Any) -> None:
    mod = _load_plugin()
    ctx = _FakeCtx()
    mod.register(ctx)

    def boom(task: str, system: str, text: str, keys: Any, **kw: Any) -> Any:
        raise RuntimeError("judge down")

    mod.llm_judge_multi = boom
    monkeypatch.setattr(mod, "_load_glossary", list)
    ret = ctx.hooks["transform_llm_output"]("正常回复文本", session_id="s9")
    assert ret is None  # 期望: judge 崩溃不外泄（fail-open），钩子恒 None
    got = ctx.hooks["pre_llm_call"](session_id="s9")
    assert got is None  # 期望: 无术语表无漂移 → 不注入垃圾


def test_plugin_disabled_env(monkeypatch: Any) -> None:
    mod = _load_plugin()
    monkeypatch.setenv("TERMINOLOGY_GUARD_DISABLE", "1")
    ctx, sid = _mk(mod, judge_result=False)
    ret = ctx.hooks["transform_llm_output"]("x", session_id=sid)
    assert ret is None  # 期望: 禁用开关生效
    got = ctx.hooks["pre_llm_call"](session_id=sid)
    assert got is None  # 期望: 禁用态注入也空转
