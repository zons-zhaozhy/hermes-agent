# dupcheck-ok: 全库 search_files guards_preflight|section_reinjection 命中仅本插件代码与测试，
# 无等价测试实现（原段文本常量测试随段文本外置同步重写）
"""文件型段与重注入通道的回归测试.

背景: 段文本原以模块级常量冻结在各插件内, 段修改对在跑的会话无效(实测: 进程
11:15 启动 / 段 16:15 提交 → 同会话渲染日志 4 段, 同刻新进程 5 段)。改造后段文本
落在段库 prompt_sections/sections.json, 渲染器每次读盘; 在跑的会话由
discipline/section_reinjection 按版本指纹补注。本测试锁三件事:

  1. 段库键与注册 id 一一对应（键缺失=该段静默不注入, 属两态状态）;
  2. 各套件注册的是「读段库的渲染器」, 段文本逐条覆盖判据并带阈值/豁免;
  3. 重注入节奏: 首轮不注入 / 指纹变更即注入 / 每 10 轮补注。
"""
from __future__ import annotations

from typing import Any

from plugins import discipline, guards
from plugins.discipline import section_reinjection
from plugins.prompt_sections import fingerprint, load_section, section_ids

_EXPECTED_SECTIONS = frozenset({
    "guards_preflight",
    "tool_blacklist",
    "db_safety",
    "test_discipline",
    "scientific_programming",
    "diff_debt",
})

_GUARD_CLAUSES = (
    "先读后写",
    "四轴证据",
    "新建查重",
    "批量写红线",
    "病例库",
    "新目录树",
    "源码直写",
)


class _StubCtx:
    """最小插件上下文桩: 记录钩子与 system prompt 段注册。"""

    def __init__(self) -> None:
        self.hooks: dict[str, list[Any]] = {}
        self.sections: dict[str, Any] = {}

    def register_hook(self, name: str, callback: Any) -> None:
        self.hooks.setdefault(name, []).append(callback)

    def register_system_prompt_section(self, section_id: str, renderer: Any) -> None:
        self.sections[section_id] = renderer


def _register_guards() -> _StubCtx:
    ctx = _StubCtx()
    guards.register(ctx)
    return ctx


def test_section_library_keys_match_registrations() -> None:
    """段库键齐备且可读（键缺失=该段静默不注入）。"""
    ids = set(section_ids())
    missing = _EXPECTED_SECTIONS - ids
    assert not missing  # 期望: 六个段键全部在段库内(键名=注册 id)
    assert fingerprint() != ""  # 期望: 段库可读, 指纹非空


def test_guards_registers_file_backed_sections() -> None:
    """guards 套件注册两个文件型段, 且不挤掉原有执法钩子。"""
    ctx = _register_guards()
    assert "guards_preflight" in ctx.sections  # 期望: 写前功课段已注册
    assert "tool_blacklist" in ctx.sections  # 期望: 必败黑名单段已注册
    assert ctx.hooks.get("pre_tool_call")  # 期望: 执法钩子未受影响
    assert ctx.hooks.get("post_tool_call")  # 期望: 执法钩子未受影响


def test_guards_sections_read_library_content() -> None:
    """渲染器返回段库内容: 判据逐条在段内并带阈值与豁免。"""
    ctx = _register_guards()
    text = ctx.sections["guards_preflight"]({"platform": "cli"})
    for clause in _GUARD_CLAUSES:
        assert clause in text  # 期望: 每条判据在段内(六个子护栏全覆盖)
    assert "达 2 个即拦" in text  # 期望: 批量改已有文件阈值=2, 与 batch_write 实现一致
    assert "≤8 个代码文件" in text  # 期望: 单 commit 文件数上限=8, 与 batch_write 一致
    assert "30 分钟" in text  # 期望: 四轴 marker 有效期=30 分钟(four_axis._MARKER_MAX_AGE_SECONDS=1800, 600s 旧值实测不够已调), 与段库/架构文档一致
    assert "terminal 豁免" in text  # 期望: 四轴对 terminal 的豁免, 与主门禁一致
    blacklist = ctx.sections["tool_blacklist"]({"platform": "cli"})
    assert "vision_analyze" in blacklist  # 期望: 必败调用在段内
    assert "mcp__glm_vision__analyze_image" in blacklist  # 期望: 段内给出正解工具


def test_sections_skip_platforms_without_write_loop() -> None:
    """子代理/批处理会话返回空段（空段被 core 跳过, 不诱导空转）。"""
    guards_ctx = _register_guards()
    discipline_ctx = _StubCtx()
    discipline.register(discipline_ctx)
    guard_render = guards_ctx.sections["guards_preflight"]
    disc_render = discipline_ctx.sections["db_safety"]
    assert guard_render({"platform": "subagent"}) == ""  # 期望: 排除平台返回空
    assert disc_render({"platform": "batch"}) == ""  # 期望: 排除平台返回空
    assert guard_render({"platform": "cli"}) != ""  # 期望: 常规会话注入非空段


def test_discipline_registers_sections_and_reinjection() -> None:
    """discipline 套件注册两个文件型段与重注入子守卫（pre_llm_call 通道）。"""
    ctx = _StubCtx()
    discipline.register(ctx)
    assert "db_safety" in ctx.sections  # 期望: SQL 前先确认 schema 段已注册
    assert "test_discipline" in ctx.sections  # 期望: 测试走包装器段已注册
    assert ctx.hooks.get("pre_llm_call")  # 期望: 重注入通道已挂 pre_llm_call


def test_coding_section_writes_measured_violation_forms() -> None:
    """科学编程段写清三类实测高频违规的精确形态（此前只写「同线」仍被拦）。"""
    text = load_section("scientific_programming")
    # 期望: 期望注释的精确位置(同一行行末)在段内——放上一行会被拦
    assert "assert 同一行行末" in text
    # 期望: 引用本地模块前先读源码的要求在段内
    assert "前先 read_file 该模块源码" in text
    # 期望: finally 内清理调用须包 try/except 的要求在段内
    assert "finally 里的 close" in text
    # 期望: 函数体预算仍在段内(改造不得丢既有判据)
    assert "圈复杂度 ≤10" in text


def test_reinjection_skips_first_turn_then_injects_on_change() -> None:
    """重注入节奏: 首轮不注入(起点已有同名段) / 指纹变更即注入。"""
    session_id = "test-reinjection-change"
    section_reinjection._session_state.pop(session_id, None)
    first = section_reinjection._on_pre_llm_call(session_id=session_id)
    assert first == {}  # 期望: 首轮只登记指纹不注入(避免与 system prompt 段重复)
    same = section_reinjection._on_pre_llm_call(session_id=session_id)
    assert same == {}  # 期望: 指纹未变且未到周期轮, 不注入
    section_reinjection._session_state[session_id]["version"] = "stale-version"
    changed = section_reinjection._on_pre_llm_call(session_id=session_id)
    assert "context" in changed  # 期望: 指纹变更即注入
    assert "唯一权威版本" in changed["context"]  # 期望: 带冲突消解头
    assert load_section("guards_preflight")[:20] in changed["context"]  # 期望: 段文本随注入


def test_reinjection_periodic_after_interval() -> None:
    """周期补注: 一个周期内注入一次（对抗长会话纪律衰减）。"""
    session_id = "test-reinjection-periodic"
    section_reinjection._session_state.pop(session_id, None)
    payloads: list[dict[str, Any]] = []
    for _ in range(section_reinjection._REINJECT_INTERVAL):
        payloads.append(section_reinjection._on_pre_llm_call(session_id=session_id))
    injected = [p for p in payloads if p]
    assert len(injected) == 1  # 期望: 一个周期内恰好注入一次(第 N 轮)
    assert "context" in injected[0]  # 期望: 注入体为 context 字段
