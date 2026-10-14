# dupcheck-ok: 全库 search_files guards_preflight|preflight_section 零命中（仅本插件新代码），
# 无等价测试实现
"""事前注入段与拦截判据同源的回归测试.

背景: guards 套件的六个子护栏此前只有 pre_tool_call 拦截、没有 system prompt 前置段——
实测同一轮里同一判据反复被拦(每拦一次=一轮 API 白跑)。本测试锁两件事:

  1. guards.register 真的注册了 guards_preflight 段, 段文本逐条覆盖六个子护栏的判据,
     且带上阈值与豁免(无数字的提示拦不住具体越界);
  2. scientific_programming 段文本写清三类实测高频违规的精确形态
     (期望注释须与 assert 同一行行末 / 引用本地模块前先读源码 / finally 内 close 须包 try/except)。

段文本漂移会让事前提示与拦截判据脱钩, 故按关键词+数字断言, 不做整段快照。
"""
from __future__ import annotations

import pathlib
from typing import Any

from plugins import guards

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_CODING_SRC = _REPO_ROOT / "plugins" / "scientific-programming-guard" / "__init__.py"

_REQUIRED_CLAUSES = (
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


def _register() -> _StubCtx:
    ctx = _StubCtx()
    guards.register(ctx)
    return ctx


def test_preflight_section_registered() -> None:
    """guards.register 注册写前功课段（事前注入与拦截同源的前置半边）。"""
    ctx = _register()
    assert "guards_preflight" in ctx.sections  # 期望: 段 id 与实现一致, 注册即生效
    assert ctx.hooks.get("pre_tool_call")  # 期望: 加段不得挤掉 pre_tool_call 钩子
    assert ctx.hooks.get("post_tool_call")  # 期望: 加段不得挤掉 post_tool_call 钩子


def test_preflight_section_covers_each_sub_guard() -> None:
    """段文本逐条覆盖有拦截、无前置提示的子护栏。"""
    ctx = _register()
    text = ctx.sections["guards_preflight"]({"platform": "cli"})
    for clause in _REQUIRED_CLAUSES:
        assert clause in text  # 期望: 每条判据在段内(六个子护栏全覆盖)


def test_preflight_section_carries_thresholds_and_exemptions() -> None:
    """段文本必须带数字阈值与豁免条件——无数字的提示拦不住具体越界。"""
    text = guards._PREFLIGHT_SECTION
    assert "达 2 个即拦" in text  # 期望: 批量改已有文件阈值=2, 与 batch_write 实现一致
    assert "≤8 个代码文件" in text  # 期望: 单 commit 文件数上限=8, 与 batch_write 实现一致
    assert "10 分钟" in text  # 期望: 四轴 marker 有效期=10 分钟, 与 four_axis 实现一致
    assert "terminal 豁免" in text  # 期望: 四轴对 terminal 的豁免, 与主闸门 ReadThinkGate 一致


def test_preflight_section_skips_platforms_without_write_loop() -> None:
    """子代理/批处理会话返回空段（空段被 core 跳过, 不诱导空转）。"""
    ctx = _register()
    render = ctx.sections["guards_preflight"]
    assert render({"platform": "subagent"}) == ""  # 期望: 排除平台返回空
    assert render({"platform": "batch"}) == ""  # 期望: 排除平台返回空
    assert render({"platform": "cli"}) != ""  # 期望: 常规会话注入非空段


def test_coding_section_writes_measured_violation_forms() -> None:
    """科学编程段写清三类实测高频违规的精确形态（此前只写「同线」仍被拦）。"""
    text = _CODING_SRC.read_text(encoding="utf-8")
    # 期望: 期望注释的精确位置(同一行行末)在段内——放上一行会被拦
    assert "assert 同一行行末" in text
    # 期望: 引用本地模块前先读源码的要求在段内
    assert "前先 read_file 该模块源码" in text
    # 期望: finally 内清理调用须包 try/except 的要求在段内
    assert "finally 里的 close" in text
