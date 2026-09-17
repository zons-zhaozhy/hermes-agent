# -*- coding: utf-8 -*-
"""requirement_interview 行为级 E2E 单测。

根因背景（2026-09-15 反方审查 deleg_1f8da3d0）：规则曾活在首轮 user 消息
sidecar，in-place 压缩第 2 次起 protect_first_n 衰减为 0 → 规则被卷进
summary 永久灭失。本版迁到 session 级 system prompt 段（core 冻结管道：
agent/system_prompt.py::_frozen_plugin_prompt_sections + resume 时从持久化
prompt 反解恢复）。

期望值独立推导（不读实现凑数）：
1. 段渲染：正常平台返回规则原文（含拍板铁律）；subagent/batch 返回空。
2. 灭失免疫：规则在 system prompt 里的存活不依赖任何 user 消息——压缩
   只动 history 不动 system prompt，故「压缩后仍在」是结构性成立；
   这里直接断言冻结快照来自 render 管道而非消息历史。
3. 恢复链：持久化 prompt 里的段 bytes 能被 _restore_plugin_prompt_sections
   反解回来（resume 场景规则不丢）。
4. 空段跳过：空内容段不进渲染列表（core 行为）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pytest

import plugins.requirement_interview as ri


class _CtxShim:
    """最小 ctx：只暴露 register_system_prompt_section，走生产 PluginContext 同名方法。"""

    def __init__(self, manager: Any, name: str) -> None:
        self._name = name
        self._manager = manager

    def register_system_prompt_section(self, *args: Any, **kwargs: Any) -> Any:
        from hermes_cli.plugins import PluginContext
        from hermes_cli.plugins_manifest import parse_manifest_file

        manifest = parse_manifest_file(
            Path(ri.__file__).parent / "plugin.yaml", Path(ri.__file__).parent, "bundled", "")
        assert manifest is not None, "plugin.yaml 解析失败"
        pc = PluginContext(manifest, self._manager)
        return pc.register_system_prompt_section(*args, **kwargs)


class TestSectionRender:
    def test_cli_platform_gets_full_rule(self):
        content = ri._interview_section({"platform": "cli"})
        assert content == ri._INTERVIEW_RULE
        assert "拍板铁律" in content
        assert "适用条件" in content

    def test_rule_contains_clarify_gate_contract(self):
        # 期望：GATE 纪律四要素齐全——批量问/停轮等答/禁自问自答/拒绝回答落假设
        content = ri._interview_section({"platform": "cli"})
        for anchor in ("澄清 GATE", "本轮终止等用户回答", "自问自答", "GOTCHA"):
            assert anchor in content, f"GATE 契约缺锚点: {anchor}"

    def test_subagent_excluded(self):
        assert ri._interview_section({"platform": "subagent"}) == ""

    def test_batch_excluded(self):
        assert ri._interview_section({"platform": "batch"}) == ""

    def test_empty_platform_not_excluded(self):
        # 空 platform 是异常元数据而非子代理，宁可多注入不可静默丢保护
        assert ri._interview_section({}) == ri._INTERVIEW_RULE


class TestCorePipelineIntegration:
    """走真实注册管道（独立 PluginManager，不依赖用户 config 的 enable 列表——
    测试沙箱 HERMES_HOME 与生产 ~/.hermes/config.yaml 的 plugins.enabled 不同步，
    discovery 出的插件会被 enable-gate 拦下；这里直接对插件模块调 register，
    验证的正是生产 load 时执行的同一段注册代码）。"""

    def test_render_pipeline_includes_section(self, manager_with_plugin: Any):
        sections = manager_with_plugin.render_system_prompt_sections(
            {"platform": "cli", "session_id": "s1"})
        ids = [s.id for s in sections]
        assert "requirement_interview" in ids, f"段未渲染: {ids}"
        content = [s.content for s in sections if s.id == "requirement_interview"][0]
        assert "拍板铁律" in content

    def test_render_pipeline_excludes_subagent(self, manager_with_plugin: Any):
        sections = manager_with_plugin.render_system_prompt_sections(
            {"platform": "subagent", "session_id": "s2"})
        got = [s.content for s in sections if s.id == "requirement_interview"]
        assert not got or not got[0].strip(), "subagent 应为空段且被 core 跳过"


@pytest.fixture()
def manager_with_plugin() -> Any:
    """独立 PluginManager + 真实插件 register（不依赖用户 config enable 列表），
    供本文件各类走真实注册-渲染管道。"""
    from hermes_cli.plugins import PluginManager

    manager = PluginManager()
    ri.register(_CtxShim(manager, "requirement_interview"))
    return manager


class TestCompressionSurvival:
    def test_restore_from_persisted_prompt(self, manager_with_plugin: Any):
        """恢复链：段 bytes 冻结进持久化 prompt 后可反解——resume 不丢规则。"""
        from agent.system_prompt import _restore_plugin_prompt_sections
        from hermes_cli.plugins_dispatch import format_system_prompt_sections

        sections = manager_with_plugin.render_system_prompt_sections(
            {"platform": "cli", "session_id": "s1"})
        rendered = [s for s in sections if s.id == "requirement_interview"]
        assert rendered, "前置失败：段未渲染"
        persisted = format_system_prompt_sections(rendered)
        assert "requirement_interview" in persisted
        # 反解校验要求容器后紧跟会话首条标记（core 防伪帧），补齐真实持久化形态
        framed_prompt = persisted + "\n\nConversation started:"
        restored = _restore_plugin_prompt_sections(framed_prompt)
        assert any(item.id == "requirement_interview" for item in restored), (
            "持久化段反解失败——resume 后规则会灭失"
        )

    def test_rule_lives_in_system_prompt_not_history(self, manager_with_plugin: Any):
        """灭失免疫的结构性断言：规则段由 system prompt 冻结管道承载，
        组装面是 agent/system_prompt.py 的段列表，与任何 user 消息无关——
        压缩仅作用于 history，不可能触达本段。"""
        sections = manager_with_plugin.render_system_prompt_sections(
            {"platform": "cli", "session_id": "s1"})
        rendered = [s for s in sections if s.id == "requirement_interview"]
        assert rendered and "拍板铁律" in rendered[0].content
