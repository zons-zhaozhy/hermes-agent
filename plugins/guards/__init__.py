"""guards —— L2 写入质量护栏套件（收拢统一入口）。

前身 6 个独立插件目录(tool-blacklist / four-axis-guard / pre-write-guard /
duplicate-check / source_code_write_guard / coding-standards-guard)收拢为
单一套件:一个 plugin.yaml、config 里一个开关 ``guards``。子模块职责:

  tool_blacklist  L0 必败工具机械黑名单(vision_analyze→glm_vision、
                  codegraph 非 hermes 仓→gitnexus、memory 批量 op→逐条)
  four_axis       四轴闸门副防线(marker 与主闸门 ReadThinkGate 对齐)
  pre_write       先读后写门禁(write/patch/execute_code 前必须 read_file)
  duplicate_check 新建文件前置查重
  source_write    terminal 绕过 patch 直写源码拦截(sed -i/cat >/python -c)
  coding_standards AST 编码规范事前拦截(R001~R022) + preflight.py 写前预检
  scope_gate      新增产物授权闸门(仓库内新开目录树须当轮用户原话授权)

分层依据与裁定规则见 docs/guard-system-architecture.md。

Contract:
  Preconditions: plugin system 提供 pre_tool_call / post_tool_call 钩子。
  Postconditions: 每个子模块的 register 被调用一次;任一子模块 import
    失败仅记录日志,不拖垮其余子模块(独立降级)。
  Invariants: 子模块代码为纯移动,检测逻辑与迁移前逐字一致。
"""

import importlib
import logging
from typing import Any, Mapping

logger = logging.getLogger(__name__)

# 逐个导入: 任一子模块 import 失败仅记录日志并跳过(独立降级,见模块 Contract)——
# 单条 from-import 会让一个 ImportError 连坐全套 11 守卫(2026-10-21 审计发现)。
_LOADED: list[tuple[str, Any]] = []
for _name in ("tool_blacklist", "four_axis", "pre_write", "duplicate_check", "source_write",
              "coding_standards", "diff_debt", "casebook_gate", "batch_write", "scope_gate",
              "groundbreaking_gate", "slow_search"):
    try:
        _LOADED.append((_name, importlib.import_module(f"plugins.guards.{_name}")))
    except Exception:
        logger.warning("guards: 子模块 %s import 失败,独立降级跳过", _name, exc_info=True)

_SUB_GUARDS = tuple(_LOADED)

# 事前注入段:六个子护栏与必败黑名单此前只有拦截、无前置提示——实测同一轮里
# 反复被拦(每拦一次=一轮 API 白跑)。段文本已外置为段库 prompt_sections/sections.json
# (单一事实源, 见 plugins/prompt_sections.py):渲染器每次读盘取最新, 故段文本修改
# 对新会话立即生效, 对在跑的会话由 section_reinjection 通道按版本指纹补注。
# 段文本与子护栏实现逐条同源:判据、阈值、豁免、后果一一对应, 检错逻辑零改动。

# 子代理与批处理会话无写代码主循环,段体诱导空转——按 platform 跳过(同
# scientific-programming-guard 口径)。
_EXCLUDED_PLATFORMS = frozenset({"subagent", "batch"})


def _render_preflight_section(session_info: Mapping[str, Any]) -> str:
    """渲染写前功课段（读段库取最新；排除平台返回空段）。

    Contract:
      Preconditions: session_info 为 core 冻结的只读映射,含 platform 键。
      Postconditions: 常规会话返回段库内 guards_preflight 文本；排除平台返回 ""。
    """
    if str(session_info.get("platform") or "") in _EXCLUDED_PLATFORMS:
        return ""
    from plugins.prompt_sections import load_section  # 延迟导入: 仅段渲染路径需要

    return load_section("guards_preflight")


def _render_blacklist_section(session_info: Mapping[str, Any]) -> str:
    """渲染必败调用黑名单段（读段库取最新；排除平台返回空段）。

    Contract:
      Preconditions: session_info 含 platform 键。
      Postconditions: 常规会话返回段库内 tool_blacklist 文本；排除平台返回 ""。
    """
    if str(session_info.get("platform") or "") in _EXCLUDED_PLATFORMS:
        return ""
    from plugins.prompt_sections import load_section

    return load_section("tool_blacklist")


def register(ctx: Any) -> None:
    """套件入口——逐子护栏注册 + 注册文件型段(单个失败不连坐)。"""
    for name, mod in _SUB_GUARDS:
        try:
            mod.register(ctx)
        except Exception:
            logger.warning("guards: sub-guard %s failed to register", name, exc_info=True)
    try:
        ctx.register_system_prompt_section("guards_preflight", _render_preflight_section)
        ctx.register_system_prompt_section("tool_blacklist", _render_blacklist_section)
    except Exception:
        logger.warning("guards: prompt section failed to register", exc_info=True)


def preflight(paths):
    """写前预检便捷入口(等价 preflight.py CLI)。"""
    from plugins.guards.preflight import main

    return main(["preflight", *paths])
