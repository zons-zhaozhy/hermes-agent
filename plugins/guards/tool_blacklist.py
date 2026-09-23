"""guards.tool_blacklist — 必败工具机械黑名单（打地鼠根治的机制层）。

背景（0905 用户拍板）：errors.log 实测 vision_analyze 17/17 全败、
codegraph_search 5/6 败，但每个会话仍被反复调用——因为「别调它」
只写在 memory 里，属于口头纪律；违规发生在动态决策瞬间，静态笔记
拦不住。本插件把黑名单下沉到 pre_tool_call 钩子：调用直接被拦截，
错误信息自带正解，一次暴露完整替代路径。

规则：
  B1 vision_analyze 全局拦截 → 指到 mcp__glm_vision__analyze_image
     等视觉分析工具。
  B2 codegraph 系工具（mcp__codegraph__*）在非 hermes 本体仓的
     工作目录下拦截 → 指到 gitnexus（mcp__gitnexus__query/context）。
     hermes 本体仓（路径含 hermes-agent）内放行。
  B3 memory 工具带 operations 数组且数组内多于 1 个元素时拦截 →
     提示逐条单 op（实测批量多行 replace 983/1235 失败）。

Contract:
  Preconditions: plugin system provides pre_tool_call hook with
                 tool_name / args / session_id.
  Postconditions:
    - vision_analyze always blocked with replacement guidance;
    - codegraph_* blocked unless cwd is inside the hermes-agent repo;
    - memory batch operations (len(operations)>1) blocked with
      single-op guidance; single-op calls pass through untouched;
    - all other tools pass through untouched.
  Invariants: plugin never raises out of hooks; never blocks
    anything outside the three rules above.

已查重（search_files plugins/ + 全库）：无现有等价实现。
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_CODEGRAPH_PREFIX = "mcp__codegraph__"
_HERMES_REPO_MARKER = "hermes-agent"

_BLOCK_VISION = (
    "[TOOL-BLACKLIST BLOCK] vision_analyze 在本环境 100% 失败（17/17 实测），"
    "禁止调用。\n正解：图片/截图分析一律用 mcp__glm_vision__analyze_image "
    "（或 ui_to_artifact / understand_technical_diagram 等按场景选）。"
)

_BLOCK_CODEGRAPH = (
    "[TOOL-BLACKLIST BLOCK] codegraph 系工具在本环境失败率 83%（5/6 实测），"
    "仅限 hermes 本体仓使用。\n正解：代码检索/影响评估用 "
    "mcp__gitnexus__query / mcp__gitnexus__context。"
)

_BLOCK_MEMORY_BATCH = (
    "[TOOL-BLACKLIST BLOCK] memory 批量 operations（多行 replace 必败："
    "实测 983/1235 失败）被拦截。\n正解：按 current_entries 逐条提交，"
    "每个操作单独一次调用（action 单独指定，不用 operations 数组）。"
)


def _in_hermes_repo() -> bool:
    cwd = os.getcwd()
    return _HERMES_REPO_MARKER in cwd


def _on_pre_tool_call(**kwargs):
    tool_name = kwargs.get("tool_name", "") or ""
    args = kwargs.get("args", {}) or {}

    # B1: vision_analyze 全局禁
    if tool_name == "vision_analyze":
        return {"action": "block", "message": _BLOCK_VISION}

    # B2: codegraph 仅 hermes 本体仓放行
    if tool_name.startswith(_CODEGRAPH_PREFIX) and not _in_hermes_repo():
        return {"action": "block", "message": _BLOCK_CODEGRAPH}

    # B3: memory 批量 operations 拦截（仅当确实走的是批量形态）
    if tool_name == "memory" and isinstance(args, dict):
        ops = args.get("operations")
        if isinstance(ops, list) and len(ops) > 1:
            return {"action": "block", "message": _BLOCK_MEMORY_BATCH}

    return {}


def register(ctx):
    ctx.register_hook("pre_tool_call", _on_pre_tool_call)
    logger.info("tool-blacklist 插件已注册——必败工具机械黑名单就绪")
