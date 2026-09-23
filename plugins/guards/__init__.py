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

分层依据与裁定规则见 docs/guard-system-architecture.md。

Contract:
  Preconditions: plugin system 提供 pre_tool_call / post_tool_call 钩子。
  Postconditions: 每个子模块的 register 被调用一次;任一子模块 import
    失败仅记录日志,不拖垮其余子模块(独立降级)。
  Invariants: 子模块代码为纯移动,检测逻辑与迁移前逐字一致。
"""

from __future__ import annotations

import logging

from plugins.guards import (
    coding_standards,
    duplicate_check,
    four_axis,
    pre_write,
    source_write,
    tool_blacklist,
)

logger = logging.getLogger(__name__)

_SUB_GUARDS = (
    ("tool_blacklist", tool_blacklist),
    ("four_axis", four_axis),
    ("pre_write", pre_write),
    ("duplicate_check", duplicate_check),
    ("source_write", source_write),
    ("coding_standards", coding_standards),
)


def register(ctx):
    """套件入口——逐子护栏注册,单个失败不连坐。"""
    for name, mod in _SUB_GUARDS:
        try:
            mod.register(ctx)
        except Exception:
            logger.warning("guards: sub-guard %s failed to register", name, exc_info=True)


def preflight(paths):
    """写前预检便捷入口(等价 preflight.py CLI)。"""
    from plugins.guards.preflight import main

    return main(["preflight", *paths])
