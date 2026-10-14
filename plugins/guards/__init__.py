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

from __future__ import annotations

import logging
from typing import Any, Mapping

from plugins.guards import (
    batch_write,
    casebook_gate,
    coding_standards,
    diff_debt,
    duplicate_check,
    four_axis,
    groundbreaking_gate,
    pre_write,
    scope_gate,
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
    ("diff_debt", diff_debt),
    ("casebook_gate", casebook_gate),
    ("batch_write", batch_write),
    ("scope_gate", scope_gate),
    ("groundbreaking_gate", groundbreaking_gate),
)

# 事前注入段:六个子护栏此前只有拦截、无前置提示——实测同一轮里反复被拦
# (每拦一次=一轮 API 白跑)。段文本与子护栏实现逐条同源:判据、阈值、豁免、
# 后果一一对应,检错逻辑零改动。段体为稳定前缀,缓存安全。
_PREFLIGHT_SECTION = """[guards 写前功课——六个子护栏的事前要求,动手前一次做齐免返工]
1. 先读后写(pre_write):write_file/patch/execute_code 写某文件前,同一轮须已 read_file 或 search_files 读过它;terminal 读不计入。
2. 四轴证据(four_axis,主闸门 ReadThinkGate):写代码前在回复正文逐项写出 影响面(受影响调用方 路径+函数名+行号)/原意图(最初 commit+不变量)/根因(症状位置 vs 根因位置,文件:行号)/风险(最坏场景+触发条件+影响范围+可恢复性);时效 10 分钟,隔轮即失效须重贴;terminal 豁免。
3. 新建查重(duplicate_check):新建 .py 前 search_files 全库查同名/同功能实现;commit message 标「已查重,无等价实现」。
4. 批量写红线(batch_write):一次 execute_code 改「已有文件」达 2 个即拦,逐个 patch;单 commit ≤8 个代码文件,超限走 manual-review 通道(先写 /tmp/.hermes_commit_msg_cache 再 commit -F)。
5. 病例库(casebook_gate):OntoX 家族写操作前先查 education/casebook/INDEX.md 防同类复发;确认与既有病例无关则重试一次即放行。
6. 新目录树(scope_gate):仓库内新开目录树须当轮用户原话授权;禁换通道重试。
7. 源码直写(source_write):terminal 里 sed -i / cat > / python -c 直写源码被拦,改用 patch 或 write_file。"""

# 子代理与批处理会话无写代码主循环,段体诱导空转——按 platform 跳过(同
# scientific-programming-guard 口径)。
_EXCLUDED_PLATFORMS = frozenset({"subagent", "batch"})


def _render_preflight_section(session_info: Mapping[str, Any]) -> str:
    """渲染写前功课段；排除平台返回空(空段被 core 跳过)。

    Contract:
      Preconditions: session_info 为 core 冻结的只读映射,含 platform 键。
      Postconditions: 常规会话返回 _PREFLIGHT_SECTION；_EXCLUDED_PLATFORMS 内返回 ""。
    """
    if str(session_info.get("platform") or "") in _EXCLUDED_PLATFORMS:
        return ""
    return _PREFLIGHT_SECTION


def register(ctx: Any) -> None:
    """套件入口——逐子护栏注册 + 注册写前功课段(单个失败不连坐)。"""
    for name, mod in _SUB_GUARDS:
        try:
            mod.register(ctx)
        except Exception:
            logger.warning("guards: sub-guard %s failed to register", name, exc_info=True)
    try:
        ctx.register_system_prompt_section("guards_preflight", _render_preflight_section)
    except Exception:
        logger.warning("guards: preflight prompt section failed to register", exc_info=True)


def preflight(paths):
    """写前预检便捷入口(等价 preflight.py CLI)。"""
    from plugins.guards.preflight import main

    return main(["preflight", *paths])
