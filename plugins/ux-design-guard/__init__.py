"""ux-design-guard — UX 设计纪律机检插件（八层方法论执行层）。

双层同源（对齐 scientific-programming-guard 范式：事前注入+事后拦截）：

  L1 稳定段（register_system_prompt_section）
     动工前四判据+五问走查+六原则压缩段（≤400 字）注入每个新会话——
     不依赖 skill 是否被加载。理论全文唯一事实源
     =~/knowledge-base/ux-design/UX_INTERACTION_METHODOLOGY.md，段内不复制正文。

  L2 写前拦截（pre_tool_call）
     UI 源文件（.vue/.tsx/.jsx/.css/.scss/.less/.html）写入时机检两类理论违规：
     U1 装饰性无限动效——动效唯一合法用途是澄清状态（Cooper「用过渡解释变化」：
        过渡用于解释刚才发生了什么；常驻循环表演不承载状态信息）；
        合法白名单=loading/spinner/skeleton/shimmer/typing/pulse/progress/blink/
        error/warning/danger/alive/live/indicator/status 等状态语义；
     U2 队列入场动画——animationDelay 绑定索引乘法（逐项渐入延迟徒增感知延迟，
        排队是表演不是信息）。

  设计约束：fail-open（插件异常不阻断会话）；dist/build/min 生成产物跳过；
  行内 ``ux-ok: <≥8字理由>`` 豁免；纯 str 方法逐行检测（零正则）。
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

logger = logging.getLogger(__name__)

_NAMESPACE = "ux_design_guard"

_EXCLUDED_PLATFORMS = {"subagent", "batch"}

_UI_SUFFIXES = (".vue", ".tsx", ".jsx", ".css", ".scss", ".less", ".html")

# 生成产物不检（库代码/构建输出，非手写设计面）
_ARTIFACT_MARKS = ("/dist/", "/build/", "/.output/", "/node_modules/", ".min.css", ".min.js")

# 状态语义白名单：命中即视为「澄清状态」的合法循环动画
_STATE_TOKENS = (
    "loading", "spinner", "skeleton", "shimmer", "typing",
    "pulse", "progress", "blink", "error", "warning", "danger",
    "alive", "live", "indicator", "status", "heartbeat", "scan",
)

_EXEMPT_MARK = "ux-ok:"

_UX_RULE = """\
[UX 设计纪律]
写界面文件(.vue/.tsx/.jsx/.css/.html)前对表八层方法论:①Kano投入顺序恒为
必备>期望>魅力——数据正确/操作不丢/可撤销/报错人话未清零,禁投入动效炫技
②五问走查:看(一眼懂)/动(几步/有反馈/可撤销)/错(能自恢复)/记(要记什么)/惯(一致吗)
③Cooper六原则:就地直接/留在页面/保持轻量/提供邀请/用过渡解释变化(动效只许澄清状态)/即时响应
④硬判据:反馈≤400ms;每屏主操作唯一;结果页/异常页投入最高;空态即引导;撤销优于确认
理论=~/knowledge-base/ux-design/UX_INTERACTION_METHODOLOGY.md;判据=skill:ux-methodology"""


def _is_ui_target(path: str) -> bool:
    """路径是否属于机检范围的 UI 源文件。

    Contract:
      Postconditions: 后缀命中 UI 集合且不含生成产物标记 → True；其余 False
    """
    if not path.endswith(_UI_SUFFIXES):
        return False
    norm = path.replace("\\", "/").lower()
    return not any(mark in norm for mark in _ARTIFACT_MARKS)


def _has_exempt(line: str) -> bool:
    """行内豁免标记——ux-ok: 之后须留 ≥8 字符实质理由。

    Contract:
      Postconditions: 含合法豁免标记 → True；其余 False
    """
    idx = line.find(_EXEMPT_MARK)
    return idx >= 0 and len(line[idx + len(_EXEMPT_MARK):].strip()) >= 8


def _line_state_semantic(lines: list, idx: int) -> bool:
    """命中行及其前 4 行（选择器上下文窗口）是否含状态语义。

    Contract:
      Preconditions: 0 <= idx < len(lines)
      Postconditions: 窗口内任一行含状态令牌 → True
    """
    window = lines[max(0, idx - 4): idx + 1]
    blob = "\n".join(window).lower()
    return any(tok in blob for tok in _STATE_TOKENS)


def _check_decorative_infinite(lines: list) -> list:
    """U1 装饰性无限动效检测——无限循环且无状态语义即拦截。

    Contract:
      Postconditions: 返回违规消息列表（空=合规）
    """
    issues: list = []
    for lineno, line in enumerate(lines, start=1):
        low = line.lower()
        if "infinite" not in low or "animation" not in low:
            continue
        if _has_exempt(line):
            continue
        if _line_state_semantic(lines, lineno - 1):
            continue
        issues.append(
            f"U1 装饰性无限动效: line {lineno} '{line.strip()[:60]}' —"
            "动效唯一合法用途是澄清状态(Cooper/用过渡解释变化),常驻表演动画"
            "不承载状态信息。若确为状态指示(loading/skeleton/typing 等)请把状态"
            "语义写进类名或声明行;确有叙事必要则行内注释 ux-ok: <理由> 豁免。"
        )
    return issues


def _check_queue_in(lines: list) -> list:
    """U2 队列入场动画检测——animationDelay 随索引递增即拦截。

    Contract:
      Postconditions: 返回违规消息列表（空=合规）
    """
    issues: list = []
    for lineno, line in enumerate(lines, start=1):
        low = line.lower()
        if "animationdelay" not in low and "animation-delay" not in low:
            continue
        if _has_exempt(line):
            continue
        if any(sig in line for sig in ("*0.", "* 0.", "*$", "* $", "* i", "*i")):
            issues.append(
                f"U2 队列入场动画: line {lineno} '{line.strip()[:60]}' —"
                "animationDelay 随索引递增=排队弹入,是表演不是信息(逐项渐入"
                "徒增感知延迟)。整组同刻入场或按状态分批;确有叙事必要则行内"
                "注释 ux-ok: <理由> 豁免。"
            )
    return issues


def _ux_section(session_info: Mapping) -> str:
    """渲染稳定段（subagent/batch 无界面写场景，返回空被 core 跳过）。

    Contract:
      Postconditions: 主会话返回 _UX_RULE 原文；排除平台返回 ""
    """
    if str(session_info.get("platform") or "") in _EXCLUDED_PLATFORMS:
        return ""
    return _UX_RULE


def on_pre_tool_call(**kwargs: Any) -> dict:
    """pre_tool_call 入口——UI 文件写前机检 UX 纪律。

    Contract:
      Preconditions: kwargs 含 tool_name 与 args（write_file/patch 为
        path+content / path+old_string+new_string）
      Postconditions: 违规返回 {"action":"block","message":...}；
        合规/非目标工具/非 UI 文件/解析失败 → {} 放行（fail-open）
    """
    try:
        tool_name = kwargs.get("tool_name", "")
        if tool_name not in ("write_file", "patch"):
            return {}
        args = kwargs.get("args", {}) or {}
        path = str(args.get("path", ""))
        if not _is_ui_target(path):
            return {}
        text = args.get("content") or args.get("new_string") or ""
        lines = text.splitlines()
        issues = _check_decorative_infinite(lines) + _check_queue_in(lines)
        if issues:
            return {
                "action": "block",
                "message": (
                    "[ux-design-guard] UX 设计纪律违规:\n"
                    + "\n".join(issues)
                    + "\n理论依据: ~/knowledge-base/ux-design/UX_INTERACTION_METHODOLOGY.md; "
                    "规则全文: skill:ux-methodology"
                ),
            }
        return {}
    except Exception:
        logger.warning("ux-design-guard 检查失败(放行)", exc_info=True)
        return {}


def register(ctx: Any) -> None:
    """插件入口：事前注入(system prompt 段) + 事后拦截(pre_tool_call) 双层同源执法。

    Contract:
      Postconditions: 两挂点已注册；异常不外泄
    """
    ctx.register_hook("pre_tool_call", on_pre_tool_call)
    ctx.register_system_prompt_section(_NAMESPACE, _ux_section)
    logger.info("ux-design-guard registered (hook + prompt section)")
