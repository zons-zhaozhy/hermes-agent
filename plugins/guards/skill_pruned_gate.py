"""guards.skill_pruned_gate — 技能剪枝机械拦截。

背景：上下文压缩会把历史注入的技能正文替换为 ``[SKILL_PRUNED: ...]`` 标记
（agent/context_compressor.py #32106 负责重注入标记本身）。skill_view 此时返回
占位符而非全文——占位符文本要求代理「reload before acting」，但这是提示而非
机械强制，实测存在代理带着占位符继续执行写操作的路径（2026-10-26 会话实例：
design-aesthetics-essence 首查即 PRUNED）。

本守卫把「自觉重载」升级为机械拦截：本会话内 skill_view 返回过 PRUNED 占位符
且同名技能尚未成功重载（再次 skill_view 取回全文）期间，写通道一律拦。

覆盖边界（诚实声明）：压缩事件本身不经工具调用，守卫只能捕获「skill_view 显式
返回过占位符」路径；代理从未调用 skill_view 的纯历史标记路径不在本守卫射程内，
仍由占位符文本与注入段约束。

STATE: per-session「待重载技能名集合」，namespace=skill_pruned_gate。

Contract:
  Preconditions: 钩子层提供 tool_name/args/session_id/result(kwargs)。
  Postconditions:
    - post: result 含 [SKILL_PRUNED → 技能名入待重载集合；result 为正常全文
      → 技能名移出集合（重载即解除）。
    - pre: 写通道(write_file/patch/execute_code 含写)在集合非空时 block，
      行内豁免标记 ``# skill-pruned-ok: <理由>`` 放行。
    - 守卫自身异常 fail-open（return None + logger），不拖垮工具执行。
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from plugins._shared_state import get_session_state

logger = logging.getLogger(__name__)

_NAMESPACE = "skill_pruned_gate"
_PRUNED_MARKER = "[SKILL_PRUNED"
_EXEMPT_MARKER = "# skill-pruned-ok"
_NAME_ANCHOR = "skill_view(name='"

_WRITE_TOOLS = frozenset({"write_file", "patch", "execute_code"})

# 与 pre_write._contains_write 同口径的写形态判定（独立持有，避免跨守卫耦合）
_WRITE_INDICATORS = (
    "write_file(", "patch(", ".write(", "open(",
    "pathlib.path(", ".write_text(", ".write_bytes(",
)


def _pending(sid: str) -> set[str]:
    return get_session_state(sid, _NAMESPACE).setdefault("pending_reload", set())


def _extract_skill_name(result: str, args: dict[str, Any]) -> Optional[str]:
    """从占位符文本提取技能名；args.name 优先（钩子层可靠字段）。

    Contract:
      Postconditions: 返回技能名或 None；零正则（str 切片解析占位符固定格式）。
    """
    name = str(args.get("name") or "").strip()
    if name:
        return name
    # 回退：占位符固定携带 skill_view(name='X') 字样
    try:  # noqa: D5 — 切片越界=格式不含锚点,按无名处理
        return result.split(_NAME_ANCHOR, 1)[1].split("'", 1)[0] or None
    except IndexError:
        return None


# skill-pruned-ok: repairing this gate's own false-positive deadlock — unrelated to skill criteria
# 占位符标记固定出现在结果头部状态行（"[skill_view] name='X' (N chars) [SKILL_PRUNED: ..."）。
# 头部窗口外的命中是技能正文引用标记字样（如 engineering-lessons 的压缩丢失教学案例），不算占位符。
_PRUNED_HEAD_WINDOW = 400


def on_post_tool_call(**kwargs: Any) -> None:
    """post: 记录/解除待重载技能。observer-only，永不阻断。"""
    tool_name = str(kwargs.get("tool_name") or "")
    if tool_name != "skill_view":
        return
    args = kwargs.get("args") or {}
    result = str(kwargs.get("result") or "")
    sid = str(kwargs.get("session_id") or kwargs.get("task_id") or "")
    name = _extract_skill_name(result, args)
    if not name:
        return
    # skill-pruned-ok: gate self-repair — head-window check (see _PRUNED_HEAD_WINDOW above)
    if _PRUNED_MARKER in result[:_PRUNED_HEAD_WINDOW]:
        _pending(sid).add(name)
        logger.info("skill_pruned_gate: %s 处于剪枝待重载状态", name)
    else:
        _pending(sid).discard(name)


def _code_has_write(code: str) -> bool:
    lower = code.lower()
    return any(ind in lower for ind in _WRITE_INDICATORS)


def _exempt_or_nonwrite(tool_name: str, args: dict[str, Any]) -> bool:
    """豁免标记命中，或 execute_code 不含写形态 → 不拦。

    Contract:
      Postconditions: write_file/patch 仅豁免标记可过；execute_code 另须含写形态。
    """
    body = str(args.get("content") or args.get("code") or args.get("new_string") or "")
    if _EXEMPT_MARKER in body:
        return True
    return tool_name == "execute_code" and not _code_has_write(body)


def on_pre_tool_call(**kwargs: Any) -> Optional[dict[str, Any]]:
    """pre: 待重载集合非空时拦写通道。异常 fail-open。"""
    try:
        tool_name = str(kwargs.get("tool_name") or "")
        if tool_name not in _WRITE_TOOLS:
            return None
        if _plugin_disabled():
            return None
        args = kwargs.get("args") or {}
        if _exempt_or_nonwrite(tool_name, args):
            return None
        sid = str(kwargs.get("session_id") or kwargs.get("task_id") or "")
        pending = _pending(sid)
        if not pending:
            return None
        return {
            "action": "block",
            "message": (
                f"[SkillPrunedGuard] 技能剪枝拦截：以下技能本会话查得 [SKILL_PRUNED] "
                f"占位符且尚未重载全文：{sorted(pending)}。占位符不含技能判据，"
                f"基于它执行写操作=凭残缺内容行动。\n"
                f"  修复: 先 skill_view(name='上述技能') 取回全文，再执行本写操作。"
                f"\n  豁免: 写内容行内加「# skill-pruned-ok: <与本技能无关的理由>」。"
            ),
        }
    except Exception:
        logger.warning("skill_pruned_gate: pre 检查异常, fail-open", exc_info=True)
        return None


def _plugin_disabled() -> bool:
    import os

    return os.environ.get("SKILL_PRUNED_GUARD_DISABLE", "").lower() in {
        "1", "true", "yes", "on",
    }


def register(ctx: Any) -> None:
    ctx.register_hook("pre_tool_call", on_pre_tool_call)
    ctx.register_hook("post_tool_call", on_post_tool_call)
    logger.info("guards.skill_pruned_gate registered")
