"""completion-boundary-audit plugin — 反面检查检查点（反者道之动）

transform_llm_output 钩子：
最终回复若包含"完成/已修复/搞定/all tests pass"类交付声明，却没有任何
"未验证/边界/未覆盖/已知风险"类边界声明，则在回复末尾追加自检红牌，
强制暴露本结论未验证的边界。

设计依据：落实笔记-七心法七功法.md 第五节——把用户"完成了吗"式人工纠偏
变成 agent 自检。与 UX 审计铁律（"完成了吗"=追加审查信号）同构。

缓存安全：只在最终回复文本尾部追加，不改 system prompt、不改历史消息、
不换 toolset —— per-conversation prompt caching 不受影响。

ACTIVATION: 需在 config.yaml plugins.enabled 中添加 "completion-boundary-audit"。
Set COMPLETION_BOUNDARY_AUDIT_DISABLE=1 to turn off.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Optional

logger = logging.getLogger(__name__)

# 交付完成信号词（finite set，与 negative-conclusion-guard 的命令白名单同思路）
_COMPLETION_SIGNALS = re.compile(
    r"(已经?(全部)?完成|已完成|全部?(通过|落盘|修复|部署|搞定|交付)"
    r"|完成了吗|大功告成|all\s+tests?\s+pass(ed)?|verified\s+and\s+working"
    r"|\bdone\b.*(tested|verified)?|ship(ped|ping)\s+it)",
    re.IGNORECASE,
)

# 边界声明信号词——回复中出现任一即视为已自曝边界，透传
_BOUNDARY_DISCLOSED = re.compile(
    r"(未验证|未覆盖|边界|未测|缺口|未查证|风险|待确认|局限"
    r"|not\s+(verified|tested|covered)|untested|known\s+(risk|limitation)"
    r"|caveat|edge\s+case)",
    re.IGNORECASE,
)

# 短回复（闲聊/确认）不触发
_MIN_LENGTH = 80

_REMINDER = (
    "\n\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    "⚠️ 反面检查（completion-boundary-audit）：本次回复包含交付完成声明，"
    "但未声明任何未验证边界。\n"
    "  强制补充：本结论未验证的边界是什么？（至少列出：\n"
    "    • 哪些路径/环境/方向未实测\n"
    "    • 验证只覆盖到哪里，之外是推断还是实测\n"
    "    • 最可能的失败场景与触发条件）\n"
    "  观复两问（功法五）：\n"
    "    • 此事的反面何时来？（成功之后紧接着的衰减/回归/反噬）\n"
    "    • 此事处于循环哪段？（起始上升/平台衰减/末期清算）\n"
    "  依据：交付验证铁律——报成功必报边界，未列边界=未审计。\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
)


def _plugin_disabled() -> bool:
    return os.environ.get("COMPLETION_BOUNDARY_AUDIT_DISABLE", "").lower() in {
        "1", "true", "yes", "on",
    }


def needs_boundary_audit(text: str) -> bool:
    """True 当且仅当：含交付完成信号、未披露边界、且是长回复（非闲聊）。

    Contract:
      Preconditions: text 为 str（可为空）
      Postconditions: 返回 bool；text 为空/短/无完成信号/已披露边界 → False
    """
    if not text or len(text) < _MIN_LENGTH:
        return False
    if _BOUNDARY_DISCLOSED.search(text):
        return False
    return bool(_COMPLETION_SIGNALS.search(text))


def register(ctx) -> None:
    """注册 transform_llm_output 钩子（与 reply-certainty-checker 同范式）。"""

    def audit_boundary(response_text, session_id=None, model=None, platform=None, **kwargs):
        if _plugin_disabled():
            return None
        try:
            if needs_boundary_audit(response_text or ""):
                return response_text + _REMINDER
        except Exception as e:  # 绝不因插件自身错误破坏回复
            logger.warning("completion-boundary-audit skipped: %s", e, exc_info=True)
        return None  # 透传

    ctx.register_hook("transform_llm_output", audit_boundary)
    logger.info("completion-boundary-audit registered")
