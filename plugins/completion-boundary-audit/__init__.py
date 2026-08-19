"""completion-boundary-audit plugin — 反面检查检查点（反者道之动）

transform_llm_output 钩子：
用 LLM judge 语义判定最终回复是否"包含交付完成声明且未披露任何未验证
边界"（人话语义不可穷举，禁关键词/正则匹配），命中则在回复末尾追加自
检红牌，强制暴露本结论未验证的边界。

设计依据：落实笔记-七心法七功法.md 第五节——把用户"完成了吗"式人工
纠偏变成 agent 自检。与 UX 审计铁律（"完成了吗"=追加审查信号）同构。

成本控制：回复 hash 去重 + judge 调用每会话硬上限 30 + 红牌每会话
上限 3 + 短回复（<80字符，闲聊）直接跳过 + fail-open。

缓存安全：只在最终回复文本尾部追加，不改 system prompt、不改历史消息、
不换 toolset —— per-conversation prompt caching 不受影响。

ACTIVATION: 需在 config.yaml plugins.enabled 中添加 "completion-boundary-audit"。
Set COMPLETION_BOUNDARY_AUDIT_DISABLE=1 to turn off.
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Any, Dict, Optional

from plugins._llm_judge import llm_judge_bool
from plugins._shared_state import get_session_state

logger = logging.getLogger(__name__)

_NAMESPACE = "completion_boundary_audit"

# 短回复（闲聊/确认）不触发
_MIN_LENGTH = 80
_MAX_REMINDERS = 3
_MAX_JUDGE_CALLS = 30

_JUDGE_SYSTEM = (
    "你是交付审查哨兵。判断下面这条 AI 最终回复是否同时满足："
    "1) 包含交付完成声明（声称任务/修复/测试/部署已完成、全部通过、已交付）；"
    "2) 未披露任何未验证边界（未列出未测路径/环境/已知风险/局限/未覆盖）。"
    "两条都满足才答 true。只是进度汇报、已含边界声明、闲聊、提问都不算。"
    "只回答 JSON：{\"needs_audit\": true} 或 {\"needs_audit\": false}"
)

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


def _state(sid: str) -> Dict[str, Any]:
    return get_session_state(sid or "_global", _NAMESPACE)


def needs_boundary_audit(text: str) -> Optional[bool]:
    """LLM judge 判"完成声明且无边界披露"；None=fail-open。

    Contract:
      Preconditions: text 为 str（可为空）
      Postconditions: 空/短文本 → False；否则 True/False/None，绝不 raise
    """
    if not text or len(text) < _MIN_LENGTH:
        return False
    return llm_judge_bool(
        task="completion_boundary_audit",
        system=_JUDGE_SYSTEM,
        text=text,
        true_key="needs_audit",
        timeout=8.0,
    )


def register(ctx) -> None:
    """注册 transform_llm_output 钩子（与 reply-certainty-checker 同范式）。"""

    def audit_boundary(response_text, session_id=None, model=None,
                       platform=None, **kwargs) -> Optional[str]:
        """judge 判"完成声明+无边界"→ 追加红牌；其余透传 None。

        Contract:
          Postconditions: 返回 None 或 text+_REMINDER（前缀=原文本）；
                          任何内部异常不破坏原回复（返回 None）
        """
        if _plugin_disabled():
            return None
        try:
            text = response_text or ""
            if len(text) < _MIN_LENGTH:
                return None
            st = _state(session_id or "")
            if int(st.get("count", 0)) >= _MAX_REMINDERS:
                return None
            if int(st.get("judge_calls", 0)) >= _MAX_JUDGE_CALLS:
                return None
            h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
            seen = st.get("seen")
            if not isinstance(seen, set):
                seen = set()
                st["seen"] = seen
            if h in seen:
                return None
            seen.add(h)
            st["judge_calls"] = int(st.get("judge_calls", 0)) + 1
            if needs_boundary_audit(text) is not True:
                return None
            st["count"] = int(st.get("count", 0)) + 1
            return text + _REMINDER
        except Exception as e:  # 绝不因插件自身错误破坏回复
            logger.warning("completion-boundary-audit skipped: %s", e,
                           exc_info=True)
        return None  # 透传

    ctx.register_hook("transform_llm_output", audit_boundary)
    logger.info("completion-boundary-audit registered")
