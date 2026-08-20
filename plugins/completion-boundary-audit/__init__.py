"""completion-boundary-audit plugin — 反面检查检查点（反者道之动）

judge 判定最终回复是否"包含交付完成声明且未披露任何未验证边界"（人话语义
不可穷举，禁关键词/正则匹配），命中不修改用户可见回复，只记入会话状态；
下一轮 pre_llm_call 把红牌注入给 agent（与 reply-certainty-checker 同范式），
强制 agent 在下一轮补披露未验证边界。

设计修正史：
- v1 transform_llm_output 直接把红牌追加到用户可见回复尾部 → 污染用户
  输出，"该由 agent 补的边界"被甩给用户看。改为记状态→下轮注入给 agent。

设计依据：落实笔记-七心法七功法.md 第五节——把用户"完成了吗"式人工
纠偏变成 agent 自检。与 UX 审计铁律（"完成了吗"=追加审查信号）同构。

成本控制：回复 hash 去重 + judge 调用每会话硬上限 30 + 红牌每会话
上限 3 + 短回复（<80字符，闲聊）直接跳过 + fail-open。

缓存安全：不改 system prompt、不改历史消息、不换 toolset——
per-conversation prompt caching 不受影响（注入走 request-scoped context）。

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
_JUDGE_TIMEOUT = 8.0

_JUDGE_SYSTEM = (
    "你是交付审查哨兵。判断下面这条 AI 最终回复是否同时满足："
    "1) 包含交付完成声明（声称任务/修复/测试/部署已完成、全部通过、已交付）；"
    "2) 未披露任何未验证边界（未列出未测路径/环境/已知风险/局限/未覆盖）。"
    "两条都满足才答 true。只是进度汇报、已含边界声明、闲聊、提问都不算。"
    "只回答 JSON：{\"needs_audit\": true} 或 {\"needs_audit\": false}"
)

_INJECT = (
    "【反面检查】你的上一条回复包含交付完成声明，但未披露任何未验证边界。"
    "本轮回复必须补充：1) 哪些路径/环境/方向未实测；2) 验证覆盖到哪里、"
    "之外是推断还是实测；3) 最可能的失败场景与触发条件。"
    "依据：交付验证铁律——报成功必报边界，未列边界=未审计。"
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
        timeout=_JUDGE_TIMEOUT,
    )


def register(ctx) -> None:
    """注册 transform_llm_output（检测+记状态）与 pre_llm_call（注入）。

    Contract:
      Postconditions: 两个钩子均已注册；钩子内部任何异常不外泄（fail-open）
    """

    def audit_boundary(response_text, session_id=None, model=None,
                       platform=None, **kwargs) -> Optional[str]:
        """judge 判"完成声明+无边界"→ 记状态供下轮注入；用户可见回复零改动。

        Contract:
          Postconditions: 恒返回 None（绝不修改用户回复）；异常不外泄
        """
        if _plugin_disabled():
            return None
        try:
            text = response_text or ""
            if len(text) < _MIN_LENGTH:
                return None
            sid = session_id or ""
            st = _state(sid)
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
            if needs_boundary_audit(text) is True:
                st["count"] = int(st.get("count", 0)) + 1
                st["pending_reminder"] = True
                logger.info(
                    "completion-boundary-audit: 完成声明未披露边界已标记"
                    "（下轮注入红牌）"
                )
            return None
        except Exception as e:  # 绝不因插件自身错误破坏回复
            logger.warning("completion-boundary-audit skipped: %s", e,
                           exc_info=True)
        return None  # 透传

    def inject_reminder(**kwargs) -> Optional[Dict[str, Any]]:
        """上一轮被标记 → 注入反面检查红牌给 agent。

        Contract:
          Postconditions: 返回 None 或 {"context": str}；注入一次即消费标记
        """
        if _plugin_disabled():
            return None
        try:
            sid = kwargs.get("session_id") or ""
            st = _state(sid)
            if not st.get("pending_reminder"):
                return None
            del st["pending_reminder"]
            return {"context": _INJECT}
        except Exception as exc:  # fail-open
            logger.warning("completion-boundary-audit inject failed: %s", exc)
            return None

    ctx.register_hook("transform_llm_output", audit_boundary)
    ctx.register_hook("pre_llm_call", inject_reminder)
    logger.info("completion-boundary-audit registered")
