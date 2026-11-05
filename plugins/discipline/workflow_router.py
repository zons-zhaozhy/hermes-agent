"""First-turn routing from engineering intent to bundled workflow skills.

The bundled software-development skills are strong, but a model can see their names in the
system prompt and still fail to load the relevant one. This non-blocking hook injects one
short routing map on the first normal-session LLM call. It deliberately classifies nothing:
every route is named, and the model loads a skill only when its task matches.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any

logger = logging.getLogger(__name__)

_EXCLUDED_PLATFORMS = frozenset({"cron", "subagent", "batch"})
_MAX_TRACKED_SESSIONS = 512
_ROUTING_HINT = (
    "[工程工作流路由·非阻断] 若本会话进入工程工作，先完整读取对应 skill，再动手："
    "新功能/重构/接口或数据迁移→skill_view(name='brainstorming')；"
    "故障/报错/性能异常→skill_view(name='systematic-debugging')；"
    "行为变更或缺陷修复→skill_view(name='test-driven-development')；"
    "已批准的多步架构方案→skill_view(name='writing-plans')；"
    "宣称完成/修复/通过前→skill_view(name='verification-before-completion')。"
    "被护栏拦截时，把拦截原因当作证据进入 systematic-debugging，禁止换工具重复尝试。"
)
_injected_sessions: OrderedDict[str, None] = OrderedDict()


def _forget(session_id: str) -> None:
    """Drop one session's routing state (also the test/reset seam)."""
    _injected_sessions.pop(session_id, None)


def on_pre_llm_call(**kwargs: Any) -> dict[str, str]:
    """Return the routing hint once per ordinary session; never classify or block."""
    session_id = str(kwargs.get("session_id") or kwargs.get("task_id") or "")
    platform = str(kwargs.get("platform") or "")
    if not session_id or platform in _EXCLUDED_PLATFORMS or session_id in _injected_sessions:
        return {}
    _injected_sessions[session_id] = None
    _injected_sessions.move_to_end(session_id)
    while len(_injected_sessions) > _MAX_TRACKED_SESSIONS:
        _injected_sessions.popitem(last=False)
    return {"context": _ROUTING_HINT}


def on_session_reset(session_id: str = "", **_kwargs: Any) -> None:
    """A reset starts a fresh working context, so the route may be needed again."""
    _forget(session_id)


def on_session_end(session_id: str = "", **_kwargs: Any) -> None:
    """Release per-session state when the host fires its lifecycle hook."""
    _forget(session_id)


def register(ctx: Any) -> None:
    """Register routing and lifecycle cleanup hooks."""
    ctx.register_hook("pre_llm_call", on_pre_llm_call)
    ctx.register_hook("on_session_reset", on_session_reset)
    ctx.register_hook("on_session_end", on_session_end)
    logger.info("workflow-router 已注册——首轮工程工作流路由就绪")
