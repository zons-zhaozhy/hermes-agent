"""First-turn routing from engineering intent to bundled workflow skills.

The bundled software-development skills are strong, but a model can see their names in the
system prompt and still fail to load the relevant one. This non-blocking hook injects one
short routing map on the first normal-session LLM call. It deliberately classifies nothing:
every route is named, and the model loads a skill only when its task matches.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
import re
from typing import Any

logger = logging.getLogger(__name__)

_EXCLUDED_PLATFORMS = frozenset({"cron", "subagent", "batch"})
_MAX_TRACKED_SESSIONS = 512
WORKFLOW_ROUTES: tuple[tuple[str, str], ...] = (
    ("新功能/重构/接口迁移", "brainstorming"),
    ("工程调查/结构/调用链", "repository-investigation"),
    ("故障/报错/性能异常", "systematic-debugging"),
    ("行为变更/缺陷修复", "test-driven-development"),
    ("等待/轮询/异步就绪", "condition-based-waiting"),
    ("隔离改造/脏工作区", "using-git-worktrees"),
    ("已批准的多步架构方案", "writing-plans"),
    ("按已写计划执行", "executing-plans"),
    ("收到评审反馈", "receiving-code-review"),
    ("实现后/提交前评审", "requesting-code-review"),
    ("完成/集成/清理分支", "finishing-a-development-branch"),
    ("代码规模/语言构成", "codebase-inspection"),
    ("宣称完成/修复/通过前", "verification-before-completion"),
)
WORKFLOW_SKILLS = frozenset(skill for _situation, skill in WORKFLOW_ROUTES)
_ROUTING_HINT = (
    "[工程工作流路由·非阻断] 先完整读取对应 skill，再动手："
    + "".join(
        f"{situation}→{skill}；"
        for situation, skill in WORKFLOW_ROUTES
    )
    + "用 skill_view 读取；被拦截时先读拦截信息并补证据，禁止换工具重复尝试。"
)
_ENGINEERING_HINT_RE = re.compile(
    r"代码|文件|报错|错误|bug|测试|接口|函数|数据库|表结构|字段|配置|部署|构建|编译|"
    r"服务|页面|样式|日志|性能|分支|提交|仓库|git|api|http|python|typescript|"
    r"javascript|java|vue|react|pytest|ci|pr",
    re.IGNORECASE,
)
_ENGINEERING_ACTION_RE = re.compile(
    r"继续|接着|重构|实现|开发|迁移|修复|编写|创建|删除|清理|部署|提交"
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
    message = str(kwargs.get("user_message") or "")
    history = kwargs.get("conversation_history") or []
    history_text = " ".join(
        str(item.get("content", ""))
        for item in history
        if isinstance(item, dict)
    )
    combined = message + "\n" + history_text
    if not (_ENGINEERING_HINT_RE.search(combined) or _ENGINEERING_ACTION_RE.search(message)):
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
