"""Lazy-deferral enforcement for blocked or handed-off responses."""

from __future__ import annotations

import logging

logger = logging.getLogger("hermes.plugins.lazy_blocker")


_DEFER_PATTERNS = (
    "挂起",
    "留给 CI",
    "留给CI",
    "CI 通道可跑",
    "CI通道可跑",
    "等其他会话",
    "等并行会话",
    "等它提交",
    "下次再补",
    "后续补跑",
    "该会话提交后",
)

_ALTERNATIVE_EVIDENCE = (
    "当轮已尝试",
    "已用",
    "实测绕开",
    "替代通道",
    "复跑命令",
    "预计",
)

_NAMESPACE = "lazy_blocker"
_COOLDOWN_TURNS = 2
_MAX_INJECTIONS = 5


def _last_assistant_text(history: list[dict[str, object]] | None) -> str:
    """Return the latest assistant text, or an empty string."""
    if not history:
        return ""
    for msg in reversed(history):
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict)
            )
    return ""


def _is_lazy_deferral(text: str) -> bool:
    """Return True for a deferral claim without attempted-alternative evidence."""
    if not text:
        return False
    hit = any(pattern in text for pattern in _DEFER_PATTERNS)
    if not hit:
        return False
    return not any(evidence in text for evidence in _ALTERNATIVE_EVIDENCE)


def _on_pre_llm_call(**kwargs: object) -> dict[str, str] | None:
    """Inject a three-alternative SOP when the prior response merely stalls."""
    try:
        from plugins._shared_state import get_session_state

        sid = kwargs.get("session_id", "") or kwargs.get("task_id", "") or "__default__"
        state = get_session_state(str(sid), _NAMESPACE)

        history = kwargs.get("conversation_history") or []
        text = _last_assistant_text(history if isinstance(history, list) else None)
        if not _is_lazy_deferral(text):
            return None

        injected = state.get("injected_count", 0)
        last_turn = state.get("last_inject_turn", -10**9)
        turn = kwargs.get("turn_id", 0)
        if isinstance(turn, (int, float)) and (turn - last_turn) < _COOLDOWN_TURNS:
            return None
        if injected >= _MAX_INJECTIONS:
            return None

        state["injected_count"] = injected + 1
        if isinstance(turn, (int, float)):
            state["last_inject_turn"] = turn

        return {
            "context": (
                "【懒病拦截 lazy-blocker】你上一条回复以「挂起/留给 CI/等并行会话」"
                "类声明收尾，且没有给出当轮已尝试的替代通道证据。禁裸挂起。"
                "立即三选一并执行（不得再以声明收尾）：\n"
                "1) 绕开阻塞源做本地最小验证（如隔离编译单文件/独立进程跑目标测试）；\n"
                "2) 给出真正依赖方可执行的验证命令并写明验收判据；\n"
                "3) 看板登记复跑条件（谁解除阻塞、解除后跑什么命令），并继续推进"
                "本任务中不依赖该阻塞的其他事项。\n"
                "选 3 时必须接着做 1 或 2 至少一项，不许只登记就停。"
            )
        }
    except Exception:
        logger.exception("lazy-blocker hook failed")
        return None


def register(ctx) -> None:
    """Register the lazy-blocker pre-LLM hook."""
    ctx.register_hook("pre_llm_call", _on_pre_llm_call)
