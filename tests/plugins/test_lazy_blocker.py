"""lazy-blocker 直接行为契约。

期望独立于实现推导：裸挂起必须注入三选一替代通道；已有替代证据不拦截；
同 session 两轮冷却，跨 session 不共享。
"""

from __future__ import annotations

import sys
import uuid
import importlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from plugins._shared_state import clear_session

lazy_blocker = importlib.import_module("plugins.lazy-blocker")


def _history(text: str) -> list[dict]:
    return [{"role": "user", "content": "继续"}, {"role": "assistant", "content": text}]


def test_bare_deferral_is_lazy_but_evidence_is_not() -> None:
    assert lazy_blocker._is_lazy_deferral("问题挂起") is True
    assert lazy_blocker._is_lazy_deferral("挂起；替代通道已用本地最小验证验证") is False
    assert lazy_blocker._is_lazy_deferral("") is False


def test_lazy_assistant_response_injects_three_alternatives() -> None:
    result = lazy_blocker._on_pre_llm_call(
        session_id=f"lazy-{uuid.uuid4()}",
        turn_id=1,
        conversation_history=_history("问题挂起"),
    )

    assert result is not None
    context = result["context"]
    assert "1) 绕开阻塞源做本地最小验证" in context
    assert "2) 给出真正依赖方可执行的验证命令" in context
    assert "3) 看板登记复跑条件" in context
    assert "不许只登记就停" in context


def test_same_session_has_two_turn_cooldown() -> None:
    session_id = f"lazy-cooldown-{uuid.uuid4()}"
    try:
        first = lazy_blocker._on_pre_llm_call(
            session_id=session_id, turn_id=1, conversation_history=_history("问题挂起")
        )
        second = lazy_blocker._on_pre_llm_call(
            session_id=session_id, turn_id=2, conversation_history=_history("问题挂起")
        )
        third = lazy_blocker._on_pre_llm_call(
            session_id=session_id, turn_id=3, conversation_history=_history("问题挂起")
        )

        assert first is not None
        assert second is None
        assert third is not None
    finally:
        clear_session(session_id)


def test_sessions_do_not_share_injection_state() -> None:
    first_id = f"lazy-first-{uuid.uuid4()}"
    second_id = f"lazy-second-{uuid.uuid4()}"
    try:
        first = lazy_blocker._on_pre_llm_call(
            session_id=first_id, turn_id=1, conversation_history=_history("问题挂起")
        )
        second = lazy_blocker._on_pre_llm_call(
            session_id=second_id, turn_id=1, conversation_history=_history("问题挂起")
        )

        assert first is not None
        assert second is not None
    finally:
        clear_session(first_id)
        clear_session(second_id)


def test_non_deferral_and_evidence_history_do_not_inject() -> None:
    session_id = f"lazy-negative-{uuid.uuid4()}"
    try:
        assert lazy_blocker._on_pre_llm_call(
            session_id=session_id,
            turn_id=1,
            conversation_history=_history("继续实现"),
        ) is None
        assert lazy_blocker._on_pre_llm_call(
            session_id=session_id,
            turn_id=1,
            conversation_history=_history("挂起；替代通道已用本地测试实测"),
        ) is None
    finally:
        clear_session(session_id)
