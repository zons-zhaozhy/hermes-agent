"""persistence_enforcer 四层拦截直接行为契约。

2026-11-10 由独立插件 plugins/persistence-enforcer 迁入 discipline 套件；
状态键同时改为 (profile, session)，这里通过 _session_key 取同一把钥匙。
"""

from __future__ import annotations

import importlib
import logging
import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

persistence = importlib.import_module("plugins.discipline.persistence_enforcer")


class _Context:
    def __init__(self, settings: dict | None = None) -> None:
        self.settings = settings or {}
        self.hooks: dict[str, list] = {}

    def get_config(self, key: str):
        return self.settings.get(key)

    def register_hook(self, name: str, callback) -> None:
        self.hooks.setdefault(name, []).append(callback)


@pytest.fixture(autouse=True)
def clean_plugin_state():
    persistence._states.clear()
    persistence.WARN_THRESHOLD = 5
    persistence.BLOCK_THRESHOLD = 10
    yield
    persistence._states.clear()
    persistence.WARN_THRESHOLD = 5
    persistence.BLOCK_THRESHOLD = 10


def _session_id() -> str:
    return f"persist-{uuid.uuid4()}"


def _record_calls(session_id: str, count: int, tool_name: str = "read_file") -> None:
    for _ in range(count):
        persistence._on_post_tool_call(
            session_id=session_id, tool_name=tool_name, status="ok"
        )


def test_invalid_thresholds_fall_back_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(persistence, "WARN_THRESHOLD", 1)
        monkeypatch.setattr(persistence, "BLOCK_THRESHOLD", 1)
        with caplog.at_level(logging.WARNING, logger="plugins.discipline.persistence_enforcer"):
            persistence.register(
                _Context({"warn_threshold": "bad", "block_threshold": 0})
            )

    assert persistence.WARN_THRESHOLD == 5
    assert persistence.BLOCK_THRESHOLD == 10
    assert "warn_threshold='bad'" in caplog.text
    assert "block_threshold=0" in caplog.text


def test_warn_and_block_thresholds_are_configurable() -> None:
    persistence.register(_Context({"warn_threshold": 2, "block_threshold": 3}))
    session_id = _session_id()
    history = [{"role": "user", "content": "全面审计这个模块"}]

    _record_calls(session_id, 2)
    warning = persistence._on_pre_llm_call(
        session_id=session_id, conversation_history=history
    )
    assert "2 次工具调用" in warning["context"]
    assert "todo_list" in warning["context"]

    _record_calls(session_id, 1)
    blocked = persistence._on_pre_tool_call(
        session_id=session_id, tool_name="write_file", args={"path": "out.txt"}
    )
    assert blocked["action"] == "block"
    assert "write_file 已被拦截" in blocked["message"]


def test_todo_call_unlocks_write_tools() -> None:
    session_id = _session_id()
    _record_calls(session_id, 10)
    assert persistence._on_pre_tool_call(
        session_id=session_id, tool_name="write_file", args={}
    )["action"] == "block"

    persistence._on_post_tool_call(
        session_id=session_id, tool_name="todo_list", status="ok"
    )
    assert persistence._on_pre_tool_call(
        session_id=session_id, tool_name="write_file", args={}
    ) == {}


def test_sessions_are_isolated() -> None:
    first = _session_id()
    second = _session_id()
    _record_calls(first, 10)

    assert persistence._on_pre_tool_call(
        session_id=first, tool_name="write_file", args={}
    )["action"] == "block"
    assert persistence._on_pre_tool_call(
        session_id=second, tool_name="write_file", args={}
    ) == {}


def test_blocked_write_is_not_persistence_evidence() -> None:
    session_id = _session_id()
    _record_calls(session_id, 9)
    persistence._on_post_tool_call(
        session_id=session_id, tool_name="write_file", status="blocked"
    )

    blocked = persistence._on_pre_tool_call(
        session_id=session_id, tool_name="write_file", args={}
    )
    assert blocked["action"] == "block"
    assert persistence._states[persistence._session_key({"session_id": session_id})].persist_called is False


def test_deadlock_breaker_passes_third_write_and_warns_next_llm_call(
    caplog: pytest.LogCaptureFixture,
) -> None:
    session_id = _session_id()
    _record_calls(session_id, 10)
    args = {"path": "result.md"}

    assert persistence._on_pre_tool_call(
        session_id=session_id, tool_name="write_file", args=args
    )["action"] == "block"
    assert persistence._on_pre_tool_call(
        session_id=session_id, tool_name="write_file", args=args
    )["action"] == "block"

    with caplog.at_level(logging.WARNING, logger="plugins.discipline.persistence_enforcer"):
        assert persistence._on_pre_tool_call(
            session_id=session_id, tool_name="write_file", args=args
        ) == {}
    assert "DEADLOCK BREAKER" in caplog.text

    warning = persistence._on_pre_llm_call(
        session_id=session_id,
        conversation_history=[{"role": "user", "content": "继续"}],
    )
    assert "死锁解除" in warning["context"]
    assert persistence._on_pre_llm_call(
        session_id=session_id,
        conversation_history=[{"role": "user", "content": "继续"}],
    ) == {}


def test_unpersisted_structured_analysis_gets_diagnostic(
    caplog: pytest.LogCaptureFixture,
) -> None:
    session_id = _session_id()
    _record_calls(session_id, 5)
    response = (
        "# 审计结论\n[实测] 5 次调用。\n\n## 证据\n[文档] schema 已确认。\n"
        "[推断] 阈值行为稳定。\n" + "详细说明。\n" * 60
    )

    with caplog.at_level(logging.INFO, logger="plugins.discipline.persistence_enforcer"):
        result = persistence._on_transform_llm_output(
            session_id=session_id, response_text=response
        )
    assert result is None
    assert persistence._states[persistence._session_key({"session_id": session_id})].output_reminded is True
    assert "unpersisted analysis diagnostic" in caplog.text
    assert "`skill_manage` 或 `write_file`" in caplog.text
