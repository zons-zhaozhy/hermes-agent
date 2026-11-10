"""pre_write must enforce the read-think gate's intent/evidence verdict.

File reads are necessary but insufficient: a read-only user request can never
authorize a write, and an explicit modification request still needs a complete
investigation before the write hook runs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from plugins._shared_state import clear_session, get_session_state
from plugins.guards import pre_write

SID = "test-pre-write-authorization"


@pytest.fixture(autouse=True)
def _isolate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PW_READ_PROBE_CACHE", str(tmp_path / "probe-cache"))
    clear_session(SID)


def _publish_read_think_state(**values: object) -> None:
    state = get_session_state(SID, "read_think_gate")
    state.clear()
    state.update(values)


def _read_target(target: Path) -> None:
    pre_write.on_post_tool_call(
        session_id=SID,
        tool_name="read_file",
        args={"path": str(target)},
    )


def test_pre_write_blocks_read_only_intent_even_when_target_was_read(tmp_path: Path) -> None:
    target = tmp_path / "app.py"
    target.write_text("x = 1\n", encoding="utf-8")
    _read_target(target)
    _publish_read_think_state(
        active=True,
        intent="read_only",
        modification_authorized=False,
        investigation_complete=True,
    )

    result = pre_write.on_pre_tool_call(
        session_id=SID,
        tool_name="patch",
        args={"path": str(target), "old": "x = 1", "new": "x = 2"},
    )

    assert result is not None and result["action"] == "block"
    assert "用户意图" in result["message"]


def test_pre_write_blocks_incomplete_investigation_for_authorized_modification(
    tmp_path: Path,
) -> None:
    target = tmp_path / "app.py"
    target.write_text("x = 1\n", encoding="utf-8")
    _read_target(target)
    _publish_read_think_state(
        active=True,
        intent="modify",
        modification_authorized=True,
        investigation_complete=False,
    )

    result = pre_write.on_pre_tool_call(
        session_id=SID,
        tool_name="patch",
        args={"path": str(target), "old": "x = 1", "new": "x = 2"},
    )

    assert result is not None and result["action"] == "block"
    assert "调查" in result["message"]


def test_ambiguous_intent_with_complete_investigation_is_not_an_authorization_wait(
    tmp_path: Path,
) -> None:
    """The gate's own contract: ambiguous means "investigate", never "ask the user".

    ReadThinkGate already lets an ambiguous turn write once the investigation
    contract is satisfied. This second line used to refuse it with
    「用户修改意图不明确；先澄清」, which deadlocked goal runs: the classify
    LLM was unavailable, the deterministic fallback said ambiguous, and the
    only way out was for the human to retype an authorization.
    """
    target = tmp_path / "app.py"
    target.write_text("x = 1\n", encoding="utf-8")
    _read_target(target)
    _publish_read_think_state(
        active=True,
        intent="ambiguous",
        modification_authorized=False,
        investigation_complete=True,
    )

    result = pre_write.on_pre_tool_call(
        session_id=SID,
        tool_name="patch",
        args={"path": str(target), "old": "x = 1", "new": "x = 2"},
    )

    assert result is None


def test_ambiguous_intent_without_investigation_blocks_on_investigation(
    tmp_path: Path,
) -> None:
    target = tmp_path / "app.py"
    target.write_text("x = 1\n", encoding="utf-8")
    _read_target(target)
    _publish_read_think_state(
        active=True,
        intent="ambiguous",
        modification_authorized=False,
        investigation_complete=False,
    )

    result = pre_write.on_pre_tool_call(
        session_id=SID,
        tool_name="patch",
        args={"path": str(target), "old": "x = 1", "new": "x = 2"},
    )

    assert result is not None and result["action"] == "block"
    assert "调查" in result["message"]
    assert "先澄清" not in result["message"]
