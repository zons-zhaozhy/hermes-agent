"""A session-scoped /model switch that fails to reach the session DB must be loud.

The switch already holds in-process, so a silent skip shows "switched" in the UI
while every resume path (_restore_session_model, #79536) restores the stale
creation-time model from the row — the user sees the model flip back minutes
later. Contract under test: _persist_model_switch_to_session returns False and
logs a warning on both failure shapes (store unavailable, DB write error),
returns True on success, and _commit_model_switch surfaces a user-visible line.
"""

from __future__ import annotations

import logging
from typing import Any

from hermes_cli.model_switch import ModelSwitchResult

_RESULT = ModelSwitchResult(
    success=True, new_model="glm-4.6", target_provider="zai", provider_changed=True,
    api_key="sk-x", base_url="https://example/v4", api_mode="chat_completions",
)


def _stub(db: Any = None, sid: str = "local-test") -> Any:
    return type("Stub", (), {"_session_db": db, "session_id": sid})()


def test_store_unavailable_returns_false_and_warns(caplog: Any) -> None:
    from hermes_cli.cli_model_switch_mixin import CLIModelSwitchMixin
    with caplog.at_level(logging.WARNING, logger="cli"):
        persisted = CLIModelSwitchMixin._persist_model_switch_to_session(_stub(db=None), _RESULT)
    assert persisted is False  # 期望: False——存储缺失必须让调用方能提示,静默 None 即旧缺陷
    assert "not persisted" in caplog.text  # 期望: warning 落日志(旧实现零日志)


def test_db_write_error_returns_false_and_warns(caplog: Any) -> None:
    from hermes_cli.cli_model_switch_mixin import CLIModelSwitchMixin

    class BrokenDB:
        def update_session_model(self, *a: object, **k: object) -> None:
            raise RuntimeError("database is locked")

        def patch_session_model_config(self, *a: object, **k: object) -> None:
            raise AssertionError("must not be reached")

    with caplog.at_level(logging.WARNING, logger="cli"):
        persisted = CLIModelSwitchMixin._persist_model_switch_to_session(_stub(db=BrokenDB()), _RESULT)
    assert persisted is False  # 期望: False——写库异常不得伪装成功
    assert "Failed to persist" in caplog.text  # 期望: warning 带异常(旧实现 debug 级吞掉)


def test_successful_persist_returns_true() -> None:
    from hermes_cli.cli_model_switch_mixin import CLIModelSwitchMixin
    written: dict[str, Any] = {}

    class OKDB:
        def update_session_model(self, sid: str, model: str, **k: object) -> None:
            written["sid"] = sid
            written["model"] = model

        def patch_session_model_config(self, sid: str, cfg: dict) -> None:
            written["route"] = cfg

    assert CLIModelSwitchMixin._persist_model_switch_to_session(_stub(db=OKDB()), _RESULT) is True  # 期望: True——成功路径不受改动静化
    assert written["model"] == "glm-4.6" and written["sid"] == "local-test"  # 期望: 值来自 _RESULT.new_model 与 stub sid
    assert written["route"]["provider"] == "zai"  # 期望: route 双形态随行,provider 来自 _RESULT.target_provider


def test_commit_model_switch_surfaces_persist_failure(capsys: Any, monkeypatch: Any) -> None:
    """_commit_model_switch with no session store must print the persist_failed line."""
    import cli
    from hermes_cli import cli_model_switch_mixin as mixin
    monkeypatch.setattr(cli, "_cprint", lambda text: print(text))
    stub = type("Stub", (), {
        "agent": None, "model": "old-model", "provider": "zai", "_pending_one_turn_model_restore": None,
        "_session_db": None, "session_id": None,
        "_stage_and_swap_model": lambda self, r, o: True})()
    mixin._commit_model_switch(stub, _RESULT, persist_global=False)
    out = capsys.readouterr().out
    # 期望: 键回退文案(测试无 locale 层)或 en 文案——两者其一必须出现,否则提示被吞
    assert "cli.model.persist_failed" in out or "session row failed" in out
