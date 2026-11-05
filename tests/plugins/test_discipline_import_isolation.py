"""One broken discipline sub-guard must not remove the whole suite."""

from __future__ import annotations

import sys
import types

from plugins import discipline


class _Ctx:
    def __init__(self) -> None:
        self.registered: list[str] = []

    def register_hook(self, _name, _fn) -> None:
        return None

    def register_system_prompt_section(self, name, _renderer) -> None:
        self.registered.append(name)


def test_single_import_failure_does_not_poison_other_subguards(monkeypatch) -> None:
    registered: list[str] = []

    def fake_import(name: str):
        short = name.rsplit(".", 1)[-1]
        if short == "db_safety":
            raise ImportError("optional driver unavailable")
        module = types.ModuleType(name)

        def register(_ctx):
            registered.append(short)

        module.register = register
        return module

    monkeypatch.setattr(discipline.importlib, "import_module", fake_import)
    monkeypatch.setattr(discipline, "_render_section", lambda _section_id: lambda _info: "")
    ctx = _Ctx()

    discipline.register(ctx)

    assert "db_safety" not in registered
    assert {"no_guessing", "tool_safety", "test_discipline"} <= set(registered)
