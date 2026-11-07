"""Actionable doctor summaries for large state stores and unavailable toolsets."""

import contextlib
import io
import sys
import types
from types import SimpleNamespace

from hermes_cli import doctor_tools, doctor_state


def test_large_state_db_does_not_ask_to_enable_auto_prune_when_it_is_enabled():
    rows = doctor_state._render_state_db_stats(
        {"logical_size_bytes": doctor_state.STATE_DB_SIZE_WARN_BYTES + 1},
        auto_prune_enabled=True,
    )

    warning = next(row for row in rows if row[0] == "warn")
    assert "enable sessions.auto_prune" not in warning[2]
    assert "sessions.auto_prune is already enabled" in warning[2]


def test_tool_setup_summary_names_enabled_unavailable_toolsets(monkeypatch):
    unavailable = [{"name": "image_gen", "env_vars": [], "tools": ["image_generate"]}]
    monkeypatch.setattr(
        doctor_tools,
        "_enabled_cli_toolsets_for_doctor",
        lambda: {"image_gen"},
    )
    monkeypatch.setattr(
        doctor_tools,
        "_apply_doctor_tool_availability_overrides",
        lambda available, unavailable: (available, unavailable),
    )
    monkeypatch.setattr(doctor_tools, "_doctor_web_capability_rows", list)
    monkeypatch.setitem(
        sys.modules,
        "model_tools",
        SimpleNamespace(
            check_tool_availability=lambda: ([], unavailable),
            TOOLSET_REQUIREMENTS={"image_gen": {"name": "image_gen"}},
        ),
    )

    with contextlib.redirect_stdout(io.StringIO()):
        finding = doctor_tools._check_tool_availability(False)

    assert finding.issues == [
        "Run 'hermes setup' to configure missing API keys: image_gen",
    ]
