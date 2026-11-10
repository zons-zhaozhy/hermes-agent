"""guard_sentinel 负例契约测试——非活跃守卫必须被哨卫点名告警（违规必红）。

Contract:
    Preconditions: plugins/guard_sentinel/__init__.py 存在
    Postconditions: 自声明 guard 标签但未启用的插件触发 GUARD ABSENT 日志；
                    transform_llm_output 不改写用户答案
"""
import importlib.util
import logging
from pathlib import Path
from types import ModuleType, SimpleNamespace


def _load_guard() -> ModuleType:
    repo = Path(__file__).resolve().parents[2]
    path = repo / "plugins" / "guard_sentinel" / "__init__.py"
    spec = importlib.util.spec_from_file_location("guard_sentinel_under_test", path)
    assert spec is not None and spec.loader is not None  # 期望: 插件文件可定位
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeManager:
    def __init__(self, plugins: dict):
        self._plugins = plugins

    def gate_plane_report(self, session_id: str | None = None):
        from hermes_cli.gate_plane import GatePlane, build_gate_report

        return build_gate_report(self._plugins.values(), GatePlane(), session_id=session_id)


def _make_plugin(
    key: str,
    enabled: bool,
    error: str | None,
    *,
    tags: list[str] | None = None,
    hooks: list[str] | None = None,
    registered_hooks: list[str] | None = None,
    gate_policy: str = "required",
) -> SimpleNamespace:
    return SimpleNamespace(
        manifest=SimpleNamespace(
            key=key,
            name=key,
            tags=["guard"] if tags is None else tags,
            provides_hooks=hooks or [],
            gate_policy=gate_policy,
        ),
        enabled=enabled,
        error=error,
        hooks_registered=registered_hooks or [],
    )


def test_negative_inactive_guard_reported(monkeypatch, caplog):
    guard = _load_guard()
    guard._reported.clear()
    fake = _FakeManager({
        "good": _make_plugin(
            "good", True, None,
            hooks=["pre_verify"], registered_hooks=["pre_verify"],
        ),
        "dead": _make_plugin("dead", False, "load failed"),
    })
    import hermes_cli.plugins as hp

    monkeypatch.setattr(hp, "get_plugin_manager", lambda: fake)
    with caplog.at_level(logging.WARNING, logger=guard.logger.name):
        guard._check_guards("session_start")
    assert any("GUARD ABSENT" in r.message for r in caplog.records)  # 期望: 非活跃守卫被点名（拦截静默失效）


def test_negative_active_guard_not_reported(monkeypatch, caplog):
    guard = _load_guard()
    guard._reported.clear()
    fake = _FakeManager({
        "good": _make_plugin(
            "good", True, None,
            hooks=["pre_verify"], registered_hooks=["pre_verify"],
        ),
    })
    import hermes_cli.plugins as hp

    monkeypatch.setattr(hp, "get_plugin_manager", lambda: fake)
    with caplog.at_level(logging.WARNING, logger=guard.logger.name):
        guard._check_guards("session_start")
    assert not any("GUARD ABSENT" in r.message for r in caplog.records)  # 期望: 活跃守卫零告警


def test_untagged_hook_observer_is_not_misreported_as_guard(monkeypatch, caplog):
    """Generic hook users such as cleanup or telemetry are not guards merely by hook name."""
    guard = _load_guard()
    guard._reported.clear()
    fake = _FakeManager({
        "disk-cleanup": _make_plugin(
            "disk-cleanup", False, "disabled via config",
            tags=[], hooks=["pre_tool_call"],
        ),
    })
    import hermes_cli.plugins as hp

    monkeypatch.setattr(hp, "get_plugin_manager", lambda: fake)
    with caplog.at_level(logging.WARNING, logger=guard.logger.name):
        guard._check_guards("session_start")
    assert not any("GUARD ABSENT" in r.message for r in caplog.records)


def test_enabled_guard_that_did_not_register_its_hook_is_reported(monkeypatch, caplog):
    """Enabled is not healthy when the declared gate hook never reached the registry."""
    guard = _load_guard()
    guard._reported.clear()
    fake = _FakeManager({
        "hollow-guard": _make_plugin(
            "hollow-guard", True, None,
            tags=["guard"], hooks=["pre_verify"], registered_hooks=[],
        ),
    })
    import hermes_cli.plugins as hp

    monkeypatch.setattr(hp, "get_plugin_manager", lambda: fake)
    with caplog.at_level(logging.WARNING, logger=guard.logger.name):
        guard._check_guards("session_start")
    assert any("hollow-guard" in r.message and "not registered" in r.message for r in caplog.records)


def test_absent_guard_diagnostic_is_logged_without_rewriting_answer(
    monkeypatch, caplog,
):
    guard = _load_guard()
    guard._reported.clear()
    guard._pending.clear()
    fake = _FakeManager({
        "dead-hook-guard": _make_plugin(
            "dead-hook-guard", False, "disabled via config",
            tags=["guard"], hooks=["pre_verify"],
        ),
    })
    import hermes_cli.plugins as hp

    monkeypatch.setattr(hp, "get_plugin_manager", lambda: fake)
    with caplog.at_level(logging.WARNING, logger=guard.logger.name):
        guard.on_session_start(session_id="sentinel-visible")
        answer = guard.transform_llm_output(
            "normal answer", session_id="sentinel-visible",
        )
    assert answer is None
    assert "门禁缺席" in caplog.text and "dead-hook-guard" in caplog.text
    assert guard.transform_llm_output(
        "next answer", session_id="sentinel-visible",
    ) is None


def test_sentinel_consumes_the_shared_gate_plane_report(monkeypatch, caplog):
    """The one-shot warning has the same authority as `hermes plugins gates`."""
    guard = _load_guard()
    guard._reported.clear()
    guard._pending.clear()
    report = {
        "overall_status": "unknown",
        "suites": [{
            "name": "completion",
            "status": "unknown",
            "missing": [{
                "name": "shared-dead-guard",
                "reason": "not enabled",
                "hooks": ["pre_verify"],
                "policy": "required",
            }],
        }],
    }
    fake = SimpleNamespace(gate_plane_report=lambda session_id=None: report)
    import hermes_cli.plugins as hp

    monkeypatch.setattr(hp, "get_plugin_manager", lambda: fake)
    with caplog.at_level(logging.WARNING, logger=guard.logger.name):
        guard.on_session_start(session_id="shared-report")
        answer = guard.transform_llm_output(
            "normal answer", session_id="shared-report",
        )

    assert answer is None
    assert "门禁缺席" in caplog.text and "shared-dead-guard" in caplog.text


def test_required_manual_guard_without_evidence_is_logged(
    monkeypatch, caplog,
):
    guard = _load_guard()
    guard._reported.clear()
    guard._pending.clear()
    report = {
        "overall_status": "unknown",
        "suites": [],
        "manual_guards": [{
            "name": "fork-guard",
            "policy": "required",
            "reason": "no fresh audit evidence",
            "status": "unknown",
        }],
    }
    fake = SimpleNamespace(gate_plane_report=lambda session_id=None: report)
    import hermes_cli.plugins as hp

    monkeypatch.setattr(hp, "get_plugin_manager", lambda: fake)
    with caplog.at_level(logging.WARNING, logger=guard.logger.name):
        guard.on_session_start(session_id="manual-required")
        answer = guard.transform_llm_output(
            "normal answer", session_id="manual-required",
        )

    assert answer is None
    assert "门禁缺席" in caplog.text and "fork-guard" in caplog.text
    assert "no fresh audit evidence" in caplog.text


def test_required_manual_guard_failed_audit_is_logged(monkeypatch, caplog):
    guard = _load_guard()
    guard._reported.clear()
    guard._pending.clear()
    report = {
        "overall_status": "unknown",
        "suites": [],
        "manual_guards": [{
            "name": "fork-guard",
            "policy": "required",
            "reason": "latest audit failed",
            "status": "fail",
        }],
    }
    fake = SimpleNamespace(gate_plane_report=lambda session_id=None: report)
    import hermes_cli.plugins as hp

    monkeypatch.setattr(hp, "get_plugin_manager", lambda: fake)
    with caplog.at_level(logging.WARNING, logger=guard.logger.name):
        guard.on_session_start(session_id="manual-failed")
        answer = guard.transform_llm_output(
            "normal answer", session_id="manual-failed",
        )

    assert answer is None
    assert "门禁缺席" in caplog.text and "fork-guard" in caplog.text
    assert "latest audit failed" in caplog.text


def test_disabled_advisory_guard_is_not_reported_as_absent(monkeypatch, caplog):
    """An optional aid that the profile deliberately leaves off is not a broken defense."""
    guard = _load_guard()
    guard._reported.clear()
    guard._pending.clear()
    fake = _FakeManager({
        "baseline-guard": _make_plugin(
            "baseline-guard", True, None,
            hooks=["pre_llm_call"], registered_hooks=["pre_llm_call"],
        ),
        "style-guard": _make_plugin(
            "style-guard", False, None,
            hooks=["pre_llm_call"], registered_hooks=["pre_llm_call"],
            gate_policy="advisory",
        ),
    })
    import hermes_cli.plugins as hp

    monkeypatch.setattr(hp, "get_plugin_manager", lambda: fake)
    with caplog.at_level(logging.WARNING, logger=guard.logger.name):
        guard.on_session_start(session_id="advisory-disabled")

    assert not any("GUARD ABSENT" in record.message for record in caplog.records)
    assert guard.transform_llm_output(
        "normal answer", session_id="advisory-disabled",
    ) is None


def test_broken_enabled_advisory_guard_is_still_reported(monkeypatch, caplog):
    """Opt-in is not a license to register a hollow or failing optional guard."""
    guard = _load_guard()
    guard._reported.clear()
    guard._pending.clear()
    fake = _FakeManager({
        "broken-advisory": _make_plugin(
            "broken-advisory", True, "import failed",
            hooks=["transform_llm_output"], registered_hooks=["transform_llm_output"],
            gate_policy="advisory",
        ),
    })
    import hermes_cli.plugins as hp

    monkeypatch.setattr(hp, "get_plugin_manager", lambda: fake)
    with caplog.at_level(logging.WARNING, logger=guard.logger.name):
        guard._check_guards("session_start")

    assert any(
        "broken-advisory" in record.message and "GUARD ABSENT" in record.message
        for record in caplog.records
    )


def test_session_end_clears_pending_guard_notice(monkeypatch):
    guard = _load_guard()
    guard._reported.clear()
    guard._pending.clear()
    fake = _FakeManager({
        "dead-hook-guard": _make_plugin(
            "dead-hook-guard", False, "disabled via config",
            tags=["guard"], hooks=["pre_verify"],
        ),
    })
    import hermes_cli.plugins as hp

    monkeypatch.setattr(hp, "get_plugin_manager", lambda: fake)
    guard.on_session_start(session_id="sentinel-ended")
    guard.on_session_end(session_id="sentinel-ended")

    assert "sentinel-ended" not in guard._pending


def test_guard_identity_comes_from_real_manifest(tmp_path, monkeypatch, caplog):
    """Parse plugin.yaml through PluginManager so the sentinel's production field path is covered."""
    import hermes_yaml as yaml

    import hermes_cli.plugins as hp

    plugin = tmp_path / "plugins" / "manifest-guard"
    plugin.mkdir(parents=True)
    (plugin / "plugin.yaml").write_text(yaml.safe_dump({
        "name": "manifest-guard",
        "version": "0.1.0",
        "hooks": ["pre_verify"],
        "tags": ["guard"],
    }), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(tmp_path / "no-bundled"))
    monkeypatch.setenv("HERMES_ENABLE_PROJECT_PLUGINS", "0")
    manager = hp.PluginManager()
    manager.discover_and_load()
    monkeypatch.setattr(hp, "get_plugin_manager", lambda: manager)

    guard = _load_guard()
    guard._reported.clear()
    with caplog.at_level(logging.WARNING, logger=guard.logger.name):
        guard._check_guards("session_start")

    assert any("manifest-guard" in r.message for r in caplog.records)


def test_sentinel_check_failure_is_logged_unknown(monkeypatch, caplog):
    guard = _load_guard()
    guard._reported.clear()
    guard._pending.clear()

    import hermes_cli.plugins as hp

    def _broken_manager():
        raise RuntimeError("plugin registry unavailable")

    monkeypatch.setattr(hp, "get_plugin_manager", _broken_manager)
    with caplog.at_level(logging.WARNING, logger=guard.logger.name):
        guard.on_session_start(session_id="sentinel-check-broken")
        answer = guard.transform_llm_output(
            "normal answer", session_id="sentinel-check-broken",
        )

    assert answer is None
    assert "门禁状态不可用 / UNKNOWN" in caplog.text
