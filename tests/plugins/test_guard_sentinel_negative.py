"""guard_sentinel 负例契约测试——非活跃守卫必须被哨卫点名告警（违规必红）。

Contract:
    Preconditions: plugins/guard_sentinel/__init__.py 存在
    Postconditions: 自声明 guard 标签但未启用的插件触发 GUARD ABSENT 告警
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


def _make_plugin(key: str, enabled: bool, error: str | None) -> SimpleNamespace:
    return SimpleNamespace(
        manifest=SimpleNamespace(key=key, name=key, tags=["guard"]),
        enabled=enabled,
        error=error,
    )


def test_negative_inactive_guard_reported(monkeypatch, caplog):
    guard = _load_guard()
    guard._reported.clear()
    fake = _FakeManager({
        "good": _make_plugin("good", True, None),
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
    fake = _FakeManager({"good": _make_plugin("good", True, None)})
    import hermes_cli.plugins as hp

    monkeypatch.setattr(hp, "get_plugin_manager", lambda: fake)
    with caplog.at_level(logging.WARNING, logger=guard.logger.name):
        guard._check_guards("session_start")
    assert not any("GUARD ABSENT" in r.message for r in caplog.records)  # 期望: 活跃守卫零告警
