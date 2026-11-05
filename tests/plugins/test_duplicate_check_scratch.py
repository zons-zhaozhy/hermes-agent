"""Duplicate checks target durable project code, not official scratch probes."""
from plugins.guards import duplicate_check


def test_hermes_cache_scratch_is_outside_duplicate_scope(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    scratch = home / "cache" / "scratch" / "probe.py"

    assert duplicate_check._normalize_new_py(str(scratch), str(home)) == ""


def test_project_python_remains_in_duplicate_scope(tmp_path):
    root = tmp_path / "repo"
    target = root / "app.py"

    assert duplicate_check._normalize_new_py(str(target), str(root)) == "app.py"
