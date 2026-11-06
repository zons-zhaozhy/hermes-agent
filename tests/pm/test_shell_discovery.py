"""The local transport can use Git Bash without a PATH entry."""
import os
from pathlib import Path
import subprocess

import pytest


def test_macos_system_bash_path_prefers_homebrew_candidates():
    """A default macOS PATH may resolve /bin/bash before Homebrew Bash 5."""
    from pm.shell import _posix_bash_candidates

    candidates = _posix_bash_candidates("/bin/bash", is_macos=True)

    assert candidates[:2] == ["/opt/homebrew/bin/bash", "/usr/local/bin/bash"]
    assert "/bin/bash" in candidates


def test_explicit_posix_bash_path_is_not_replaced_by_homebrew():
    """A caller that deliberately puts another bash first keeps that choice."""
    from pm.shell import _posix_bash_candidates

    candidates = _posix_bash_candidates("/custom/bash", is_macos=True)

    assert candidates[0] == "/custom/bash"
    assert "/opt/homebrew/bin/bash" not in candidates


def test_linux_path_bash_remains_first():
    from pm.shell import _posix_bash_candidates

    candidates = _posix_bash_candidates("/usr/bin/bash", is_macos=False)

    assert candidates[0] == "/usr/bin/bash"
    assert "/opt/homebrew/bin/bash" not in candidates


@pytest.mark.platforms("windows")
def test_local_transport_finds_conventional_bash_without_path(monkeypatch, tmp_path):
    from tools.environments.local import _find_bash

    program_files = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
    conventional = program_files / "Git/bin/bash.exe"
    if not conventional.is_file():
        pytest.skip("a conventional Git for Windows installation is required")

    monkeypatch.setenv("ProgramFiles", str(program_files))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "tools"))
    empty_bin = tmp_path / "empty-bin"
    alias_bin = tmp_path / "WindowsApps"
    empty_bin.mkdir()
    alias_bin.mkdir()
    (alias_bin / "bash.exe").write_bytes(b"unusable execution alias")
    monkeypatch.setenv("PATHEXT", ".EXE;.CMD")

    for directory in (empty_bin, alias_bin):
        monkeypatch.setenv("PATH", str(directory))
        selected = _find_bash()
        assert Path(selected) == conventional
        child = subprocess.run(
            [selected, "--noprofile", "--norc", "-c", "printf shell-ready"],
            cwd=tmp_path, capture_output=True, text=True, timeout=10,
            check=False,
        )
        assert child.returncode == 0, child.stderr
        assert child.stdout == "shell-ready"
