"""patch-first 直接行为契约：sed 分类委托唯一事实源。"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

patch_first = importlib.import_module("plugins.patch-first")


def test_terminal_read_sed_passes_and_write_sed_blocks() -> None:
    assert patch_first._is_sed_write(
        "terminal", {"command": "sed -n '10,20p' src.py"}
    ) is False
    assert patch_first._is_sed_write(
        "terminal", {"command": "sed -n '/pattern/p' src.py"}
    ) is False

    for command in (
        "sed -i 's/a/b/' src.py",
        "sed 's/a/b/' src.py",
        "sed -n '1p' src.py; sed -i 's/a/b/' src.py; printf ' -p'",
    ):
        assert patch_first._is_sed_write("terminal", {"command": command}) is True


def test_execute_code_embedded_sed_write_is_blocked() -> None:
    code = "subprocess.run(['sed', '-i', 's/a/b/', 'src.py'], check=True)"
    assert patch_first._is_sed_write("execute_code", {"code": code}) is True


def test_execute_code_write_open_blocks_and_read_open_passes() -> None:
    assert patch_first._is_sed_write(
        "execute_code", {"code": "open('out.txt', 'w')"}
    ) is True
    assert patch_first._is_sed_write(
        "execute_code", {"code": "with open('in.txt') as f: print(f.read())"}
    ) is False


def test_pre_tool_callback_blocks_only_detected_write() -> None:
    assert patch_first._on_pre_tool_call(
        tool_name="terminal", args={"command": "sed -n '1,2p' README.md"}
    ) is None
    blocked = patch_first._on_pre_tool_call(
        tool_name="terminal", args={"command": "sed -i 's/a/b/' README.md"}
    )
    assert blocked is not None
    assert blocked["action"] == "block"
    assert "[patch-first]" in blocked["message"]


def test_non_string_and_malformed_input_do_not_raise() -> None:
    assert patch_first._is_sed_write("terminal", {"command": None}) is False
    assert patch_first._is_sed_write("execute_code", {"code": "sed -n '1p"}) in {
        True,
        False,
    }
    assert patch_first._is_sed_write("unknown", {"command": "sed -i 's/a/b/' x"}) is False
