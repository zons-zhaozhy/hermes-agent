"""pre_write 从 execute_code 提取目标路径的判据（红绿测试）。

规则来源（独立期望）：execute_code 里带斜杠的字符串字面量被当成待写目标，
但单独的分隔符（'/'）、首尾斜杠去掉后为空的字面量不可能是文件路径；
把它们当成目标会让只读分析脚本因「根目录没被读过」而被拦。
真正的路径（含目录分隔的绝对/相对路径、以 .py/.md 结尾的文件名）必须照常返回。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from plugins.guards import pre_write


def test_separator_only_literal_is_not_a_path() -> None:
    code = "parts = name.split('/')\nrows = [p for p in parts if p]"
    assert pre_write._extract_paths_from_code(code) == []


def test_slash_with_spaces_is_not_a_path() -> None:
    code = "text = ' / '.join(names)"
    assert pre_write._extract_paths_from_code(code) == []


def test_real_paths_are_still_extracted() -> None:
    code = "open('/tmp/data/out.csv')\nopen('notes.md')\nopen('a/b.py')"
    assert pre_write._extract_paths_from_code(code) == [
        "/tmp/data/out.csv",
        "notes.md",
        "a/b.py",
    ]
