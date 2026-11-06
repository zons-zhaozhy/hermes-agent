"""Coding standards keep production rules while avoiding scratch-tool churn."""
from pathlib import Path

from hermes_constants import get_hermes_home
from plugins.guards import coding_standards


def test_execute_code_diagnostic_truncation_is_not_blocking() -> None:
    code = "rows = [1, 2, 3]\nprint(rows[-2:])"

    assert coding_standards.on_pre_tool_call(
        tool_name="execute_code", args={"code": code}
    ) is None


def test_hermes_cache_script_style_rules_are_not_blocking(tmp_path: Path) -> None:
    target = get_hermes_home() / "cache" / "scratch" / "probe.py"
    content = "import re\nprint(re.findall(r'\\d+', '1 2')[:2])"

    assert coding_standards.on_pre_tool_call(
        tool_name="write_file", args={"path": str(target), "content": content}
    ) is None


def test_execute_code_high_risk_rules_remain_blocking() -> None:
    code = "value = eval('1 + 1')\nprint(value)"

    verdict = coding_standards.on_pre_tool_call(
        tool_name="execute_code", args={"code": code}
    )

    assert verdict is not None
    assert verdict["action"] == "block"
    assert "R014" in verdict["message"]
