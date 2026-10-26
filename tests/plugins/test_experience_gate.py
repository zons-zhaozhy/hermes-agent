"""experience_gate 行为契约测试——正例负例全覆盖(fail-open/分页/档位/豁免)。

Regression for aml_agents 2026-10-26: 体验完备性判据机器化。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pytest

from plugins.guards import experience_gate as eg


@pytest.fixture
def enabled_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """启用配置的项目根(tmp_path)——直接调 hook,不经插件发现(单元层)。"""
    (tmp_path / ".hermes").mkdir()
    (tmp_path / ".hermes" / "experience_gate.yaml").write_text(
        "type_scale: [12, 13, 22]\n", encoding="utf-8"
    )
    return tmp_path


def _call(path: str, content: str) -> Optional[dict[str, Any]]:
    return eg.on_pre_tool_call(tool_name="write_file", args={"path": path, "content": content})


def test_no_config_fails_open(tmp_path: Path) -> None:
    """无启用配置的项目:前端写一律放行(fail-open)。"""
    target = tmp_path / "page.html"
    assert _call(str(target), "<table><tbody></tbody></table>") is None  # 期望: 无配置不执法


def test_non_frontend_extension_skipped(enabled_project: Path) -> None:
    """非前端扩展名(.py/.md):不执法。"""
    target = enabled_project / "app.py"
    assert _call(str(target), "<table><tbody>") is None  # 期望: 扩展名不在射程


def test_unpaged_table_blocked(enabled_project: Path) -> None:
    """表格渲染无分页特征:拦(核心判据)。"""
    target = enabled_project / "page.html"
    result = _call(str(target), "<div><table><tbody><tr><td>x</td></tr></tbody></table></div>")
    assert result is not None and result["action"] == "block"  # 期望: 拦
    assert "分页" in result["message"]  # 期望: 拦截信息指出修法


def test_paged_table_allowed(enabled_project: Path) -> None:
    """表格渲染带分页特征:放行。"""
    target = enabled_project / "page.html"
    body = "function renderTable(a){return pager(a)}"  # pager=分页特征
    assert _call(str(target), body) is None  # 期望: 放行


def test_off_scale_font_size_blocked(enabled_project: Path) -> None:
    """font-size 不在档位表(12/13/22):拦。"""
    target = enabled_project / "page.vue"
    result = _call(str(target), '<template><div style="font-size: 19px">x</div></template>')
    assert result is not None and result["action"] == "block"  # 期望: 拦
    assert "19px" in result["message"]  # 期望: 指明超档值


def test_in_scale_font_size_allowed(enabled_project: Path) -> None:
    """font-size 在档位表:放行。"""
    target = enabled_project / "page.vue"
    assert _call(str(target), '<template><div style="font-size: 22px">x</div></template>') is None  # 期望: 22 在档内放行


def test_inline_waiver_allows(enabled_project: Path) -> None:
    """行内 exp-ok 豁免:档位外值放行。"""
    target = enabled_project / "page.vue"
    body = '<template><div style="font-size: 19px">x</div><!-- exp-ok: 设计稿特例 --></template>'
    assert _call(str(target), body) is None  # 期望: 豁免放行


def test_patch_tool_in_scope(enabled_project: Path) -> None:
    """patch 通道(new_string)同受执法。"""
    target = enabled_project / "page.html"
    result = eg.on_pre_tool_call(
        tool_name="patch",
        args={"path": str(target), "new_string": "<table><tbody></tbody></table>"},
    )
    assert result is not None and result["action"] == "block"  # 期望: patch 同拦
