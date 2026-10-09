"""duplicate_check v3 回归 — 符号级判定 / 分档反馈 / 按文件豁免。

被测对象是插件模块本体（plugins/guards/duplicate_check.py），
在 tmp_path 里建临时 git 仓库隔离，不触碰真实项目。
"""
from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

GUARD_PATH = (Path(__file__).resolve().parents[1]
              / "plugins" / "guards" / "duplicate_check.py")


def _load_guard() -> Any:
    """按路径加载守卫模块（每次都重载，避免跨用例缓存污染）。"""
    spec = importlib.util.spec_from_file_location("dupcheck_under_test", GUARD_PATH)
    assert spec is not None, "无法为守卫模块构造 spec"  # 期望: 测试路径存在
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None, "守卫模块 loader 缺失"  # 期望: 源码模块可执行
    spec.loader.exec_module(mod)
    return mod


def _make_repo(tmp_path: Path) -> Path:
    """建临时 git 仓库：含一个既有文件（顶层 ConnectorSpec + 小写主题词符号）。"""
    repo = tmp_path / "proj"
    pkg = repo / "pkg"
    pkg.mkdir(parents=True)
    (pkg / "connector_spec.py").write_text(
        "class ConnectorSpec:\n    pass\n\n\ndef get_connector(name):\n    return name\n",
        encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    return repo


def _hook(repo: Path, path: str, content: str) -> Optional[dict[str, str]]:
    """调用守卫 pre_tool_call hook。"""
    m = _load_guard()
    return m._on_pre_tool_call(
        tool_name="write_file", args={"path": path, "content": content}, cwd=str(repo))


def test_topic_keyword_only_is_advisory(tmp_path: Path) -> None:
    """仅主题词命中（异目录、顶层符号全新）→ 放行。"""
    repo = _make_repo(tmp_path)
    content = "class BrandNewWorker:\n    pass\n"
    result = _hook(repo, "other/connector_module.py", content)
    assert result is None  # 期望: 主题词命中不等于等价实现，放行


def test_same_dir_stem_collision_blocks(tmp_path: Path) -> None:
    """同目录文件名词干共享 → 阻断（策略1）。"""
    repo = _make_repo(tmp_path)
    result = _hook(repo, "pkg/connector_broker.py", "class BrandNewWorker:\n    pass\n")
    assert result is not None  # 期望: 同目录共享词干是硬证据，阻断
    assert "文件名共享词干" in result["message"]  # 期望: 反馈指明命中词干


def test_same_top_level_symbol_blocks(tmp_path: Path) -> None:
    """新文件声明已有同名顶层符号 → 阻断，且消息含证据与下一步。"""
    repo = _make_repo(tmp_path)
    result = _hook(repo, "pkg/zzbeta.py", "class ConnectorSpec:\n    pass\n")
    assert result is not None  # 期望: 同名符号是硬证据，阻断
    msg = result["message"]
    assert "同名符号已存在于" in msg  # 期望: 反馈给出证据出处
    assert "复用已有实现" in msg      # 期望: 反馈给出可执行下一步


def test_inline_ok_marker_passes(tmp_path: Path) -> None:
    """按文件豁免标记有效（理由 ≥8 字符）→ 放行。"""
    repo = _make_repo(tmp_path)
    content = "# dupcheck-ok: 内核连接器层首个消费步骤，非重复实现\nclass X:\n    pass\n"
    result = _hook(repo, "pkg/connector_broker.py", content)
    assert result is None  # 期望: 显式豁免放行


def test_short_marker_reason_does_not_pass(tmp_path: Path) -> None:
    """豁免理由过短 → 视为无效豁免，同名符号仍阻断。"""
    repo = _make_repo(tmp_path)
    content = "# dupcheck-ok: 略\nclass ConnectorSpec:\n    pass\n"
    result = _hook(repo, "pkg/zzbeta.py", content)
    assert result is not None  # 期望: 无理由豁免无效，硬证据仍阻断


def test_framework_method_names_ignored(tmp_path: Path) -> None:
    """类内框架约定方法名（validate_config）不计入同名判定 → 放行。"""
    repo = _make_repo(tmp_path)
    content = ("class AnotherBrandNew(_Base):\n"
               "    def validate_config(self) -> list:\n"
               "        return []\n")
    result = _hook(repo, "pkg/zzdelta.py", content)
    assert result is None  # 期望: 人人同名的方法名不是重复信号
