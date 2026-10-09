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


def test_same_dir_stem_only_is_advisory(tmp_path: Path) -> None:
    """同目录文件名词干共享但无同名符号 → 提示档放行（名字档不足以阻断）。"""
    repo = _make_repo(tmp_path)
    result = _hook(repo, "pkg/connector_broker.py", "class BrandNewWorker:\n    pass\n")
    assert result is None  # 期望: 仅名字档命中 → 放行（阻断须符号级佐证）


def test_same_dir_stem_with_symbol_collision_blocks(tmp_path: Path) -> None:
    """同目录词干共享 + 同名符号佐证 → 阻断，且消息含词干与符号证据。"""
    repo = _make_repo(tmp_path)
    result = _hook(repo, "pkg/connector_broker.py", "class ConnectorSpec:\n    pass\n")
    assert result is not None  # 期望: 词干+同名符号=内容级硬证据，阻断
    assert "文件名共享词干" in result["message"]  # 期望: 反馈指明名字档命中
    assert "同名符号" in result["message"]        # 期望: 反馈指明符号级佐证

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


def test_tool_failure_is_reported_not_silent(tmp_path: Path, caplog: Any) -> None:
    """检索工具故障（rg 缺失/超时）→ 提示档放行且明示降级，不谎报零重复。"""
    repo = _make_repo(tmp_path)

    def _fake_rg_lines(pattern: str, cwd: str) -> Optional[list[str]]:
        return None  # 期望: 模拟 rg 不可用（与「零命中」严格区分）

    m = _load_guard()
    m._rg_lines = _fake_rg_lines  # type: ignore[method-assign]  # 期望: 注入 rg 故障
    with caplog.at_level("INFO", logger="dupcheck_under_test"):
        result = m._on_pre_tool_call(
            tool_name="write_file",
            args={"path": "pkg/zzomega.py", "content": "class BrandNewOmega:\n    pass\n"},
            cwd=str(repo))
    assert result is None            # 期望: 基础设施故障不拦人（fail-open）
    assert "检索降级" in caplog.text  # 期望: 降级可见，不静默谎报「零重复」


# ── v4 射程：execute_code / terminal 通道 ─────────────────────────────


def test_execute_code_with_open_blocks(tmp_path: Path) -> None:
    """execute_code 通道：with open(...) as f + f.write 字面量 → 同名符号阻断。"""
    repo = _make_repo(tmp_path)
    code = ('with open("pkg/zzeta.py", "w") as f:\n'
            '    f.write("class ConnectorSpec:\\n    pass\\n")\n')
    m = _load_guard()
    result = m._on_pre_tool_call(tool_name="execute_code", args={"code": code}, cwd=str(repo))
    assert result is not None            # 期望: execute_code 不再能绕过查重
    assert result["action"] == "block"   # 期望: 同名符号=内容级硬证据
    assert "同名符号" in result["message"]  # 期望: 反馈给出符号级证据


def test_execute_code_write_text_blocks(tmp_path: Path) -> None:
    """execute_code 通道：Path().write_text 字面量 → 同名符号阻断。"""
    repo = _make_repo(tmp_path)
    code = ('from pathlib import Path\n'
            'Path("pkg/zzbeta.py").write_text("class ConnectorSpec:\\n    pass\\n")\n')
    m = _load_guard()
    result = m._on_pre_tool_call(tool_name="execute_code", args={"code": code}, cwd=str(repo))
    assert result is not None            # 期望: write_text 属被识别写形态
    assert result["action"] == "block"   # 期望: 同名符号阻断


def test_terminal_heredoc_blocks(tmp_path: Path) -> None:
    """terminal 通道：heredoc 写 .py 且正文含同名符号 → 阻断。"""
    repo = _make_repo(tmp_path)
    cmd = ("cat > pkg/zzgamma.py <<'EOF'\n"
           "class ConnectorSpec:\n"
           "    pass\n"
           "EOF\n")
    m = _load_guard()
    result = m._on_pre_tool_call(tool_name="terminal", args={"command": cmd}, cwd=str(repo))
    assert result is not None            # 期望: terminal 直写不再能绕过查重
    assert result["action"] == "block"   # 期望: heredoc 正文可解析 → 内容级判定


def test_execute_code_dynamic_path_not_judged(tmp_path: Path) -> None:
    """execute_code 通道：路径非字面量 → 不猜（不误拦）。"""
    repo = _make_repo(tmp_path)
    code = 'name = "zzdelta"\nopen("pkg/" + name + ".py", "w").write("x = 1\\n")\n'
    m = _load_guard()
    result = m._on_pre_tool_call(tool_name="execute_code", args={"code": code}, cwd=str(repo))
    assert result is None  # 期望: 动态路径无法解析 → 不猜不拦（射程边界如实标注）


def test_execute_code_unknown_content_is_advisory(tmp_path: Path, caplog: Any) -> None:
    """execute_code 通道：路径字面量但内容非字面量 → 提示档明示比对未生效。"""
    repo = _make_repo(tmp_path)
    code = ('data = build_payload()\n'
            'with open("pkg/zzepsilon.py", "w") as f:\n'
            '    f.write(data)\n')
    m = _load_guard()
    with caplog.at_level("INFO", logger="dupcheck_under_test"):
        result = m._on_pre_tool_call(tool_name="execute_code", args={"code": code}, cwd=str(repo))
    assert result is None                        # 期望: 提示档不阻断写入
    assert "符号级比对未生效" in caplog.text        # 期望: 内容不可解析时降级可见，不静默
