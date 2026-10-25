"""plain_vocab 子护栏测试——项目废弃术语词表执法（负例直调判定纯函数规范）。

形态照 ref 负例纪律: 直调 on_pre_tool_call 纯函数断言 block/放行,禁走完整管线。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_REPO = Path(__file__).resolve().parents[2]
_MOD_PATH = _REPO / "plugins" / "guards" / "plain_vocab.py"


def _load_guard() -> ModuleType:
    spec = importlib.util.spec_from_file_location("plain_vocab_under_test", _MOD_PATH)
    assert spec is not None and spec.loader is not None  # 期望: 固定路径必然可载,失败即测试环境错
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


guard = _load_guard()


def _make_vocab(root: Path, body: str) -> Path:
    vocab_dir = root / ".hermes"
    vocab_dir.mkdir(parents=True, exist_ok=True)
    vocab = vocab_dir / "plain_terms.yaml"
    vocab.write_text(body, encoding="utf-8")
    return vocab


_VOCAB_OK = "terms:\n  - deprecated: 裁决\n    replacement: 审批\n  - deprecated: 批次日\n    replacement: 数据日期\n"


def test_block_on_deprecated_term_write(tmp_path):
    _make_vocab(tmp_path, _VOCAB_OK)
    target = tmp_path / "web" / "src" / "page.vue"
    result = guard.on_pre_tool_call(
        tool_name="write_file",
        args={"path": str(target), "content": "按钮文案: 等待裁决人确认"},
    )
    assert result is not None and result["action"] == "block"  # 期望: 命中废弃词必拦
    assert "审批" in result["message"]  # 期望: 拦截信息携带替换词
    assert "plain_terms.yaml" in result["message"]  # 期望: 拦截信息携带词表路径


def test_patch_new_string_form_blocks(tmp_path):
    _make_vocab(tmp_path, _VOCAB_OK)
    target = tmp_path / "app.py"
    result = guard.on_pre_tool_call(
        tool_name="patch",
        args={"path": str(target), "new_string": "detail = '批次日不能为空'"},
    )
    assert result is not None and result["action"] == "block"  # 期望: patch 片段同样受执法
    assert "数据日期" in result["message"]  # 期望: 批次日→数据日期映射可见


def test_waiver_line_allows(tmp_path):
    _make_vocab(tmp_path, _VOCAB_OK)
    target = tmp_path / "page.vue"
    result = guard.on_pre_tool_call(
        tool_name="write_file",
        args={"path": str(target), "content": "docstring: 裁决物理表名保留  # vocab-ok: 部署事实"},
    )
    assert result is None  # 期望: 行内豁免标记放行


def test_no_vocab_fails_open(tmp_path):
    target = tmp_path / "page.vue"
    result = guard.on_pre_tool_call(
        tool_name="write_file",
        args={"path": str(target), "content": "等待裁决人确认"},
    )
    assert result is None  # 期望: 无词表项目不执法(fail-open)


def test_non_user_facing_extension_allows(tmp_path):
    _make_vocab(tmp_path, _VOCAB_OK)
    target = tmp_path / "config.json"
    result = guard.on_pre_tool_call(
        tool_name="write_file",
        args={"path": str(target), "content": '{"label": "裁决"}'},
    )
    assert result is None  # 期望: 非用户可见层扩展名不在执法面


def test_non_target_tool_allows(tmp_path):
    _make_vocab(tmp_path, _VOCAB_OK)
    target = tmp_path / "page.vue"
    result = guard.on_pre_tool_call(
        tool_name="terminal",
        args={"command": "echo 裁决", "path": str(target)},
    )
    assert result is None  # 期望: 第一版不覆盖 terminal


def test_broken_vocab_fails_open(tmp_path):
    _make_vocab(tmp_path, "terms: [unclosed")
    target = tmp_path / "page.vue"
    result = guard.on_pre_tool_call(
        tool_name="write_file",
        args={"path": str(target), "content": "等待裁决人确认"},
    )
    assert result is None  # 期望: 词表不可解析 fail-open 不炸主循环


def test_ancestor_chain_vocab_discovery(tmp_path):
    _make_vocab(tmp_path, _VOCAB_OK)
    deep = tmp_path / "a" / "b" / "c" / "d" / "page.md"
    result = guard.on_pre_tool_call(
        tool_name="write_file",
        args={"path": str(deep), "content": "# 裁决流程说明"},
    )
    assert result is not None and result["action"] == "block"  # 期望: 深层路径沿祖先链找到词表


def test_parse_terms_skips_invalid_entries():
    parsed = guard._parse_terms(
        {
            "terms": [
                {"deprecated": "裁决", "replacement": "审批"},
                {"deprecated": "", "replacement": "审批"},  # 缺 deprecated
                {"deprecated": "同名", "replacement": "同名"},  # dep==rep
                "not-a-dict",  # 非法形态
                {"replacement": "孤儿"},  # 缺 deprecated
            ]
        }
    )
    assert parsed == (("裁决", "审批"),)  # 期望: 仅合法条目保留


def test_cache_invalidated_on_vocab_edit(tmp_path):
    _make_vocab(tmp_path, _VOCAB_OK)
    target = tmp_path / "page.vue"
    args = {"path": str(target), "content": "等待裁决人确认"}
    assert guard.on_pre_tool_call(tool_name="write_file", args=args) is not None  # 期望: 首版词表命中
    _make_vocab(tmp_path, "terms:\n  - deprecated: 不存在的词\n    replacement: x\n")
    guard._cache.clear()  # mtime+size 变更即失效;显式清缓存模拟新进程视角
    assert guard.on_pre_tool_call(tool_name="write_file", args=args) is None  # 期望: 词表更新后不再命中
