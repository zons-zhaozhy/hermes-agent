"""rule_reinjection 规则源探测链不变量。

覆盖 gateway/cron 类会话的第三候选(代码根):cwd 不在仓库内时
仍能找到仓库根的 .hermes-rules.md——该缺口曾致 ~/.hermes 为 cwd
的会话永久零注入(gateway.log 0924 23:21 起实测告警)。
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from plugins.discipline.rule_reinjection import _find_rules_file


def test_probe_finds_rules_from_repo_cwd() -> None:
    """仓库内 cwd:首候选(cwd)直接命中,无需回退。"""
    # 期望: 本测试自身在仓库内运行,cwd 候选即仓库根规则文件
    repo_root = Path(__file__).resolve().parents[2]
    result = _find_rules_file()
    assert result == repo_root / ".hermes-rules.md"


def test_probe_finds_rules_from_outside_repo(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """cwd 在仓库外(如 ~/.hermes):cwd 与 git root 候选皆空,代码根候选兜底命中。

    Contract:
      Preconditions: cwd 被切到无 .git 的临时目录。
      Postconditions: 返回非 None,且指向本仓库根的 .hermes-rules.md。
    """
    # 期望: 代码根候选(文件上三级=仓库根)命中,禁返回 None
    outside = tmp_path / "hermes-home-like"
    outside.mkdir()
    monkeypatch.chdir(outside)
    result = _find_rules_file()
    repo_root = Path(__file__).resolve().parents[2]
    assert result == repo_root / ".hermes-rules.md"


def test_probe_returns_none_when_no_rules_anywhere(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """三候选全空(仓库外 cwd + 代码根无规则文件):返回 None,由调用方告警。

    模拟方式:将模块 __file__ 指向临时树,代码根候选落空;cwd 也落空。
    """
    # 期望: 候选链全部 miss 时返回 None(不抛异常)
    import plugins.discipline.rule_reinjection as rr

    fake_tree = tmp_path / "pkg" / "plugins" / "discipline"
    fake_tree.mkdir(parents=True)
    fake_mod = fake_tree / "rule_reinjection.py"
    fake_mod.write_text("", encoding="utf-8")
    outside = tmp_path / "cwd"
    outside.mkdir()
    monkeypatch.chdir(outside)
    monkeypatch.setattr(rr, "__file__", str(fake_mod), raising=False)
    assert _find_rules_file() is None
