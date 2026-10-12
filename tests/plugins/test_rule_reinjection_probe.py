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


def test_digest_keeps_every_line_when_under_cap() -> None:
    """上限内:去空行后逐行保留,无截断标记,末行必须在位。"""
    import plugins.discipline.rule_reinjection as rr

    digest = rr._digest("# 标题\n\n## 第一节\n- 规则一\n## 末节\n- 最后一条规则\n")
    assert "…(截断,全文见规则文件)" not in digest  # 期望: 文字量在上限内,不触发截断分支
    assert digest.splitlines()[0] == "# 标题"  # 期望: 首行原样保留(头部优先)
    assert digest.splitlines()[-1] == "- 最后一条规则"  # 期望: 末行在位,尾部未被丢弃


def test_digest_warns_loudly_when_truncating(caplog: pytest.LogCaptureFixture) -> None:
    """超上限:必须响亮告警并报出本轮未注入的行数——静默丢尾违反 A3。"""
    import logging

    import plugins.discipline.rule_reinjection as rr

    # 期望: 首行长度=上限-1,+1 换行后恰好用满上限,故第二行起触发截断,丢失 2 行
    filler = "x" * (rr._MAX_DIGEST_CHARS - 1)
    with caplog.at_level(logging.WARNING, logger="plugins.discipline.rule_reinjection"):
        digest = rr._digest(f"{filler}\n尾部规则一\n尾部规则二\n")
    assert digest.endswith("…(截断,全文见规则文件)")  # 期望: 超限时追加截断标记
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert warnings, "超限必须告警,不得静默截断"  # 期望: 至少一条 WARNING 记录
    assert "尾部 2 行" in warnings[0].getMessage()  # 期望: 告警给出丢失行数=2
