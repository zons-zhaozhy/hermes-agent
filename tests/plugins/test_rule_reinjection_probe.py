"""rule_reinjection 规则源探测与注入节奏不变量。"""

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


def test_probe_does_not_fallback_to_hermes_code_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """cwd 在仓库外(如 ~/.hermes):不得把 Hermes 仓库规则注入普通会话。

    Contract:
      Preconditions: cwd 被切到无 .git、无 .hermes-rules.md 的临时目录。
      Postconditions: 返回 None；规则只属于当前 cwd 或其 git root。
    """
    # 期望: 代码根候选必须移除,否则 OntoX/AML/普通网关会话都会收到 Hermes fork 规则
    outside = tmp_path / "hermes-home-like"
    outside.mkdir()
    monkeypatch.chdir(outside)
    assert _find_rules_file() is None


def test_probe_returns_none_when_no_rules_anywhere(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """候选全空(仓库外 cwd + 无项目规则文件):返回 None,由调用方告警。

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


def test_reinjection_repeats_only_when_version_changes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """首轮登记后，未变化的规则不得每 10 轮再次全文注入。"""
    import plugins.discipline.rule_reinjection as rr

    rules = tmp_path / ".hermes-rules.md"
    rules.write_text("# 项目规则\n- 先读现状\n", encoding="utf-8")
    monkeypatch.setattr(rr, "_find_rules_file", lambda: rules)
    session_id = "rule-version-only"
    rr._session_state.pop(session_id, None)
    try:
        first = rr._on_pre_llm_call(session_id=session_id)
        assert "先读现状" in first["context"]
        for _ in range(35):
            assert rr._on_pre_llm_call(session_id=session_id) == {}

        rules.write_text("# 项目规则\n- 规则已更新\n", encoding="utf-8")
        changed = rr._on_pre_llm_call(session_id=session_id)
        assert "规则已更新" in changed["context"]
    finally:
        rr._session_state.pop(session_id, None)


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
