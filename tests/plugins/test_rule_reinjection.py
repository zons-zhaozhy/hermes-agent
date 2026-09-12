"""rule_reinjection 行为测试——期望值独立推导,不读源码。

行为契约(衰减残留处理三件套):
  A. 无规则文件 → {} 不注入
  B. 首注:首次调用注入,前缀含版本指纹+作废声明(冲突消解语义)
  C. 版本变更:规则文件内容变化 → 立即注入新版本
  D. 周期补注:内容不变时,每 _REINJECT_INTERVAL 轮补注一次
  E. 中间轮:首注与周期之间,同版本不重复注入(防历史堆积)
  F. 空文件/纯空白 → {} 不注入
  G. 摘要硬上限 1200 字符,超出截断并标注
  H. register 注册 pre_llm_call 钩子
"""

from __future__ import annotations

from plugins.discipline import rule_reinjection as rr


class _Ctx:
    def __init__(self):
        self.hooks = {}

    def register_hook(self, name, fn):
        self.hooks[name] = fn


def test_no_rules_file_no_injection(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert rr._on_pre_llm_call(session_id="s1") == {}


def test_first_injection_has_version_and_supersede(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".hermes-rules.md").write_text("规则一:写前必读目标文件", encoding="utf-8")
    out = rr._on_pre_llm_call(session_id="s2")
    assert "context" in out
    assert "唯一权威版本" in out["context"]  # 冲突消解声明
    assert "一律作废" in out["context"]
    assert "规则一:写前必读目标文件" in out["context"]


def test_version_change_reinjects_immediately(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    f = tmp_path / ".hermes-rules.md"
    f.write_text("v1-rule", encoding="utf-8")
    first = rr._on_pre_llm_call(session_id="s3")
    f.write_text("v2-rule 完全不同内容", encoding="utf-8")
    second = rr._on_pre_llm_call(session_id="s3")
    assert "v2-rule 完全不同内容" in second["context"]
    assert first["context"] != second["context"]


def test_periodic_reinjection_at_interval(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".hermes-rules.md").write_text("stable-rule", encoding="utf-8")
    rr._on_pre_llm_call(session_id="s4")  # 首注(call 1)
    injected = 0
    injected_ctx = ""
    for _ in range(rr._REINJECT_INTERVAL):  # calls 2..N+1,含周期轮 N
        out = rr._on_pre_llm_call(session_id="s4")
        if out:
            injected += 1
            injected_ctx = out["context"]
    assert injected >= 1  # 周期补注确实发生
    # 且周期补注的文本与首注同版本同内容(对抗衰减,不是新版本)
    assert "stable-rule" in injected_ctx


def test_no_redundant_injection_between_checkpoints(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".hermes-rules.md").write_text("stable-rule", encoding="utf-8")
    rr._on_pre_llm_call(session_id="s5")  # 首注
    # 紧随其后几轮(未到周期)不重复注入——防历史堆积
    for _ in range(3):
        assert rr._on_pre_llm_call(session_id="s5") == {}


def test_blank_rules_file_no_injection(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".hermes-rules.md").write_text("\n \n", encoding="utf-8")
    assert rr._on_pre_llm_call(session_id="s6") == {}


def test_digest_hard_cap(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".hermes-rules.md").write_text("字" * 5000, encoding="utf-8")
    out = rr._on_pre_llm_call(session_id="s7")
    assert "context" in out
    assert len(out["context"]) < 1500  # 摘要+前缀远小于原文
    assert "截断" in out["context"]


def test_register_hooks():
    ctx = _Ctx()
    rr.register(ctx)
    assert "pre_llm_call" in ctx.hooks
