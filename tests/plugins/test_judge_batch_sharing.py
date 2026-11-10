"""合并 judge 通道的行为契约（批量判定：用户侧 3 键 + 回复侧 7 键）。

每回合的同步 judge 从 6 次串行调用收敛为 2 次：
  用户侧：challenge/decision/authorized（yinyang + devil-advocate + scope_gate 共享）
  回复侧：uncertain/needs_audit/done_claim/has_boundary/violation/naming_drift/jargon_coined
          （reply-certainty + completion-boundary + declare-act + terminology 共享）

契约：
  1. 同一条文本进程内只发起一次 llm_judge_multi（缓存命中不再调用）；
  2. 用户侧键集含 authorized、回复侧键集含新增的 violation/naming_drift/jargon_coined；
  3. 各消费者读合并结果的自己的键；
  4. 合并通道不可用（插件未加载/导入失败）时回退各自独立 judge——fail-open 语义不变。

Preconditions: 全部 judge 均被 monkeypatch（零真实 LLM 调用）。
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _clean_modules():
    yield
    for name in ("hermes_plugins.reply_certainty_checker", "hermes_plugins.yinyang_restate_guard"):
        sys.modules.pop(name, None)


def _load(name: str, rel: str):
    """按 PluginManager 命名约定加载插件模块（连字符目录 → 下划线模块名）。"""
    if "hermes_plugins" not in sys.modules:
        ns = types.ModuleType("hermes_plugins")
        ns.__path__ = []
        sys.modules["hermes_plugins"] = ns
    path = _REPO / rel
    spec = importlib.util.spec_from_file_location(f"hermes_plugins.{name}", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = f"hermes_plugins.{name}"
    sys.modules[f"hermes_plugins.{name}"] = mod
    spec.loader.exec_module(mod)
    return mod


def _stub_shared(sibling: str, **verdicts):
    """把一个共享判定模块挂进 sys.modules（生产运行时由插件加载器注册）。"""
    stub = types.ModuleType(f"hermes_plugins.{sibling}")
    stub.judge_reply_side = lambda _text: dict(verdicts)
    stub.judge_user_side = lambda _text: dict(verdicts)
    sys.modules[f"hermes_plugins.{sibling}"] = stub
    return stub


def _no_judge(*_args, **_kwargs):
    raise AssertionError("合并通道可用时不应回退独立 judge")


def test_user_side_single_call_carries_authorized(monkeypatch):
    """challenge/decision/authorized 一次调用；同一消息缓存命中不再调用。"""
    mod = _load("yinyang_restate_guard", "plugins/yinyang-restate-guard/__init__.py")
    calls: list[list[str]] = []

    def fake_multi(task, system, text, keys, **kw):
        calls.append(list(keys))
        return {k: True for k in keys}

    monkeypatch.setattr(mod, "llm_judge_multi", fake_multi)
    mod._USER_SIDE_CACHE.clear()
    message = "你这个结论不对，凭什么说部署已经完成？"
    first = mod.judge_user_side(message)
    second = mod.judge_user_side(message)

    assert calls == [["challenge", "decision", "authorized"]]  # 同文本只判一次
    assert first == second and first["authorized"] is True


def test_reply_side_single_call_carries_seven_keys(monkeypatch):
    """七个回复维度一次调用（输出预算随键数放宽）；同一回复缓存命中不再调用。"""
    mod = _load("reply_certainty_checker", "plugins/reply-certainty-checker/__init__.py")
    calls: list[list[str]] = []

    def fake_multi(task, system, text, keys, **kw):
        calls.append(list(keys))
        assert kw.get("max_tokens") == 200  # 七键 JSON 需要比默认 64 更宽的预算
        return {k: False for k in keys}

    monkeypatch.setattr(mod, "llm_judge_multi", fake_multi)
    mod._REPLY_SIDE_CACHE.clear()
    reply = "本次修复已完成并全部通过，未经测试的边界未单独验证。" * 2
    mod.judge_reply_side(reply)
    mod.judge_reply_side(reply)

    assert len(calls) == 1
    assert calls[0] == list(mod.REPLY_SIDE_KEYS)
    assert {"uncertain", "needs_audit", "done_claim", "has_boundary",
            "violation", "naming_drift", "jargon_coined"} <= set(calls[0])


def test_declare_act_reads_shared_violation(monkeypatch):
    """declare-act 的「宣告未动手」判定来自共享回复侧判定的 violation 键。"""
    mod = _load("declare_act_guard", "plugins/discipline/declare_act_guard.py")
    _stub_shared("reply_certainty_checker", violation=True)
    monkeypatch.setattr("plugins._llm_judge.llm_judge_bool", _no_judge)

    assert mod._judge_violation("我建议实施这个方案，接下来就做。") is True


def test_terminology_reads_shared_drift_keys(monkeypatch):
    """术语漂移判定来自共享回复侧判定的 naming_drift/jargon_coined 键。"""
    mod = _load("terminology_guard", "plugins/terminology-guard/__init__.py")
    _stub_shared("reply_certainty_checker", naming_drift=True, jargon_coined=False)
    monkeypatch.setattr("plugins._llm_judge.llm_judge_multi", _no_judge)

    got = mod._judge_output_quality("这一段足够长的回复文本用于触发判定路径。" * 2)
    assert got == {"naming_drift": True, "jargon_coined": False}


def test_scope_gate_reads_shared_authorization(monkeypatch):
    """新建产物的授权判定来自共享用户侧判定的 authorized 键。"""
    from plugins.guards import scope_gate

    _stub_shared("yinyang_restate_guard", authorized=False)
    monkeypatch.setattr("plugins._llm_judge.llm_judge_bool", _no_judge)

    assert scope_gate._judge_authorized("帮我把这个模块重构一下") is False


def test_scope_gate_falls_back_when_shared_unavailable(monkeypatch):
    """共享通道不可用（插件未加载）时回退独立 judge，判定语义不变。"""
    from plugins.guards import scope_gate

    sys.modules.pop("hermes_plugins.yinyang_restate_guard", None)
    monkeypatch.setattr("plugins._llm_judge.llm_judge_bool", lambda **kw: True)

    assert scope_gate._judge_authorized("实现一个全新的应用") is True
