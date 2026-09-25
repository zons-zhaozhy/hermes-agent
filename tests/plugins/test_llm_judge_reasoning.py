"""llm_judge_multi 关思考纪律回归测试。

背景（2026-09-25 用户点名）：内存先例「judge挂死=qwen3.5缺reasoning_effort」
——logprobs 通道（_logprob_verdict）修了（extra_body 显式带
reasoning_effort="none"），但 multi 键文本通道（llm_judge_multi）没修：
裸跑本地思考型模型（qwen3.5:4b-mlx）默认开思考，超时前思考块耗尽预算，
判定 fail-open，整条防线空转（errors.log 实测 reply_side_guards 三次
20s 超时）。

根治：multi 键通道与 logprobs 通道同款自持——调用层显式带
extra_body={"reasoning_effort": "none"}，不依赖 config 的
auxiliary.<task>.extra_body 条目存在（call_llm 会 merge，显式传入
覆盖 task config 同键，语义一致但代码级自持）。

Contract:
  Preconditions: _llm_judge 以 hermes_plugins 命名空间加载；call_llm 全部
                 monkeypatch（零网络）
  Postconditions: 全部断言通过 = multi/bool 文本通道均显式关思考，
                  既有解析语义不变
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_JUDGE_FILE = _REPO_ROOT / "plugins" / "_llm_judge.py"


def _load_judge() -> "types.ModuleType":
    """按 PluginManager 命名约定加载 _llm_judge.py。"""
    if "hermes_plugins" not in sys.modules:
        ns = types.ModuleType("hermes_plugins")
        ns.__path__ = []
        sys.modules["hermes_plugins"] = ns
    spec = importlib.util.spec_from_file_location(
        "hermes_plugins._llm_judge", _JUDGE_FILE)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "hermes_plugins"
    sys.modules["hermes_plugins._llm_judge"] = mod
    spec.loader.exec_module(mod)
    return mod


class _Resp:
    """最小 chat 响应桩。"""

    def __init__(self, content: str) -> None:
        self.choices = [types.SimpleNamespace(
            message=types.SimpleNamespace(content=content))]


@pytest.fixture(autouse=True)
def captured(monkeypatch: pytest.MonkeyPatch) -> dict:
    """拦截 agent.auxiliary_client 导入点：call_llm 替换为记录桩（零网络）。"""
    captured: dict = {}

    def fake_call_llm(**kwargs: object) -> "_Resp":
        captured.update(kwargs)
        return _Resp('{"uncertain": false, "needs_audit": false, '
                     '"done_claim": true, "has_boundary": true}')

    fake_mod = types.ModuleType("agent.auxiliary_client")
    fake_mod.call_llm = fake_call_llm
    monkeypatch.setitem(sys.modules, "agent.auxiliary_client", fake_mod)
    return captured


def test_multi_sends_reasoning_none(captured: dict) -> None:
    """multi 键通道必须显式带 reasoning_effort=none（关思考自持）。

    期望：旧代码（不传 extra_body）本用例红。
    """
    mod = _load_judge()
    out = mod.llm_judge_multi(
        task="reply_side_guards",
        system="只回答一个 JSON 对象，含全部键",
        text="修复已完成，全部通过。",
        keys=["uncertain", "needs_audit", "done_claim", "has_boundary"],
    )
    eb = captured.get("extra_body") or {}
    assert eb.get("reasoning_effort") == "none"  # 期望: 显式关思考——判定件不需要思考，思考型模型裸跑必超时
    assert captured["task"] == "reply_side_guards"  # 期望: task 原样透传，路由不受影响
    assert out == {"uncertain": False, "needs_audit": False, "done_claim": True, "has_boundary": True}  # 期望: 桩返回的四键原样解析（两 True 两 False）


def test_multi_caller_extra_body_preserved(captured: dict) -> None:
    """调用方自带 extra_body 时：reasoning_effort 合并进去不覆盖其他键。"""
    mod = _load_judge()
    mod.llm_judge_multi(
        task="reply_side_guards",
        system="只回答一个 JSON 对象",
        text="修复已完成。",
        keys=["a"],
        extra_body={"temperature": 0},
    )
    eb = captured.get("extra_body") or {}
    assert eb.get("reasoning_effort") == "none" and eb.get("temperature") == 0  # 期望: 合并非替换——关思考键补入且调用方 temperature 保留


def test_bool_text_channel_sends_reasoning_none(captured: dict) -> None:
    """llm_judge_bool 文本通道同样显式关思考（同缺陷类一次修完）。"""
    mod = _load_judge()
    mod.llm_judge_bool(
        task="completion_boundary_audit",
        system="只回答 JSON",
        text="修复已完成并全部提交。" + "x" * 80,
    )
    eb = captured.get("extra_body") or {}
    assert eb.get("reasoning_effort") == "none"  # 期望: bool 文本通道与 logprobs 通道同款自持


def test_no_network_no_raise_on_import_error(
        monkeypatch: pytest.MonkeyPatch, captured: dict) -> None:
    """call_llm 抛异常 → 全键 None fail-open，不 raise（既有语义）。"""
    mod = _load_judge()

    def boom(**kwargs: object) -> "_Resp":
        raise ImportError("agent.auxiliary_client unavailable")

    monkeypatch.setitem(sys.modules, "agent.auxiliary_client",
                        types.SimpleNamespace(call_llm=boom))
    out = mod.llm_judge_multi(task="t", system="s", text="x" * 20, keys=["k"])
    assert out == {"k": None}  # 期望: fail-open 全 None——判定件失败不外泄异常
