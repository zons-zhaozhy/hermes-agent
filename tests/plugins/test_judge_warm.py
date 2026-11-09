"""并行点燃：同一轮里互不依赖的判定同时跑，只等一次。

`pre_llm_call` 每轮只触发一次，且是唯一能把文本注入本轮的缝——所以本轮要用的判定
必须在第一 token 之前就绪，没有「稍后补送」。能拿回来的只有时间：`user_side_guards`
（~2.5s）与 `read_think_intent`（~4s）判的是同一条用户消息、互不依赖，串行等于白等一次。

契约：
  1. `warm()` 立刻返回，判定在守护线程里跑，异常全吞；
  2. 同一调用在飞行中不重复点燃；
  3. 点燃之后同一判定再被同步索要时，**复用同一份结果**，不发第二次真实调用；
  4. read_think_gate 的 pre_llm_call 会点燃用户侧合并判定（即使本闸对该会话没启用）。

Preconditions: call_llm 被 monkeypatch，零真实调用。
"""

from __future__ import annotations

import importlib.util
import sys
import threading
import types
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]


class _Resp:
    def __init__(self, content: str) -> None:
        self.choices = [types.SimpleNamespace(message=types.SimpleNamespace(content=content))]


@pytest.fixture()
def judge(monkeypatch):
    import plugins._llm_judge as module
    import plugins._judge_warm as warm_module

    module.reset_verdict_cache()
    module.reset_judge_circuit()
    module.reset_judge_inflight()
    warm_module.reset()
    return module, warm_module


def test_warm_returns_at_once_and_swallows_errors(judge):
    _module, warm_module = judge
    started = threading.Event()

    def boom():
        started.set()
        raise RuntimeError("预热失败不该影响任何人")

    assert warm_module.warm(boom) is True
    assert started.wait(5.0), "后台线程必须真的跑起来"


def test_warm_does_not_double_start_the_same_call(judge):
    _module, warm_module = judge
    release = threading.Event()
    runs: list[int] = []

    def slow():
        runs.append(1)
        release.wait(5.0)

    warm_module.warm(slow)
    assert warm_module.warm(slow) is False, "同一调用在飞行中不得重复点燃"
    release.set()


def test_warmed_judge_is_reused_by_the_synchronous_caller(judge, monkeypatch):
    """预热点燃的那次真实调用，被随后同步索要的守卫复用（不是各判一次）。"""
    module, warm_module = judge
    started = threading.Event()
    release = threading.Event()
    calls: list[int] = []

    def slow(**kwargs):
        calls.append(1)
        started.set()
        release.wait(5.0)
        return _Resp('{"challenge": true}')

    monkeypatch.setattr("agent.auxiliary_client.call_llm", slow)

    warm_module.warm(
        module.llm_judge_multi,
        task="user_side_guards", system="return JSON", text="这不对吧", keys=["challenge"],
    )
    assert started.wait(5.0), "预热必须真的进入调用"

    result: list[dict] = []

    def consumer():
        result.append(module.llm_judge_multi(
            task="user_side_guards", system="return JSON", text="这不对吧", keys=["challenge"],
        ))

    consumer_thread = threading.Thread(target=consumer)
    consumer_thread.start()
    release.set()  # 无论消费者是等到了飞行中的结果还是直接命中缓存，真实调用都只能是 1 次
    consumer_thread.join(10)

    assert calls == [1], "预热与同步索要必须共用一次真实调用"
    assert result and result[0] == {"challenge": True}


def test_read_think_gate_warms_the_user_side_judge(monkeypatch):
    """read_think_gate 的 pre_llm_call 必须点燃用户侧合并判定，且不依赖本闸是否启用。"""
    warmed: list[str] = []

    fake_yinyang = types.ModuleType("hermes_plugins.yinyang_restate_guard")
    fake_yinyang.judge_user_side = lambda message: warmed.append(message)
    monkeypatch.setitem(sys.modules, "hermes_plugins.yinyang_restate_guard", fake_yinyang)

    real_warm: list[tuple] = []

    def fake_warm(fn, *args, **kwargs):
        real_warm.append((fn, args))
        return True

    import plugins._judge_warm as warm_module
    monkeypatch.setattr(warm_module, "warm", fake_warm)

    spec = importlib.util.spec_from_file_location(
        "read_think_gate_under_test", _REPO / "plugins" / "read_think_gate" / "__init__.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["read_think_gate_under_test"] = module
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "_current_gate", lambda *_a, **_k: None)  # 本闸对该会话不启用

    module.pre_llm_call(user_message="你这结论是不是搞错了？", session_id="s1")

    assert real_warm, "即使本闸未启用，也必须点燃其他守卫要用的判定"
    fn, args = real_warm[0]
    assert fn is fake_yinyang.judge_user_side
    assert args == ("你这结论是不是搞错了？",)
