"""通道对称性契约——LLM 意图判定与 fallback 共享同一拒绝不对称原理。

2026-11-12 钉钉群实测：用户追问「这个是咋回事？为啥没处理？」被 LLM 通道判
read_only，代理随后的 write_file 连续被拒（round 1/5，每轮一次完整模型往返），
形成只有拒绝没有出路的空转。fallback 层的设计原则是「不该受的拒绝死锁整轮，
不该给的授予只花一次调查」（``_fallback_detect_intent`` docstring），但该不对称
约束此前只施加在 fallback 上，LLM 通道的 read_only 拒绝面反而更大。本文件把同
一原理钉在三条通道上：

1. LLM 只读判定对「运营追问」（追问处理状态/现象/进度）降级 ambiguous——写仍需
   调查契约解锁，但不再整轮死锁；
2. 会话级修改授权不被一条只读追问清零（显式禁止除外）；
3. 只读拒绝的第一轮直接引导 clarify 表单授权，不再文字空转。
"""
import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _load_gate() -> ModuleType:
    repo = Path(__file__).resolve().parents[2]
    path = repo / "plugins" / "read_think_gate" / "gate.py"
    spec = importlib.util.spec_from_file_location("read_think_gate_asymmetry_test", path)
    assert spec is not None and spec.loader is not None  # 期望: 插件文件可定位
    mod = importlib.util.module_from_spec(spec)
    sys.modules["read_think_gate_asymmetry_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_llm_read_only_downgrades_operational_followup(monkeypatch) -> None:
    """运营追问不许被 LLM 通道判成 read_only——那是死锁，不是拒绝。"""
    gate = _load_gate()
    monkeypatch.setattr(
        gate, "_classify_intent_via_llm", lambda *a, **k: gate.INTENT_READ_ONLY
    )
    verdict = gate._detect_user_intent("这个是咋回事？为啥没处理？", None, use_llm=True)
    assert verdict == gate.INTENT_AMBIGUOUS, verdict  # 期望: 降级 ambiguous 走调查契约


def test_llm_read_only_keeps_pure_information_question(monkeypatch) -> None:
    """纯信息问句的 read_only 判定保持不变——门禁的本职不能丢。"""
    gate = _load_gate()
    monkeypatch.setattr(
        gate, "_classify_intent_via_llm", lambda *a, **k: gate.INTENT_READ_ONLY
    )
    verdict = gate._detect_user_intent("为什么这个函数返回 None", None, use_llm=True)
    assert verdict == gate.INTENT_READ_ONLY, verdict


def test_explicit_prohibition_wins_over_llm(monkeypatch) -> None:
    """显式禁止是硬证据，语义层说 modify 也不能翻。"""
    gate = _load_gate()
    monkeypatch.setattr(
        gate, "_classify_intent_via_llm", lambda *a, **k: gate.INTENT_MODIFY
    )
    verdict = gate._detect_user_intent("别改代码，只分析", None, use_llm=True)
    assert verdict == gate.INTENT_READ_ONLY, verdict


def _make_gate(monkeypatch, intent: str):
    gate_mod = _load_gate()
    monkeypatch.setattr(gate_mod, "_detect_user_intent", lambda *a, **k: intent)
    cfg = gate_mod.ReadThinkGateConfig(
        complexity_adaptive=False, use_llm_classifier=False, use_llm_intent_classifier=True
    )
    instance = gate_mod.ReadThinkGate(config=cfg, session_id="")
    return gate_mod, instance


def test_session_authorization_survives_read_only_followup(monkeypatch) -> None:
    """已授权会话里的一条只读追问不清零授权——降级 ambiguous，写走调查契约。"""
    gate_mod, instance = _make_gate(monkeypatch, "read_only")
    instance._session_modification_authorized = True
    instance._authorized_user_message = "请修复网关断连问题"
    instance.reset_for_turn("为什么这么慢？", internal=False)
    assert instance._session_modification_authorized is True  # 期望: 授权保留
    assert instance._intent == gate_mod.INTENT_AMBIGUOUS, instance._intent  # 期望: 降级而非死锁


def test_explicit_prohibition_still_revokes_authorization(monkeypatch) -> None:
    """显式禁止仍然当场收回授权——撤销权不受授权保留影响。"""
    gate_mod, instance = _make_gate(monkeypatch, "read_only")
    instance._session_modification_authorized = True
    instance._authorized_user_message = "请修复网关断连问题"
    instance.reset_for_turn("别改了，只汇报", internal=False)
    assert instance._session_modification_authorized is False  # 期望: 显式禁止收回授权
    assert instance._intent == gate_mod.INTENT_READ_ONLY, instance._intent


def test_first_read_only_refusal_directs_clarify(monkeypatch) -> None:
    """首轮只读拒绝必须引导 clarify 表单，砍掉多轮文字空转。"""
    gate_mod, instance = _make_gate(monkeypatch, "read_only")
    instance.reset_for_turn("这是怎么回事？", internal=False)
    instance._intent = gate_mod.INTENT_READ_ONLY
    message = instance._refuse_read_only_intent(["write_file"], "请求写文档", 6)
    assert "clarify" in message  # 期望: 首拒即转表单授权
    assert "立即" in message  # 期望: 明确要求当轮发起，不是留一句提示
    assert instance._awaiting_user_authorization is True
