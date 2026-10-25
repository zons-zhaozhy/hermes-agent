"""todo_debt 派发层与 prompt 收集链 E2E（债 4 收口）。

T4 回归只覆盖回调级（直调 on_session_open）。反方审查（2026-10-25）指出缺口：
「注册→派发→进 prompt 全链未验证」——回调绿只能宣称回调级验证。

本文件锁两层：
  D1 派发层：真实 invoke_hook('pre_llm_call', ...)（hermes_cli.plugins 全插件
     发现后路由到 finish_guard 注册的 on_session_open）返回的 results 含
     todo_debt context dict——注册与派发真实接线证明。
  D2 收集层：agent/turn_context._collect_pre_llm_call_context（生产消费入口，
     其返回串被拼入发往模型的用户消息）在预置账本 + temp HERMES_HOME 下返回
     含账单文本——注入文本真实到达 prompt 载荷收集边界。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from plugins.finish_guard import todo_debt


def _seed_home(home: Path, sid_owner: str) -> None:
    """temp HERMES_HOME 预置：启用 finish_guard（bundled 插件默认 disabled，须 config 显式列名）+ 他账账本。"""
    home.joinpath("config.yaml").write_text(
        "plugins:\n  enabled:\n    - finish_guard\n", encoding="utf-8")
    cache = home / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    ledger = {
        "sessions": {
            sid_owner: {
                "updated_at": 1_700_000_000,
                "items": [{"id": "7", "content": "E2E跨会话未清账项", "status": "pending"}],
            }
        }
    }
    (cache / "todo_debt.json").write_text(json.dumps(ledger, ensure_ascii=False), encoding="utf-8")


@pytest.mark.parametrize("mode", ["invoke_hook", "collect_context"])
def test_pre_llm_call_chain_delivers_debt_context(tmp_path, monkeypatch, mode):
    """D1/D2：账单文本经真实派发/收集链到达 prompt 载荷边界。"""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    sid_owner, sid_fresh = "e2e_owner", f"e2e_fresh_{mode}"
    _seed_home(home, sid_owner)
    if mode == "invoke_hook":
        from hermes_cli.plugins import invoke_hook

        results = invoke_hook(
            "pre_llm_call",
            session_id=sid_fresh,
            task_id=sid_fresh,
            turn_id="t1",
            user_message="继续干活",
            conversation_history=[{"role": "user", "content": "继续干活"}],
            is_first_turn=True,
            model="test-model",
            platform="cli",
            parent_session_id="",
            sender_id="",
        )
        contexts = [r["context"] for r in results if isinstance(r, dict) and r.get("context")]
        joined = "\n".join(contexts)
    else:
        from agent import turn_context

        agent = SimpleNamespace(
            session_id=sid_fresh,
            _persist_disabled=False,
            model="test-model",
            platform="cli",
            _parent_session_id="",
            _user_id="",
        )
        joined = turn_context._collect_pre_llm_call_context(
            agent,
            effective_task_id=sid_fresh,
            turn_id="t1",
            original_user_message="继续干活",
            messages=[{"role": "user", "content": "继续干活"}],
            conversation_history=[],
        )
    assert "todo_debt" in joined  # 期望: 账单提醒到达派发/收集层输出
    assert "E2E跨会话未清账项" in joined  # 期望: 预置账本条目内容在注入文本中
    assert sid_owner[:12] in joined  # 期望: 来源会话槽位可追溯


def test_collect_context_no_debt_is_empty(tmp_path, monkeypatch):
    """负例：无账本时收集层零注入（不产生噪声文本）。"""
    home = tmp_path / "home2"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    from agent import turn_context

    agent = SimpleNamespace(
        session_id="e2e_clean",
        _persist_disabled=False,
        model="test-model",
        platform="cli",
        _parent_session_id="",
        _user_id="",
    )
    out = turn_context._collect_pre_llm_call_context(
        agent,
        effective_task_id="e2e_clean",
        turn_id="t1",
        original_user_message="hi",
        messages=[{"role": "user", "content": "hi"}],
        conversation_history=[],
    )
    assert "todo_debt" not in out  # 期望: 无账零注入


def test_resume_session_also_injects(tmp_path, monkeypatch):
    """续开会话（历史含 assistant）进程内首见同样注入——is_first_turn 缺口回归锁。"""
    home = tmp_path / "home3"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    _seed_home(home, "e2e_owner")
    out = todo_debt.on_session_open(
        conversation_history=[
            {"role": "user", "content": "早"},
            {"role": "assistant", "content": "已开工"},
        ],
        session_id="e2e_resume_proc",
    )
    assert out is not None and "E2E跨会话未清账项" in out["context"]  # 期望: 续开首轮注入
