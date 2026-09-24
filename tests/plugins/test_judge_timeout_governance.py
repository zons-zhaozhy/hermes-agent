"""Judge 插件预算治理不变量。

背景（2026-09-24 根因）：call_llm 的 "Explicit timeout wins, else config" 机制下，
插件源码里写死 timeout= 秒数会永久屏蔽 auxiliary.<task>.timeout 配置面——
用户在 config.yaml 改超时永不生效，每轮烧固定墙钟。本测试固化两条行为契约：
1. 源面：judge 插件源码不得再出现硬编码 timeout= 实参（AST 级检查，重构安全）。
2. 行为面：call_llm 不传 timeout 时，实际预算必须来自 auxiliary.<task>.timeout。

注意：AST 检查的是"插件源码形状"，属于行为契约（call 签名与配置面的关系），
非快照断言。agent/__init__.py 已核（仅 jiter_preload re-export，无符号遮蔽）。
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from agent.auxiliary_client import _effective_aux_timeout  # agent 为裸包(仅 jiter preload)

PLUGINS_DIR = Path(__file__).resolve().parents[2] / "plugins"

# 全部经 _llm_judge 发起 aux 调用的插件（llm_judge_bool / llm_judge_multi 消费方）。
_JUDGE_PLUGIN_SOURCES = sorted(
    str(p) for p in PLUGINS_DIR.glob("*/__init__.py")
    if "llm_judge" in p.read_text(encoding="utf-8")
)


def test_no_hardcoded_timeout_in_judge_plugin_calls() -> None:
    """契约：judge 插件对 _llm_judge 的调用不得自带 timeout= 实参。

    预算的唯一治理面是 auxiliary.<task>.timeout（config.yaml）。插件再写死秒数
    会重新屏蔽配置面（call_llm Explicit timeout wins）。
    """
    offenders: list[str] = []
    for src in _JUDGE_PLUGIN_SOURCES:
        tree = ast.parse(Path(src).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                called = getattr(node.func, "id", "") or getattr(node.func, "attr", "")
                if called in {"llm_judge_bool", "llm_judge_multi", "llm_decision_typed"}:
                    for kw in node.keywords:
                        if kw.arg == "timeout":
                            offenders.append(f"{src}:{node.lineno}")
    assert not offenders, (  # 期望: 空——六插件硬编码已在 ef562cb74a 全数移除,任何新 timeout= 实参都是回归
        f"judge 插件硬编码 timeout 会屏蔽 auxiliary.<task>.timeout 配置面: {offenders}"
    )


def test_call_llm_without_timeout_uses_task_config_budget(monkeypatch) -> None:
    """契约：call_llm(timeout=None) 时解析到的预算 == auxiliary.<task>.timeout。

    _effective_aux_timeout 语义（auxiliary_client.py L6227-6233）:显式 timeout 优先,
    否则取 _get_task_timeout(task)。本测试 monkeypatch 后者恒返回 7.0,期望值由该
    函数契约独立推导（无显式参→透传配置值;有显式参→显式胜出）,非实现反推。
    """
    import agent.auxiliary_client as aux

    monkeypatch.setattr(aux, "_get_task_timeout", lambda task, default=30.0: 7.0)
    assert _effective_aux_timeout("some_task", None) == 7.0  # 期望: 7.0 — patch 后配置面唯一来源
    # 显式 timeout 仍胜出（插件需要临时覆盖的合法通道保留）
    assert _effective_aux_timeout("some_task", 3.0) == 3.0  # 期望: 3.0 — Explicit wins 契约
