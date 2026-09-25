"""completion-boundary-audit 跨轮消项闭环回归测试。

背景（2026-09-25 用户实录）：AI 每轮回复都披露未验证边界，但披露完就
撂挑子——下一轮把同一批边界原样再抄一遍，从不逐项消。根因：防线只有
「披露义务」没有「闭环义务」——judge 判据是「完成声明且未披露」才标记，
披露了就放行，披露成了免罪金牌。

根治：跨轮对账状态机——披露了边界 → 记 pending；下轮注入消项红牌
（能验证的当轮贴原始输出，外部依赖项写依赖方+等待条件）；同批 pending
连续 ≥2 轮未消 → 注入升级语；回复不再含边界且含完成声明 → 视为已消项。

Contract:
  Preconditions: 独立会话 state；judge 全部 monkeypatch（零真实 LLM 调用）
  Postconditions: 全部断言通过 = 状态机各转移正确、fail-open 不动状态、
                  旧「补披露」路径不回归
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PLUGIN_DIR = _REPO_ROOT / "plugins" / "completion-boundary-audit"


def _load_plugin():
    """按 PluginManager 命名约定加载插件 __init__.py。"""
    if "hermes_plugins" not in sys.modules:
        ns = types.ModuleType("hermes_plugins")
        ns.__path__ = []
        sys.modules["hermes_plugins"] = ns
    spec = importlib.util.spec_from_file_location(
        "hermes_plugins.completion_boundary_audit",
        _PLUGIN_DIR / "__init__.py",
        submodule_search_locations=[str(_PLUGIN_DIR)],
    )
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "hermes_plugins.completion_boundary_audit"
    mod.__path__ = [str(_PLUGIN_DIR)]
    sys.modules["hermes_plugins.completion_boundary_audit"] = mod
    spec.loader.exec_module(mod)
    return mod


class _FakeCtx:
    """收集 register_hook 注册的钩子，供测试直接驱动。"""

    def __init__(self):
        self.hooks = {}

    def register_hook(self, name, fn):
        self.hooks[name] = fn


def _mk(mod, judge_result, sid):
    """装配插件：monkeypatch judge 返回固定结果，返回 (audit, inject, state)。"""
    ctx = _FakeCtx()
    mod.register(ctx)
    mod._judge_reply_side = lambda text: judge_result  # noqa: SLF001 测试桩
    mod.llm_judge_bool = lambda **kw: None  # noqa: SLF001 独立回退通道也不发真调用
    mod.llm_judge_multi = lambda **kw: {k: None for k in kw["keys"]}  # noqa: SLF001
    audit = ctx.hooks["transform_llm_output"]
    inject = ctx.hooks["pre_llm_call"]
    state = mod._state(sid)  # noqa: SLF001
    state.clear()
    return audit, inject, state


_LONG_TEXT = "修复已完成并全部提交。" + "细节说明" * 40  # 超过 _MIN_LENGTH
_LONG_TEXT2 = "部署已完成，全部通过。" + "补充说明" * 40  # 第 2 轮不同措辞（同文会被去重跳过）


@pytest.fixture(autouse=True)
def _clean_state():
    yield
    from plugins._shared_state import clear_session
    clear_session("cba-test")


def test_disclosed_boundary_sets_pending():
    """披露了边界 + 完成声明 → 记 pending（下一轮注入消项红牌）。

    期望：旧代码（披露即放行）不记 pending → 本用例在旧代码上跑红。
    """
    mod = _load_plugin()
    audit, inject, state = _mk(
        mod,
        {"uncertain": False, "needs_audit": False,
         "done_claim": True, "has_boundary": True},
        "cba-test",
    )
    assert audit(_LONG_TEXT, session_id="cba-test") is None  # 恒不改用户回复
    assert state.get("pending_reminder") is True  # 期望: 披露≠豁免，记 pending
    msg = inject(session_id="cba-test")
    assert msg is not None and "消" in msg["context"]  # 期望: 注入消项红牌


def test_stale_escalates():
    """同批边界连续 2 轮未消 → 注入升级语。"""
    mod = _load_plugin()
    judge = {"uncertain": False, "needs_audit": False,
             "done_claim": True, "has_boundary": True}
    audit, inject, state = _mk(mod, judge, "cba-test")
    audit(_LONG_TEXT, session_id="cba-test")  # 第 1 轮
    inject(session_id="cba-test")
    audit(_LONG_TEXT2, session_id="cba-test")  # 第 2 轮：换了措辞仍原样披露未消项
    msg = inject(session_id="cba-test")
    assert msg is not None
    # 期望: stale>=2 时升级语点名「连续未消」
    assert ("连续" in msg["context"]) or ("撂挑子" in msg["context"])


def test_resolved_boundary_clears_pending():
    """红牌后回复不再列边界且含完成声明 → 视为已消项，清状态。"""
    mod = _load_plugin()
    audit, inject, state = _mk(
        mod,
        {"uncertain": False, "needs_audit": False,
         "done_claim": True, "has_boundary": True},
        "cba-test",
    )
    audit(_LONG_TEXT, session_id="cba-test")
    # 下一轮：全部消项，回复只报结果无边界清单（换措辞避开同文去重）
    mod._judge_reply_side = lambda text: {  # noqa: SLF001
        "uncertain": False, "needs_audit": False,
        "done_claim": True, "has_boundary": False}
    audit(_LONG_TEXT2, session_id="cba-test")
    assert not state.get("pending_reminder")  # 期望: 已消项清 pending
    assert inject(session_id="cba-test") is None  # 期望: 不再注入


def test_undisclosed_path_unchanged():
    """旧路径（完成声明+无披露→补披露红牌）不回归。"""
    mod = _load_plugin()
    audit, inject, state = _mk(
        mod,
        {"uncertain": False, "needs_audit": True,
         "done_claim": True, "has_boundary": False},
        "cba-test",
    )
    audit(_LONG_TEXT, session_id="cba-test")
    assert state.get("pending_reminder") is True
    msg = inject(session_id="cba-test")
    assert msg is not None and "披露" in msg["context"]  # 期望: 仍是补披露红牌


def test_judge_fail_open_leaves_state():
    """judge 返回 None（fail-open）→ 状态零改动。"""
    mod = _load_plugin()
    audit, inject, state = _mk(mod, None, "cba-test")
    audit(_LONG_TEXT, session_id="cba-test")
    assert not state.get("pending_reminder")  # 期望: fail-open 不误伤
    assert inject(session_id="cba-test") is None


def test_no_done_claim_no_pending():
    """无完成声明的进度汇报（列了待办边界）→ 不记 pending。"""
    mod = _load_plugin()
    audit, inject, state = _mk(
        mod,
        {"uncertain": False, "needs_audit": False,
         "done_claim": False, "has_boundary": True},
        "cba-test",
    )
    audit(_LONG_TEXT, session_id="cba-test")
    assert not state.get("pending_reminder")  # 期望: 进行中汇报不打扰
