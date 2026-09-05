# -*- coding: utf-8 -*-
"""finish_guard v2.1 宣言式假继续检测——用 0902 真实失效样本回归。

期望值独立推导（不依实现反推）：
- 事发原文尾部「继续 dag_router 12 端点批3。」= 宣布继续却停轮 → 必须命中
- 合法完工收尾（「全部完成」「如需」边界话术）→ 必须放行
- 合法请示（拍板/付费）→ 必须放行（_LEGIT_ASK_RE 通道）
- pre_verify 命中宣言式 → 必须返回 {"action": "continue"}（核心续跑契约）
"""

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "finish_guard", str(Path(__file__).resolve().parents[2] / "plugins" / "finish_guard" / "__init__.py")
)
assert _SPEC is not None and _SPEC.loader is not None, "finish_guard 插件文件不存在"
fg = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fg)


# ---- _is_false_continue：真实失效样本 ----

def test_incident_sample_declares_continue():
    """0902 事发原文：宣言式继续尾巴必须命中。"""
    tail = "进度节奏正常，继续 dag_router 12 端点批3。"
    assert fg._is_false_continue(tail) is True


def test_next_batch_phrase():
    assert fg._is_false_continue("loom 剩余弱型分布已列，接下来处理 delivery_router。") is True


def test_done_closure_passes():
    """合法完工收尾必须放行。"""
    assert fg._is_false_continue("loom P1 契约化收官，全部已推送。") is False


def test_conditional_boundary_passes():
    """「如需/待拍板」边界话术放行。"""
    assert fg._is_false_continue("loom P1 已全部完成并推送；如需调整判型请告知。") is False


def test_legit_ask_passes():
    """合法请示（付费）放行。"""
    assert fg._is_false_continue("下一步需要采购扩容，是否付费？") is False


def test_empty_text():
    assert fg._is_false_continue("") is False


# ---- pre_verify 通道契约 ----

def test_pre_verify_continues_on_declared_continue():
    res = fg._on_pre_verify(
        attempt=0,
        final_response="dbchat 已清零。进度正常，继续 loom P1 批3。",
        changed_paths=[],
    )
    assert res.get("action") == "continue"


def test_pre_verify_passes_clean_closure():
    res = fg._on_pre_verify(
        attempt=0,
        final_response="loom P1 契约化全部完成并推送，任务收官。",
        changed_paths=["a.py"],
    )
    assert res == {}


def test_pre_verify_attempt_cap():
    res = fg._on_pre_verify(
        attempt=3,
        final_response="进度节奏正常，继续下一批。",
        changed_paths=[],
    )
    assert res == {}


def test_pre_verify_zero_edit_still_consulted():
    """纯总结回合（changed_paths 空）也必须能触发续跑——0902 根因2 回归。"""
    res = fg._on_pre_verify(
        attempt=0,
        final_response="批2 闭环。继续 dag_router 批3。",
        changed_paths=[],
    )
    assert res.get("action") == "continue"


# ---- L3 可见标记 ----

def test_l3_marks_declared_continue():
    out = fg._transform_llm_output(response_text="全部清零。接下来继续 loom。")
    assert "[finish_guard]" in out
