"""User-facing output stays the model's answer; plugin health stays in diagnostics.

Regression shape: a plain status answer was hidden twice by ``finish_guard`` and the
published rewrite carried audit/guard instructions.  Zero-edit status turns must not be
re-continued merely because they mention a decision boundary or “root fix”, and output
transforms must return ``None`` rather than appending plugin diagnostics.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from plugins import finish_guard as fg
from plugins import guard_sentinel
from plugins.finish_guard import completion_boundary_audit as completion


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
jspace = importlib.import_module("plugins.discipline.jspace_guard")
persistence = importlib.import_module("plugins.persistence-enforcer")


_PLAIN_STATUS_ANSWER = """
一句话：都处理完了，今天又实测了一遍，没反弹。

- 奇安信报告里的高危和超危，后端 110 条全部闭环（升级版本、删组件）。
- 前端今天刚跑的审计：严重 0、高危 0，剩 9 个中危 5 个低危。

还剩 3 件事不是“没修完”，是修不动、要单独立项：替换 OFD 组件才能根治；
Spring 要跨大版本；webpack 4 只影响构建机。要你拍板是否立项。
"""


def test_zero_edit_status_answer_is_not_continued(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    verdict = fg._on_pre_verify(
        session_id="plain-status",
        attempt=0,
        final_response=_PLAIN_STATUS_ANSWER,
        changed_paths=[],
    )
    assert verdict == {}


def test_finish_guard_does_not_append_its_marker_to_the_answer():
    assert fg._transform_llm_output(response_text="问题已彻底根治。") is None


def test_completion_audit_unavailable_never_rewrites_or_nudges(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    class _Context:
        def __init__(self):
            self.hooks = {}

        def register_hook(self, name, callback):
            self.hooks[name] = callback

    ctx = _Context()
    completion.register(ctx)
    completion._judge_reply_side = lambda _text: {
        "done_claim": None, "has_boundary": None,
    }
    completion.llm_judge_bool = lambda **_kwargs: None
    completion.llm_judge_multi = lambda **kwargs: {key: None for key in kwargs["keys"]}
    completion._state("audit-down").clear()
    response = "修复已完成并全部提交。" + "细节说明" * 40

    assert ctx.hooks["transform_llm_output"](response, session_id="audit-down") is None
    assert ctx.hooks["pre_llm_call"](session_id="audit-down") is None


def test_guard_sentinel_diagnostic_never_rewrites_answer():
    session_id = "sentinel-diagnostic"
    guard_sentinel._pending[session_id] = "\n\n[门禁缺席 / UNKNOWN] internal-only"

    assert guard_sentinel.transform_llm_output(
        "normal answer", session_id=session_id,
    ) is None


def test_jspace_warnings_never_rewrite_answer():
    kwargs = {"session_id": "jspace-diagnostic"}
    state = jspace._state_for(kwargs)
    state.tool_call_count = jspace.ACTIVATION_THRESHOLD
    state.current_turn_tools = set()

    assert jspace._on_transform_llm_output(
        session_id="jspace-diagnostic",
        response_text="测试通过了，功能正常，已经修复了这个 bug。",
    ) is None


def test_persistence_reminder_never_rewrites_answer():
    session_id = "persistence-diagnostic"
    persistence._states.clear()
    for _ in range(persistence.WARN_THRESHOLD):
        persistence._on_post_tool_call(
            session_id=session_id, tool_name="read_file", status="ok"
        )
    response = (
        "# 审计结论\n[实测] 5 次调用。\n\n## 证据\n[文档] schema 已确认。\n"
        "[推断] 阈值行为稳定。\n" + "详细说明。\n" * 60
    )

    assert persistence._on_transform_llm_output(
        session_id=session_id, response_text=response,
    ) is None

