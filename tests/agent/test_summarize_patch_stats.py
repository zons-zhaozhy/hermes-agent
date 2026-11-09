"""Invariant: the patch tool's compression summary keeps the net change.

After mid-conversation compression the diff is gone from context; the summary
line is the only place the post-compaction agent can read "what did the patch
change" from. It must carry +N/-M line counts on success, the FAILED suffix on
error, and never raise on hostile payload shapes (the outer
_summarize_tool_result guard would silently degrade to a content-length stub).
"""
import json

from agent.context_compressor import _summarize_tool_result


def _patch_args(path="x.py"):
    return json.dumps({"path": path, "old_string": "a", "new_string": "b"})


def test_success_summary_carries_line_counts():
    diff = "--- a/x.py\n+++ b/x.py\n@@ -1,1 +1,2 @@\n-old\n+new1\n+new2"
    result = _summarize_tool_result("patch", _patch_args(), json.dumps({"success": True, "diff": diff}))
    # 期望: 2 additions / 1 removal, header lines excluded
    assert "[patch] replace in x.py (+2/-1 lines)" == result


def test_error_summary_keeps_failure_suffix():
    payload = json.dumps({"success": False, "error": "old_string not found"})
    result = _summarize_tool_result("patch", _patch_args("y.py"), payload)
    # 期望: 失败结果不携带行数统计, 且失败后缀可见
    assert "FAILED" in result
    assert "not found" in result
    assert "lines)" not in result


def test_non_json_result_falls_back_to_length_stub():
    result = _summarize_tool_result("patch", _patch_args(), "not-json")
    # 期望: 无法解析时退化为长度桩, 仍是单行
    assert result.startswith("[patch] replace in x.py (")
    assert "chars result" in result


def test_hostile_payload_shapes_never_raise():
    # 期望: 非字符串/畸形 diff 均不抛异常, 守卫不退化为内容长度桩
    for content in ('{"diff": 3}', '{"diff": null}', '{"diff": ["+"]}', '{"diff": "+\x00"}', ""):
        result = _summarize_tool_result("patch", _patch_args(), content)
        assert isinstance(result, str) and result.startswith("[patch]")
