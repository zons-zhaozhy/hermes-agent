"""Regression: read_file 成功载荷携带 is_binary/not_found 键（值恒 False），
ToolSafety 的子串判定曾把每次成功读取误判为空/错误，计数永不清零，
同路径第 4 次读取被误拦（分页阅读大文件的正常用法被切断）。
契约：成功读取必须清零计数；真错误（error 键/值为 True 的 is_binary/not_found/空内容）仍计数。
"""
import json

from plugins.discipline import tool_safety


def _ok_read_result(content: str = "line_01 aaaa\nline_02 bbbb\nline_03 cccc") -> str:
    """复刻 read_file 成功返回的载荷形态（含恒在的 is_binary/not_found 键）。"""
    return json.dumps(
        {
            "content": content,
            "total_lines": 22,
            "file_size": 404,
            "truncated": True,
            "is_binary": False,
            "is_image": False,
            "not_found": False,
        },
        ensure_ascii=False,
    )


def test_successful_read_payload_is_not_classified_empty():
    got = tool_safety._is_empty_or_error_result(_ok_read_result(), "ok")
    assert got is False  # 期望: False——成功读取不得计入失败（is_binary 键值恒 False，键名非失败信号）


def test_fourth_read_after_successes_not_blocked():
    sid = "test-reset-s1"
    path = "/tmp/toolsafety_reset_probe.txt"
    for _ in range(3):
        tool_safety.on_post_tool_call(
            session_id=sid,
            tool_name="read_file",
            args={"path": path},
            result=_ok_read_result(),
            status="ok",
        )
    verdict = tool_safety.on_pre_tool_call(
        session_id=sid, tool_name="read_file", args={"path": path}
    )
    assert verdict is None  # 期望: None——每次成功读取清零计数，三次成功后计数为 0，第 4 次不拦


def test_genuine_failures_still_count():
    got = tool_safety._is_empty_or_error_result(json.dumps({"error": "File appears to be binary"}), "ok")
    assert got is True  # 期望: True——error 键存在即真失败
    got = tool_safety._is_empty_or_error_result(json.dumps({"is_binary": True}), "ok")
    assert got is True  # 期望: True——is_binary 值为 True 是真二进制
    got = tool_safety._is_empty_or_error_result(json.dumps({"not_found": True}), "ok")
    assert got is True  # 期望: True——not_found 值为 True 是文件不存在
    got = tool_safety._is_empty_or_error_result(json.dumps({"content": ""}), "ok")
    assert got is True  # 期望: True——空内容是真空读
    got = tool_safety._is_empty_or_error_result("", "error")
    assert got is True  # 期望: True——status=error 一票判失败
