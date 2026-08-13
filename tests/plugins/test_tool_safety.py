"""Tests for tool-safety plugin — patch counter reset + verification logic.

Regression: patch_attempts never reset on success, so 3 successful patches
on the same file blocked the 4th with a false "consecutive patch failure"
message. Fixed by resetting counter on successful patch and recognizing
read_file as a verification tool.
"""

import os

import pytest

import importlib.util as _ilu

_spec = _ilu.spec_from_file_location(
    "plugins.tool_safety",
    os.path.join(os.path.dirname(__file__), "..", "..", "plugins", "tool-safety", "__init__.py"),
)
_ts = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_ts)

_is_verification_tool_call = _ts._is_verification_tool_call
_is_batch_replace = _ts._is_batch_replace
on_post_tool_call = _ts.on_post_tool_call
on_pre_tool_call = _ts.on_pre_tool_call

from plugins._shared_state import clear_session


@pytest.fixture(autouse=True)
def reset_state():
    clear_session("test-sid")
    yield
    clear_session("test-sid")


class TestPatchCounterReset:
    """Successful patches must reset the counter — only failures accumulate."""

    def test_successful_patch_resets_counter(self):
        """3 successful patches should NOT block the 4th."""
        path = "/tmp/test_file.py"
        for _ in range(3):
            on_post_tool_call(
                session_id="test-sid",
                tool_name="patch",
                args={"path": path},
                result='{"success": true, "diff": "..."}',
                status="ok",
            )
        # 4th patch should pass
        decision = on_pre_tool_call(
            session_id="test-sid",
            tool_name="patch",
            args={"path": path},
        )
        assert decision is None  # not blocked

    def test_failed_patch_accumulates(self):
        """3 failed patches SHOULD block the 4th."""
        path = "/tmp/test_fail.py"
        for _ in range(3):
            on_post_tool_call(
                session_id="test-sid",
                tool_name="patch",
                args={"path": path},
                result='{"error": "old_string not found"}',
                status="error",
            )
        # 4th patch should be blocked
        decision = on_pre_tool_call(
            session_id="test-sid",
            tool_name="patch",
            args={"path": path},
        )
        assert decision is not None
        assert decision["action"] == "block"


class TestReadFileAsVerification:
    """read_file after patch should mark the file as verified."""

    def test_read_file_is_verification(self):
        assert _is_verification_tool_call("read_file", {}) is True

    def test_search_files_is_verification(self):
        assert _is_verification_tool_call("search_files", {}) is True

    def test_terminal_grep_is_verification(self):
        assert _is_verification_tool_call(
            "terminal", {"command": "grep foo bar.py"}
        ) is True

    def test_terminal_non_verify_is_not_verification(self):
        assert _is_verification_tool_call(
            "terminal", {"command": "npm install"}
        ) is False

    def test_read_file_after_patch_clears_attempts(self):
        """patch → read_file (verify) → next patch should not be blocked."""
        path = "/tmp/test_verify.py"
        # 3 successful patches (counter resets each time)
        for _ in range(3):
            on_post_tool_call(
                session_id="test-sid",
                tool_name="patch",
                args={"path": path},
                result='{"success": true}',
                status="ok",
            )
        # Even without read_file, counter is already 0.
        # Now simulate 2 failures + read_file verification + patch:
        on_post_tool_call(
            session_id="test-sid",
            tool_name="patch",
            args={"path": path},
            result='{"error": "mismatch"}',
            status="error",
        )
        on_post_tool_call(
            session_id="test-sid",
            tool_name="patch",
            args={"path": path},
            result='{"error": "mismatch"}',
            status="error",
        )
        # read_file verification marks as verified
        on_post_tool_call(
            session_id="test-sid",
            tool_name="read_file",
            args={"path": path},
            result="some content here",
            status="ok",
        )
        # Next patch: only 2 unverified failures, should pass
        decision = on_pre_tool_call(
            session_id="test-sid",
            tool_name="patch",
            args={"path": path},
        )
        assert decision is None


class TestBatchReplaceDetection:
    """execute_code batch str.replace should still be blocked."""

    def test_batch_replace_blocked(self):
        code = (
            "for f in files:\n"
            "    content = open(f).read()\n"
            "    content = content.replace('old', 'new')\n"
            "    open(f, 'w').write(content)\n"
            "    read_file(f)  # verify\n"
        )
        assert _is_batch_replace(code) is True

    def test_single_replace_not_blocked(self):
        code = "content = open('/tmp/a.py').read().replace('x', 'y')"
        # Only 1 indicator (replace), not enough
        assert _is_batch_replace(code) is False
