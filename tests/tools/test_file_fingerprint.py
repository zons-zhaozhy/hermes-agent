"""Fingerprint freshness protocol tests (hashline-style, adapted).

Contracts under test (expectations derived from the protocol, not the code):
  1. content_fingerprint: same bytes -> same 12-hex; BOM/CRLF/CR differences
     do NOT change it (editor re-serialization must never false-reject);
     different content -> different fingerprint.
  2. Registry: record_read/last_seen round-trip; cap evicts oldest;
     drop_task clears; record_write refreshes.
  3. patch_tool integration: stale explicit fingerprint -> rejected, file
     untouched; no fingerprint -> old behavior; registry auto-check warns
     (does not block) on externally-changed files.
  4. write_file integration: stale explicit fingerprint -> rejected;
     overwrite without credential proceeds (with warning when registry knows).
  5. finish_guard stall counting: same clarify question 3x -> stall hint in
     block message; different question resets the streak.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.file_fingerprint import (
    content_fingerprint, fingerprint_matches, record_read, last_seen,
    record_write, drop_task, _REGISTRY_CAP,
)


class TestContentFingerprint:
    def test_same_content_same_hash(self):
        assert content_fingerprint("a\nb\n") == content_fingerprint("a\nb\n")

    def test_bom_crlf_normalization(self):
        assert content_fingerprint("﻿a\nb\n") == content_fingerprint("a\nb\n")
        assert content_fingerprint("a\r\nb\r\n") == content_fingerprint("a\nb\n")
        assert content_fingerprint("a\rb\n") == content_fingerprint("a\nb\n")

    def test_different_content_different_hash(self):
        assert content_fingerprint("a\nb\n") != content_fingerprint("a\nb\nc\n")

    def test_length_12_hex(self):
        fp = content_fingerprint("x")
        assert len(fp) == 12 and all(c in "0123456789abcdef" for c in fp)

    def test_matches_helper(self):
        assert fingerprint_matches("a\n", content_fingerprint("a\n"))
        assert not fingerprint_matches("b\n", content_fingerprint("a\n"))


class TestRegistry:
    def test_roundtrip_and_drop(self):
        record_read("t1", "/x.py", "aaa")
        assert last_seen("t1", "/x.py") == "aaa"
        assert last_seen("t2", "/x.py") is None  # task isolation
        drop_task("t1")
        assert last_seen("t1", "/x.py") is None

    def test_cap_evicts_oldest(self):
        drop_task("cap")
        for i in range(_REGISTRY_CAP + 5):
            record_read("cap", f"/f{i}.py", f"{i:012x}")
        assert last_seen("cap", "/f0.py") is None      # oldest evicted
        assert last_seen("cap", f"/f{_REGISTRY_CAP + 4}.py") is not None

    def test_record_write_refreshes(self):
        record_read("t3", "/y.py", "old")
        record_write("t3", "/y.py", "new")
        assert last_seen("t3", "/y.py") == "new"


class TestPatchIntegration:
    def test_stale_fingerprint_rejected_file_untouched(self, tmp_path):
        from tools.file_tools import read_file_tool, patch_tool
        f = tmp_path / "t.py"
        f.write_text("a = 1\n", encoding="utf-8")
        fp = json.loads(read_file_tool(str(f), task_id="fp-p1"))["fingerprint"]
        f.write_text("a = 999\n", encoding="utf-8")  # external change
        out = patch_tool(path=str(f), old_string="a = 1", new_string="a = 2",
                         expected_fingerprint=fp, task_id="fp-p1")
        assert "FINGERPRINT MISMATCH" in out
        assert f.read_text(encoding="utf-8") == "a = 999\n"

    def test_no_fingerprint_keeps_old_behavior(self, tmp_path):
        from tools.file_tools import patch_tool
        f = tmp_path / "t.py"
        f.write_text("a = 1\n", encoding="utf-8")
        out = patch_tool(path=str(f), old_string="a = 1", new_string="a = 2",
                         task_id="fp-p2")
        assert "error" not in json.loads(out)
        assert "a = 2" in f.read_text(encoding="utf-8")

    def test_registry_autocheck_warns_not_blocks(self, tmp_path):
        from tools.file_tools import read_file_tool, patch_tool
        f = tmp_path / "t.py"
        f.write_text("a = 1\n", encoding="utf-8")
        json.loads(read_file_tool(str(f), task_id="fp-p3"))  # registry entry
        f.write_text("a = 1\nb = 2\n", encoding="utf-8")     # external append
        out = patch_tool(path=str(f), old_string="a = 1", new_string="a = 2",
                         task_id="fp-p3")                     # NO credential
        d = json.loads(out)
        assert not d.get("error"), d.get("error")             # must not block
        assert "fingerprint" in str(d.get("_warning", ""))    # must warn
        assert "a = 2" in f.read_text(encoding="utf-8")       # edit landed

    def test_post_write_restamp_clears_warning(self, tmp_path):
        from tools.file_tools import read_file_tool, patch_tool
        f = tmp_path / "t.py"
        f.write_text("a = 1\n", encoding="utf-8")
        json.loads(read_file_tool(str(f), task_id="fp-p4"))
        patch_tool(path=str(f), old_string="a = 1", new_string="a = 2",
                   task_id="fp-p4")
        # second patch of the just-written content: registry was restamped -> no warning
        out = patch_tool(path=str(f), old_string="a = 2", new_string="a = 3",
                         task_id="fp-p4")
        assert "fingerprint" not in str(json.loads(out).get("_warning", ""))


class TestWriteFileIntegration:
    def test_stale_fingerprint_rejected(self, tmp_path):
        from tools.file_tools import read_file_tool, write_file_tool
        f = tmp_path / "t.py"
        f.write_text("keep me\n", encoding="utf-8")
        fp = json.loads(read_file_tool(str(f), task_id="fp-w1"))["fingerprint"]
        f.write_text("keep me\nexternal edit\n", encoding="utf-8")
        out = write_file_tool(str(f), "destroyed\n", task_id="fp-w1",
                              expected_fingerprint=fp)
        assert "FINGERPRINT MISMATCH" in out
        assert "external edit" in f.read_text(encoding="utf-8")

    def test_overwrite_without_credential_proceeds_with_warning(self, tmp_path):
        from tools.file_tools import read_file_tool, write_file_tool
        f = tmp_path / "t.py"
        f.write_text("v1\n", encoding="utf-8")
        json.loads(read_file_tool(str(f), task_id="fp-w2"))
        f.write_text("v2\n", encoding="utf-8")  # external change
        out = write_file_tool(str(f), "v3\n", task_id="fp-w2")
        d = json.loads(out)
        assert not d.get("error"), d.get("error")
        assert "fingerprint" in str(d.get("_warning", ""))
        assert f.read_text(encoding="utf-8") == "v3\n"


class TestFinishGuardStall:
    def _call(self, questions):
        from plugins import finish_guard
        return finish_guard._on_pre_tool_call(
            tool_name="clarify", args={"questions": questions})

    def test_same_question_three_times_yields_stall_hint(self):
        q = [{"question": "要不要继续？"}]
        r1 = self._call(q); r2 = self._call(q); r3 = self._call(q)
        assert r1["action"] == "block"
        assert "原地打转" not in r1["message"]
        assert "原地打转" not in r2["message"]
        assert "原地打转" in r3["message"]  # 3rd consecutive same-gap

    def test_different_question_resets_streak(self):
        self._call([{"question": "要不要继续？"}])
        self._call([{"question": "要不要继续？"}])
        r = self._call([{"question": "现在删除生产库吗？"}])  # different gap + legit ask
        assert r == {}  # legit exemption (生产库) passes through

    def test_reordered_questions_same_fingerprint(self):
        qs1 = [{"question": "要不要继续A？"}, {"question": "要不要继续B？"}]
        qs2 = [{"question": "要不要继续B？"}, {"question": "要不要继续A？"}]
        self._call(qs1); self._call(qs1)
        r = self._call(qs2)  # same content, different order -> same gap
        assert "原地打转" in r["message"]
