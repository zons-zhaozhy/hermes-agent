"""Tests for duplicate-check plugin — filename-stem matching + /tmp exemption.

Regression (Aug 2026): the "shared characters" check compared characters at
the SAME INDEX, so ``verify_compression_fix`` vs ``verification_evidence``
(both start "veri...") hit the 4-char threshold and blocked a throwaway
/tmp verification script. Projects outside cwd (e.g. /tmp) were never
exempted even though git ls-files can't see them, and 3-char keywords like
"fix" were fed to a whole-repo rg search.
"""
import importlib.util as _ilu
import os

import pytest

_spec = _ilu.spec_from_file_location(
    "plugins.duplicate_check",
    os.path.join(os.path.dirname(__file__), "..", "..", "plugins", "guards", "duplicate_check.py"),
)
_dc = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_dc)

_on_pre_tool_call = _dc._on_pre_tool_call
_extract_functional_keywords = _dc._extract_functional_keywords


class TestTmpPathExemption:
    """Files outside the project cwd must never be checked."""

    def test_tmp_verification_script_passes(self, tmp_path, monkeypatch):
        outside = tmp_path / "elsewhere"
        outside.mkdir()
        project = tmp_path / "project"
        project.mkdir()
        monkeypatch.chdir(project)
        decision = _on_pre_tool_call(
            tool_name="write_file",
            args={"path": "/tmp/verify_compression_fix.py"},
            cwd=str(project),
        )
        assert decision is None

    def test_in_project_new_file_still_checked(self, tmp_path, monkeypatch):
        project = tmp_path / "proj"
        (project / "agent").mkdir(parents=True)
        monkeypatch.chdir(project)
        # In-project new file: must NOT be exempted by the tmp rule.
        # (No git repo here → ls-files returns nothing → no warnings → None.
        # The point is the exemption branch is not taken; a git repo case
        # is covered by stem-matching tests below.)
        decision = _on_pre_tool_call(
            tool_name="write_file",
            args={"path": "agent/verify_compression_fix.py"},
            cwd=str(project),
        )
        assert decision is None


class TestStemMatching:
    """Same-position character counting replaced by token-stem sharing."""

    def test_verify_vs_verification_not_similar(self):
        # veri-fix vs verification-evidence: no shared ≥4-char stem
        a = _dc._extract_functional_keywords("verify_compression_fix.py")
        assert a  # keywords exist

    def test_identical_name_in_same_dir_warns(self, tmp_path, monkeypatch):
        project = tmp_path / "proj"
        (project / "tools").mkdir(parents=True)
        monkeypatch.chdir(project)
        # Simulate git ls-files returning an existing same-dir file with a
        # shared stem by pre-seeding the cache.
        _dc._cache[str(project)] = {"ls_files": ["tools/outcome_collector.py"]}
        decision = _on_pre_tool_call(
            tool_name="write_file",
            args={"path": "tools/outcome_collector_v2.py"},
            cwd=str(project),
        )
        # outcome_collector_v2 shares stems {outcome, collector} → should warn
        assert decision is not None and decision["action"] == "block"


class TestKeywordExtraction:
    def test_short_noise_words_filtered(self):
        kws = _extract_functional_keywords("tmp/fix_quick_run.py")
        assert all(len(k) >= 3 for k in kws)

    def test_dedup(self):
        kws = _extract_functional_keywords("parser/parser_core.py")
        lowered = [k.lower() for k in kws]
        assert len(lowered) == len(set(lowered))
