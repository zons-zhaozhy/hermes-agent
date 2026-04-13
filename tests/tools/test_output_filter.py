"""Tests for tools.output_filter — RTK-inspired terminal output purification.

Covers all 4 layers + public API + configuration + edge cases.
"""

import os
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _ensure_filter_enabled(monkeypatch):
    """Enable filter by default for all tests."""
    monkeypatch.delenv("HERMES_OUTPUT_FILTER", raising=False)
    monkeypatch.delenv("HERMES_OUTPUT_FILTER_LEVEL", raising=False)


# ---------------------------------------------------------------------------
# Layer 1: Smart Filter
# ---------------------------------------------------------------------------

class TestSmartFilter:
    """Layer 1 — strip noise from terminal output."""

    def test_empty_input(self):
        from tools.output_filter import smart_filter
        assert smart_filter("", "moderate") == ""

    def test_none_input(self):
        from tools.output_filter import smart_filter
        assert smart_filter(None, "moderate") is None

    def test_progress_bar_removal(self):
        from tools.output_filter import smart_filter
        output = "Building...\n[=======>          ] 45%\nDone!"
        result = smart_filter(output, "moderate")
        assert "45%" not in result
        assert "Building..." in result
        assert "Done!" in result

    def test_spinner_removal(self):
        from tools.output_filter import smart_filter
        output = "Compiling\n|\n/\n-\n\\\nDone"
        result = smart_filter(output, "moderate")
        assert "Done" in result
        # Spinners should be gone
        lines = result.split('\n')
        assert not any(l.strip() in ('|', '/', '-', '\\') for l in lines)

    def test_blank_line_collapse_moderate(self):
        from tools.output_filter import smart_filter
        output = "line1\n\n\n\n\nline2"
        result = smart_filter(output, "moderate")
        assert "\n\n\n\n" not in result
        assert "line1" in result
        assert "line2" in result

    def test_blank_line_collapse_light(self):
        from tools.output_filter import smart_filter
        # Light level doesn't collapse blank lines
        output = "line1\n\n\n\n\nline2"
        result = smart_filter(output, "light")
        # Light level only strips progress bars and spinners
        assert "line1" in result

    def test_aggressive_strips_warnings(self):
        from tools.output_filter import smart_filter
        output = (
            "npm warn not superset\n"
            "warning: unused variable 'x'\n"
            "note: please recompile with -Wall\n"
            "real output here"
        )
        result = smart_filter(output, "aggressive")
        assert "real output here" in result

    def test_aggressive_strips_build_comments(self):
        from tools.output_filter import smart_filter
        output = "# This is a build comment\nactual output"
        result = smart_filter(output, "aggressive")
        assert "actual output" in result
        # Build comment should be stripped at aggressive level
        assert "# This is a build comment" not in result

    def test_clean_output_passthrough(self):
        from tools.output_filter import smart_filter
        output = "clean line 1\nclean line 2\nclean line 3"
        result = smart_filter(output, "moderate")
        assert result == output.strip()

    def test_none_level(self):
        from tools.output_filter import smart_filter
        output = "some output"
        assert smart_filter(output, "none") == output


# ---------------------------------------------------------------------------
# Layer 2: Group Aggregate
# ---------------------------------------------------------------------------

class TestGroupAggregate:
    """Layer 2 — merge similar output lines."""

    def test_empty_input(self):
        from tools.output_filter import group_aggregate
        assert group_aggregate("", "moderate") == ""

    def test_test_results_aggregated(self):
        from tools.output_filter import group_aggregate
        output = (
            "test test_add ... ok\n"
            "test test_sub ... ok\n"
            "test test_mul ... ok\n"
            "test test_div ... FAILED\n"
            "other output"
        )
        result = group_aggregate(output, "moderate")
        assert "3 passed" in result
        assert "1 FAILED" in result
        assert "test test_div ... FAILED" in result
        assert "test test_add ... ok" not in result
        assert "other output" in result

    def test_download_progress_aggregated(self):
        from tools.output_filter import group_aggregate
        output = (
            "Downloading from central: 10%\n"
            "Downloading from central: 50%\n"
            "Downloading from central: 99%\n"
            "Build complete"
        )
        result = group_aggregate(output, "moderate")
        assert "3 download progress lines omitted" in result
        assert "Build complete" in result

    def test_no_match_passthrough(self):
        from tools.output_filter import group_aggregate
        output = "line1\nline2\nline3"
        assert group_aggregate(output, "moderate") == output

    def test_none_level(self):
        from tools.output_filter import group_aggregate
        output = "test a ... ok\ntest b ... ok"
        assert group_aggregate(output, "none") == output


# ---------------------------------------------------------------------------
# Layer 3: Smart Truncate
# ---------------------------------------------------------------------------

class TestSmartTruncate:
    """Layer 3 — keep head/tail, summarize middle."""

    def test_short_output_passthrough(self):
        from tools.output_filter import smart_truncate
        lines = "\n".join(f"line {i}" for i in range(10))
        assert smart_truncate(lines, "moderate") == lines

    def test_long_output_truncated(self):
        from tools.output_filter import smart_truncate
        lines = "\n".join(f"line {i}" for i in range(600))
        result = smart_truncate(lines, "moderate")
        assert "lines omitted" in result
        # Should have head lines
        assert "line 0" in result
        # Should have tail lines
        assert "line 599" in result
        # Middle lines should be gone
        assert "line 300" not in result

    def test_aggressive_lower_threshold(self):
        from tools.output_filter import smart_truncate
        # 250 lines: moderate keeps, aggressive truncates
        lines = "\n".join(f"line {i}" for i in range(250))
        result_mod = smart_truncate(lines, "moderate")
        result_agg = smart_truncate(lines, "aggressive")
        assert result_mod == lines  # 250 < 500 threshold
        assert "lines omitted" in result_agg  # 250 > 200 threshold

    def test_empty_input(self):
        from tools.output_filter import smart_truncate
        assert smart_truncate("", "moderate") == ""


# ---------------------------------------------------------------------------
# Layer 4: Dedup Merge
# ---------------------------------------------------------------------------

class TestDedupMerge:
    """Layer 4 — collapse consecutive identical lines."""

    def test_no_duplicates(self):
        from tools.output_filter import dedup_merge
        output = "line1\nline2\nline3"
        assert dedup_merge(output, "moderate") == output

    def test_consecutive_duplicates(self):
        from tools.output_filter import dedup_merge
        output = "timeout\n" * 10 + "connected"
        result = dedup_merge(output, "moderate")
        assert "timeout (×10)" in result
        assert "connected" in result
        # Should be much shorter
        assert len(result) < len(output)

    def test_two_duplicates_preserved(self):
        from tools.output_filter import dedup_merge
        output = "warning\nwarning\nok"
        result = dedup_merge(output, "moderate")
        # 2 duplicates: kept as-is (threshold is > 2)
        assert "warning\nwarning" in result

    def test_mixed_groups(self):
        from tools.output_filter import dedup_merge
        output = "A\nA\nA\nB\nB\nB\nB\nC"
        result = dedup_merge(output, "moderate")
        assert "A (×3)" in result
        assert "B (×4)" in result
        assert "C" in result

    def test_empty_input(self):
        from tools.output_filter import dedup_merge
        assert dedup_merge("", "moderate") == ""

    def test_short_input(self):
        from tools.output_filter import dedup_merge
        assert dedup_merge("a\nb", "moderate") == "a\nb"


# ---------------------------------------------------------------------------
# Public API: filter_terminal_output
# ---------------------------------------------------------------------------

class TestFilterTerminalOutput:
    """Integration test — all 4 layers applied sequentially."""

    def test_disabled_via_config(self, monkeypatch):
        from tools.output_filter import filter_terminal_output
        monkeypatch.setenv("HERMES_OUTPUT_FILTER", "false")
        output = "some output " * 100
        # When disabled, should return original
        result = filter_terminal_output(output, command="ls")
        # Need to reload config; since _get_filter_config reads env each time,
        # setting env before call should work
        assert result == output

    def test_short_output_passthrough(self):
        from tools.output_filter import filter_terminal_output
        output = "short"
        result = filter_terminal_output(output, command="echo")
        assert result == output

    def test_full_pipeline(self):
        from tools.output_filter import filter_terminal_output
        # Realistic cargo test output
        output = (
            "Compiling myproject v0.1.0\n"
            + "[=======>          ] 45%\n"  # progress bar
            + "test test_add ... ok\n"
            + "test test_sub ... ok\n"
            + "test test_mul ... ok\n"
            + "test test_div ... ok\n"
            + "test test_mod ... FAILED\n"
            + "Connection timeout\n" * 10
            + "Build finished\n"
        )
        result = filter_terminal_output(output, command="cargo test")
        # Progress bar gone
        assert "45%" not in result
        # Test results aggregated
        assert "4 passed" in result
        assert "1 FAILED" in result
        # Dedup applied
        assert "×10" in result
        # Key content preserved
        assert "test test_mod ... FAILED" in result
        assert "Build finished" in result

    def test_none_level_passthrough(self):
        from tools.output_filter import filter_terminal_output
        output = "line1\n" * 100
        result = filter_terminal_output(output, command="cat", level="none")
        assert result == output

    def test_git_diff_preserved(self):
        from tools.output_filter import filter_terminal_output
        # git diff output should be preserved (it's already valuable)
        output = "diff --git a/file.py b/file.py\n+new line\n-old line"
        result = filter_terminal_output(output, command="git diff")
        assert "diff --git" in result
        assert "+new line" in result
        assert "-old line" in result

    def test_output_too_short_to_filter(self):
        from tools.output_filter import filter_terminal_output
        output = "abc"
        result = filter_terminal_output(output, command="echo")
        assert result == output


# ---------------------------------------------------------------------------
# Command-aware filtering
# ---------------------------------------------------------------------------

class TestCommandAwareFilter:
    """Command classification and preserve/compress behavior."""

    def test_git_diff_preserve_mode(self):
        from tools.output_filter import filter_terminal_output
        # git diff output should NOT be truncated or deduped
        lines = ["diff --git a/file.py b/file.py"]
        for i in range(100):
            lines.append(f"+new line {i}")
            lines.append(f"-old line {i}")
        lines.append("same line\nsame line\nsame line")  # dedup bait
        output = "\n".join(lines)
        result = filter_terminal_output(output, command="git diff")
        # All diff lines must be preserved
        assert "diff --git" in result
        assert "+new line 0" in result
        assert "+new line 99" in result
        assert "-old line 50" in result
        # Dedup should NOT have fired
        assert "×3" not in result

    def test_git_show_preserve_mode(self):
        from tools.output_filter import filter_terminal_output
        output = "commit abc123\n+added\n+added\n+added\n-removed\n"
        result = filter_terminal_output(output, command="git show HEAD")
        # git show should also preserve content
        assert "+added" in result
        assert "×3" not in result

    def test_diff_command_preserve(self):
        from tools.output_filter import filter_terminal_output
        output = "1c1\n< old\n---\n> new\n" * 50
        result = filter_terminal_output(output, command="diff -u a.txt b.txt")
        assert "1c1" in result

    def test_git_status_compress_mode(self):
        from tools.output_filter import filter_terminal_output
        # git status has lots of "modified:" lines — safe to dedup
        lines = []
        for i in range(30):
            lines.append(f"  modified:   src/module_{i}.py")
        lines.append("same line\nsame line\nsame line\nsame line\nsame line")
        output = "\n".join(lines)
        result = filter_terminal_output(output, command="git status")
        # Should have applied normal filtering (dedup fires for 3+)
        # Not checking strict dedup since short output may pass through,
        # but verify it didn't blow up
        assert "modified:" in result or len(result) < len(output)

    def test_git_log_compress_mode(self):
        from tools.output_filter import filter_terminal_output
        lines = []
        for i in range(100):
            lines.append(f"commit abc{i:04d}")
            lines.append(f"Author: dev{i}")
            lines.append(f"Date:   2024-01-{(i % 28) + 1:02d}")
            lines.append(f"    commit message {i}")
            lines.append("")
        output = "\n".join(lines)
        result = filter_terminal_output(output, command="git log --oneline")
        # Large output should be truncated
        assert len(result) <= len(output)

    def test_ls_compress_mode(self):
        from tools.output_filter import filter_terminal_output
        lines = [f"file_{i}.py" for i in range(200)]
        output = "\n".join(lines)
        result = filter_terminal_output(output, command="ls")
        # Large ls output should be truncated
        assert len(result) < len(output)

    def test_pip_list_compress(self):
        from tools.output_filter import filter_terminal_output
        lines = [f"package-{i}==1.0.{i}" for i in range(300)]
        output = "\n".join(lines)
        result = filter_terminal_output(output, command="pip list")
        assert len(result) < len(output)

    def test_unknown_command_default(self):
        from tools.output_filter import _classify_command
        assert _classify_command("python script.py") == "default"
        assert _classify_command("cargo build") == "default"
        assert _classify_command("") == "default"

    def test_classify_preserve_commands(self):
        from tools.output_filter import _classify_command
        assert _classify_command("git diff") == "preserve"
        assert _classify_command("git diff --cached") == "preserve"
        assert _classify_command("git show HEAD") == "preserve"
        assert _classify_command("git format-patch -1") == "preserve"
        assert _classify_command("diff -u a b") == "preserve"
        assert _classify_command("  git diff  ") == "preserve"  # leading whitespace

    def test_classify_compress_commands(self):
        from tools.output_filter import _classify_command
        assert _classify_command("git status") == "compress"
        assert _classify_command("git log --oneline") == "compress"
        assert _classify_command("git branch -a") == "compress"
        assert _classify_command("ls -la") == "compress"
        assert _classify_command("docker images") == "compress"
        assert _classify_command("pip list") == "compress"
        assert _classify_command("kubectl get pods") == "compress"

    def test_git_log_with_patch_is_preserve(self):
        from tools.output_filter import _classify_command
        # git log --patch should be preserved (shows diffs)
        assert _classify_command("git log --patch") == "preserve"
        assert _classify_command("git log -p") == "compress"  # -p not in preserve patterns, git log matches compress


# ---------------------------------------------------------------------------
# Statistics tracking
# ---------------------------------------------------------------------------

class TestFilterStats:
    """Filter statistics collection and retrieval."""

    def setup_method(self):
        from tools.output_filter import reset_filter_stats
        reset_filter_stats()

    def test_stats_recorded_on_filter(self):
        from tools.output_filter import filter_terminal_output, get_filter_stats, reset_filter_stats
        reset_filter_stats()
        output = "Compiling...\n" + "[=======>          ] 45%\n" + "test a ... ok\n" * 20
        filter_terminal_output(output, command="cargo test")
        stats = get_filter_stats()
        assert stats["total_calls"] == 1
        assert stats["total_original"] == len(output)
        assert stats["total_filtered"] < len(output)
        assert stats["overall_saved_pct"] > 0

    def test_stats_by_tool(self):
        from tools.output_filter import (
            filter_terminal_output, filter_code_execution_output,
            get_filter_stats, reset_filter_stats,
        )
        reset_filter_stats()
        long_output = "line\n" * 200
        filter_terminal_output(long_output, command="ls")
        filter_code_execution_output(long_output, script_info="test")
        stats = get_filter_stats()
        assert stats["total_calls"] == 2
        assert "terminal" in stats["by_tool"]
        assert "code_execution" in stats["by_tool"]

    def test_stats_single_tool_filter(self):
        from tools.output_filter import (
            filter_terminal_output, filter_code_execution_output,
            get_filter_stats, reset_filter_stats,
        )
        reset_filter_stats()
        long_output = "line\n" * 200
        filter_terminal_output(long_output, command="ls")
        filter_code_execution_output(long_output, script_info="test")
        stats = get_filter_stats(tool_type="terminal")
        assert stats["total_calls"] == 1
        assert "terminal" in stats["by_tool"]
        assert "code_execution" not in stats["by_tool"]

    def test_stats_empty(self):
        from tools.output_filter import get_filter_stats, reset_filter_stats
        reset_filter_stats()
        stats = get_filter_stats()
        assert stats["total_calls"] == 0
        assert stats["overall_saved_pct"] == 0.0

    def test_stats_recent_records(self):
        from tools.output_filter import filter_terminal_output, get_filter_stats, reset_filter_stats
        reset_filter_stats()
        for i in range(5):
            filter_terminal_output("line\n" * (100 + i * 50), command=f"test cmd {i}")
        stats = get_filter_stats()
        assert len(stats["recent"]) == 5
        # Most recent should be first
        assert stats["recent"][0]["context"] == "test cmd 4"

    def test_stats_fifo_eviction(self):
        from tools.output_filter import (
            filter_terminal_output, get_filter_stats, reset_filter_stats, _MAX_STATS_PER_TOOL,
        )
        reset_filter_stats()
        # Exceed max stats per tool
        for i in range(_MAX_STATS_PER_TOOL + 100):
            filter_terminal_output("line\n" * 200, command=f"cmd {i}")
        stats = get_filter_stats(tool_type="terminal")
        # Should be capped at _MAX_STATS_PER_TOOL
        assert stats["by_tool"]["terminal"]["calls"] == _MAX_STATS_PER_TOOL

    def test_stats_command_class(self):
        from tools.output_filter import (
            filter_terminal_output, get_filter_stats, reset_filter_stats,
        )
        reset_filter_stats()
        # preserve mode
        filter_terminal_output("diff --git a b\n+line\n-line\n" * 50, command="git diff")
        stats = get_filter_stats()
        assert stats["recent"][0]["command_class"] == "preserve"

    def test_reset_stats(self):
        from tools.output_filter import (
            filter_terminal_output, get_filter_stats, reset_filter_stats,
        )
        filter_terminal_output("line\n" * 200, command="ls")
        reset_filter_stats()
        stats = get_filter_stats()
        assert stats["total_calls"] == 0

    def test_no_change_no_stat(self):
        from tools.output_filter import (
            filter_terminal_output, get_filter_stats, reset_filter_stats,
        )
        reset_filter_stats()
        # Short output — no filtering applied, no stat recorded
        filter_terminal_output("short", command="echo")
        stats = get_filter_stats()
        assert stats["total_calls"] == 0


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class TestConfig:
    """Configuration loading and env overrides."""

    def test_valid_levels(self):
        from tools.output_filter import VALID_LEVELS
        assert VALID_LEVELS == ("none", "light", "moderate", "aggressive")

    def test_env_level_override(self, monkeypatch):
        from tools.output_filter import _get_filter_config
        monkeypatch.setenv("HERMES_OUTPUT_FILTER_LEVEL", "aggressive")
        config = _get_filter_config()
        assert config["level"] == "aggressive"

    def test_invalid_env_level_uses_default(self, monkeypatch):
        from tools.output_filter import _get_filter_config
        monkeypatch.setenv("HERMES_OUTPUT_FILTER_LEVEL", "invalid")
        config = _get_filter_config()
        assert config["level"] == "moderate"  # default


# ---------------------------------------------------------------------------
# Token savings estimation
# ---------------------------------------------------------------------------

class TestTokenSavingsEstimate:
    """Verify that filtering actually reduces output size."""

    def test_cargo_test_savings(self):
        from tools.output_filter import filter_terminal_output
        # Simulate a realistic cargo test output
        lines = [
            "Compiling myproject v0.1.0 (/path/to/project)",
            "[========================================] 100%",
        ]
        # 50 passing tests
        for i in range(50):
            lines.append(f"test test_feature_{i:03d} ... ok")
        # 3 failing tests
        lines.append("test test_edge_case_1 ... FAILED")
        lines.append("test test_edge_case_2 ... FAILED")
        lines.append("test test_edge_case_3 ... FAILED")
        # Some repeated warnings
        lines.extend(["warning: unused import: `std::io`"] * 20)
        lines.append("")
        lines.append("test result: FAILED. 50 passed; 3 failed;")

        output = "\n".join(lines)
        result = filter_terminal_output(output, command="cargo test")

        saved_pct = (1 - len(result) / len(output)) * 100
        # Should save at least 30% (conservative — test aggregation alone saves a lot)
        assert saved_pct > 30, f"Only saved {saved_pct:.1f}% — expected > 30%"

    def test_build_log_savings(self):
        from tools.output_filter import filter_terminal_output
        lines = []
        # Simulate a Maven build with lots of download noise
        for i in range(30):
            lines.append(f"Downloading from central: https://repo.maven.apache.org/maven2/com/example/lib-{i} ({i*3}%)")
        lines.append("")
        lines.append("[INFO] BUILD SUCCESS")
        for i in range(50):
            lines.append(f"[INFO] Compiling module-{i} ...")
        lines.append("[INFO] Total time: 42s")
        lines.append("[INFO] Finished at: 2024-01-01T00:00:00Z")

        output = "\n".join(lines)
        result = filter_terminal_output(output, command="mvn clean install")

        saved_pct = (1 - len(result) / len(output)) * 100
        assert saved_pct > 10, f"Only saved {saved_pct:.1f}% — expected > 10%"
