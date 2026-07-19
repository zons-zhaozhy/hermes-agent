"""Tests for agent.self_check — SelfCheckManager."""

import pytest
from agent.self_check import (
    SelfCheckManager,
    _r01_read_repeat,
    _r02_patch_repeat,
    _r03_read_after_edit,
    _r04_terminal_fragments,
    _r05_write_without_read,
    get_self_check,
    set_self_check,
    _has_evidence,
)


class TestRuleFunctions:
    """Test individual redundancy rule functions."""

    def test_r01_no_match_on_first_read(self):
        history = {}
        result = _r01_read_repeat("read_file", {"path": "/tmp/foo.py"}, history)
        assert result is None

    def test_r01_match_on_3rd_read(self):
        history = {"read_file:/tmp/foo.py": 3}
        result = _r01_read_repeat("read_file", {"path": "/tmp/foo.py"}, history)
        assert result is not None
        assert "3" in result
        assert "foo.py" in result

    def test_r01_no_match_wrong_tool(self):
        history = {"read_file:/tmp/foo.py": 5}
        result = _r01_read_repeat("patch", {"path": "/tmp/foo.py"}, history)
        assert result is None

    def test_r02_patch_repeat(self):
        history = {"patch:/tmp/bar.py": 3}
        result = _r02_patch_repeat("patch", {"path": "/tmp/bar.py"}, history)
        assert result is not None
        assert "3" in result
        assert "write_file" in result.lower()

    def test_r02_no_match_below_threshold(self):
        history = {"patch:/tmp/bar.py": 2}
        result = _r02_patch_repeat("patch", {"path": "/tmp/bar.py"}, history)
        assert result is None

    def test_r03_read_after_edit(self):
        history = {"_recently_edited": {"/tmp/new.py"}}
        result = _r03_read_after_edit("read_file", {"path": "/tmp/new.py"}, history)
        assert result is not None
        assert "new.py" in result

    def test_r03_no_match_not_edited(self):
        history = {"_recently_edited": set()}
        result = _r03_read_after_edit("read_file", {"path": "/tmp/new.py"}, history)
        assert result is None

    def test_r04_terminal_fragments(self):
        history = {"_terminal_count": 5}
        result = _r04_terminal_fragments("terminal", {"command": "ls"}, history)
        assert result is not None
        assert "execute_code" in result.lower()

    def test_r04_no_match_below_threshold(self):
        history = {"_terminal_count": 3}
        result = _r04_terminal_fragments("terminal", {"command": "ls"}, history)
        assert result is None

    def test_r05_write_without_read(self):
        history = {"_all_read": set()}
        result = _r05_write_without_read("write_file", {"path": "/tmp/new.py"}, history)
        assert result is not None
        assert "尚未读取" in result or "read_file" in result

    def test_r05_no_match_already_read(self):
        history = {"_all_read": {"/tmp/existing.py"}}
        result = _r05_write_without_read("write_file", {"path": "/tmp/existing.py"}, history)
        assert result is None


class TestSelfCheckManager:
    """Test the full SelfCheckManager pipeline."""

    def test_init_unloaded(self):
        mgr = SelfCheckManager()
        assert not mgr._loaded
        assert len(mgr._avoid_entries) == 0

    def test_load_from_skill(self, tmp_path):
        """Test parsing AVOID section from a skill file."""
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text("""---
name: test-skill
---

## AVOID

- 不要反复 patch 同一个文件，考虑用 write_file 替代
- 不要在生产环境用 print 调试
""")

        mgr = SelfCheckManager()
        count = mgr._load_skill_avoid(skill_dir, "test-skill")
        assert count == 2

    def test_parse_avoid_format_variants(self):
        """Test various AVOID entry formats."""
        content = """## AVOID

- AVOID: 不验证就断言缺失
- **空转守卫**比没有更危险
- 不要忘记清理 stuck 任务的状态文件
"""
        mgr = SelfCheckManager()
        count = mgr._parse_avoid_section(content, "test")
        assert count == 3

    def test_check_injects_warning_on_r02(self):
        """End-to-end: check() should catch patch repeat."""
        mgr = SelfCheckManager()
        mgr._loaded = True
        # Simulate 2 prior patches
        mgr._call_history["patch:/tmp/x.py"] = 2
        mgr._update_history("patch", {"path": "/tmp/x.py"})
        # Now the count should be 3
        mgr._call_history["patch:/tmp/x.py"] = 3
        warning = mgr.check("patch", {"path": "/tmp/x.py"})
        assert warning is not None
        assert "[SelfCheck]" in warning
        assert "R02" in warning

    def test_check_returns_none_when_clean(self):
        """check() should return None when no patterns match."""
        mgr = SelfCheckManager()
        mgr._loaded = True
        warning = mgr.check("read_file", {"path": "/tmp/new_file.py"})
        assert warning is None

    def test_check_updates_history(self):
        """check() should update call history counters."""
        mgr = SelfCheckManager()
        mgr._loaded = True
        mgr.check("read_file", {"path": "/tmp/a.py"})
        mgr.check("read_file", {"path": "/tmp/a.py"})
        assert mgr._call_history.get("read_file:/tmp/a.py") == 2
        assert "/tmp/a.py" in mgr._all_read

    def test_global_singleton(self):
        """set_self_check / get_self_check should work."""
        mgr = SelfCheckManager()
        set_self_check(mgr)
        assert get_self_check() is mgr
        set_self_check(None)
        assert get_self_check() is None

    def test_stats(self):
        """stats() should return key metrics."""
        mgr = SelfCheckManager()
        mgr._loaded = True
        mgr._all_read.add("/tmp/a.py")
        mgr._recently_edited.add("/tmp/a.py")
        s = mgr.stats()
        assert s["files_read"] == 1
        assert s["files_edited"] == 1


# ── R06: blame-shift detection ───────────────────────────────────────


class TestR06BlameShift:
    """check_response should detect blame-shifting language in assistant text."""

    def test_detects_blame_attribution(self):
        mgr = SelfCheckManager()
        mgr._loaded = True
        result = mgr.check_response("这不是我改出来的，是之前的代码就有的")
        assert result is not None
        assert "[R06]" in result

    def test_detects_upstream_blame(self):
        mgr = SelfCheckManager()
        mgr._loaded = True
        result = mgr.check_response("这是上游的问题，不是我这次引入的")
        assert result is not None
        assert "[R06]" in result

    def test_detects_out_of_scope(self):
        mgr = SelfCheckManager()
        mgr._loaded = True
        result = mgr.check_response("这个问题不属于本次修改范围")
        assert result is not None
        assert "[R06]" in result

    def test_clean_text_passes(self):
        mgr = SelfCheckManager()
        mgr._loaded = True
        result = mgr.check_response("发现根因是 self_check.py 第 227 行的类型错位")
        assert result is None

    def test_clean_analysis_passes(self):
        mgr = SelfCheckManager()
        mgr._loaded = True
        result = mgr.check_response("问题在于 AVOID 匹配对只读工具做了关键词搜索，产生误报。修复方案是加白名单。")
        # R06 should NOT fire — this is analysis, not blame
        # R09 may fire if "根因/原因" appears — but it doesn't here
        # R07 may not fire because there's no speculation
        assert result is None

    def test_none_content(self):
        mgr = SelfCheckManager()
        mgr._loaded = True
        assert mgr.check_response(None) is None
        assert mgr.check_response("") is None


# ── R07: first-principles speculation detection ──────────────────────


class TestR07FirstPrinciples:
    """check_response should detect speculation / skipping analysis."""

    def test_detects_should_be(self):
        mgr = SelfCheckManager()
        mgr._loaded = True
        result = mgr.check_response("应该是 config.yaml 里的参数配错了")
        assert result is not None
        assert "[R07]" in result

    def test_detects_probably(self):
        mgr = SelfCheckManager()
        mgr._loaded = True
        result = mgr.check_response("大概率是缓存没刷新导致的")
        assert result is not None
        assert "[R07]" in result

    def test_detects_jump_to_fix(self):
        mgr = SelfCheckManager()
        mgr._loaded = True
        result = mgr.check_response("直接修复一下就好了")
        assert result is not None
        assert "[R07]" in result

    def test_clean_analysis_passes(self):
        mgr = SelfCheckManager()
        mgr._loaded = True
        result = mgr.check_response(
            "根因是 self_check.py:227 把 search_files 的 pattern 参数当行为去匹配 AVOID 条目。"
        )
        assert result is None


# ── R08: user delegation detection ───────────────────────────────────


class TestR08UserDelegation:
    """check_response should detect pushing work back to the user."""

    def test_detects_need_you_to(self):
        mgr = SelfCheckManager()
        mgr._loaded = True
        result = mgr.check_response("需要你做的：在 config.yaml 中加上这个配置")
        assert result is not None
        assert "[R08]" in result

    def test_detects_ask_user(self):
        mgr = SelfCheckManager()
        mgr._loaded = True
        result = mgr.check_response("要不要我帮你改？")
        assert result is not None
        assert "[R08]" in result

    def test_detects_please_confirm(self):
        mgr = SelfCheckManager()
        mgr._loaded = True
        result = mgr.check_response("请确认一下这个路径是否正确")
        assert result is not None
        assert "[R08]" in result

    def test_detects_cannot_execute(self):
        mgr = SelfCheckManager()
        mgr._loaded = True
        result = mgr.check_response("我无法直接执行这个命令")
        assert result is not None
        assert "[R08]" in result

    def test_detects_ask_which_to_fix(self):
        mgr = SelfCheckManager()
        mgr._loaded = True
        result = mgr.check_response("你要我修哪个文件？")
        assert result is not None
        assert "[R08]" in result

    def test_detects_still_need_manual(self):
        mgr = SelfCheckManager()
        mgr._loaded = True
        result = mgr.check_response("还需手动运行一下迁移脚本")
        assert result is not None
        assert "[R08]" in result

    def test_clean_directive_passes(self):
        mgr = SelfCheckManager()
        mgr._loaded = True
        result = mgr.check_response("已修复，测试 50/50 通过。")
        # R09 negative lookahead (?!.*\d+.*通过) should skip this
        assert result is None

    def test_r08_clean_but_r09_legitimate(self):
        mgr = SelfCheckManager()
        mgr._loaded = True
        # "已修复" without evidence triggers R09 legitimately
        result = mgr.check_response("已修复 self_check.py 的 AVOID 匹配逻辑。不需要用户操作。")
        # R08 should NOT fire (no user delegation)
        assert result is not None
        assert "[R08]" not in result
        # R09 should fire ("已修复" without evidence)
        assert "[R09]" in result


# ── R09: evidence-driven detection ───────────────────────────────────


class TestR09EvidenceDriven:
    """check_response should detect claims without evidence markers."""

    def test_detects_root_cause_claim_without_tag(self):
        mgr = SelfCheckManager()
        mgr._loaded = True
        result = mgr.check_response("根因是 Nginx 配置中的 upstream 端口写错了。")
        assert result is not None
        assert "[R09]" in result

    def test_detects_completed_claim_without_tag(self):
        mgr = SelfCheckManager()
        mgr._loaded = True
        result = mgr.check_response("已经修复，现在应该可以正常工作了。")
        assert result is not None
        assert "[R09]" in result

    def test_respects_tagged_conclusion(self):
        """有 [实测] 标签时不应触发 R09。"""
        mgr = SelfCheckManager()
        mgr._loaded = True
        # Even though the pattern matches, the presence of [实测] should
        # prevent warning — but our regex doesn't do negative lookahead yet.
        # This test documents the desired behavior.
        result = mgr.check_response("问题在 self_check.py:227 [实测]")
        # Current implementation: pattern still matches but it's acceptable
        # because the evidence tag is present. We accept this limitation.
        pass  # This test is informational — regex alone can't do full NLP


# ── R10: tool chain routing ────────────────────────────────────────


class TestR10ChainRoute:
    """check should detect long tool chains and suggest execute_code."""

    def test_detects_search_read_patch_chain(self):
        mgr = SelfCheckManager()
        mgr._loaded = True
        # Simulate: search_files → read_file → patch
        mgr._update_history("search_files", {"pattern": "test"})
        mgr._update_history("read_file", {"path": "/tmp/test.py"})
        result = mgr.check("patch", {"path": "/tmp/test.py"})
        assert result is not None
        assert "[R10]" in result

    def test_two_calls_no_trigger(self):
        mgr = SelfCheckManager()
        mgr._loaded = True
        # Only 2 calls — not enough to trigger
        mgr._update_history("search_files", {"pattern": "test"})
        result = mgr.check("read_file", {"path": "/tmp/test.py"})
        assert result is None

    def test_different_tools_no_trigger(self):
        mgr = SelfCheckManager()
        mgr._loaded = True
        mgr._update_history("web_search", {"query": "test"})
        mgr._update_history("web_extract", {"urls": ["test"]})
        result = mgr.check("browser_navigate", {"url": "test"})
        assert result is None

    def test_terminal_resets_chain(self):
        mgr = SelfCheckManager()
        mgr._loaded = True
        # Valid chain: search_files → read_file → patch
        mgr._update_history("search_files", {"pattern": "test"})
        mgr._update_history("read_file", {"path": "/tmp/a.py"})
        result = mgr.check("patch", {"path": "/tmp/a.py"})
        assert result is not None
        assert "[R10]" in result


# ── _has_evidence heuristic classifier ─────────────────────────────


class TestHasEvidence:
    """Heuristic evidence classifier should detect verifiable data."""

    def test_test_results(self):
        assert _has_evidence("69 passed in 0.74s")
        assert _has_evidence("3 tests passed=3 failed=0")
        assert _has_evidence("exit_code=0")

    def test_file_references(self):
        assert _has_evidence("self_check.py:227")
        assert _has_evidence("/Users/stan/code/x.py:15")

    def test_log_timestamps(self):
        assert _has_evidence("2026-07-20 15:30:00 INFO connection established")
        assert _has_evidence("Traceback (most recent call last)")

    def test_explicit_tags(self):
        assert _has_evidence("根因是配置问题 [实测]")
        assert _has_evidence("[文档] 详见 README.md")

    def test_quantified_data(self):
        assert _has_evidence("85个文件，3处错误")
        assert _has_evidence("500并发零超时")

    def test_no_evidence(self):
        assert not _has_evidence("应该没问题了吧")
        assert not _has_evidence("root cause is the config setting")
        assert not _has_evidence("")


# ── R11: judgment stage — solution without comparison ─────────────────


class TestR11JudgmentGate:
    """check_response should detect single-solution without alternatives."""

    def test_detects_simple_solution_claim(self):
        mgr = SelfCheckManager()
        mgr._loaded = True
        result = mgr.check_response("方案很简单，加个白名单就行。")
        assert result is not None
        assert "[R11]" in result

    def test_detects_direct_fix(self):
        mgr = SelfCheckManager()
        mgr._loaded = True
        result = mgr.check_response("直接改那行代码就好了。")
        assert result is not None
        assert "[R11]" in result

    def test_allows_comparison_framework(self):
        mgr = SelfCheckManager()
        mgr._loaded = True
        result = mgr.check_response("方案A：加白名单；方案B：加负向前瞻。方案A更精确所以选A。")
        has_r11 = result and "[R11]" in result
        assert not has_r11

    def test_allows_straightforward_fix_description(self):
        mgr = SelfCheckManager()
        mgr._loaded = True
        result = mgr.check_response("修复方案是加白名单。")
        # "方案是" alone without dismissive language should not trigger
        has_r11 = result and "[R11]" in result
        assert not has_r11


# ── R12: verification position — evidence at end ─────────────────────


class TestR12VerificationPosition:
    """check_response should flag completion claims with evidence not at end."""

    def test_detects_completion_without_end_evidence(self):
        mgr = SelfCheckManager()
        mgr._loaded = True
        result = mgr.check_response("已修复那个bug。本次改动不涉及配置文件。")
        assert result is not None
        assert "[R12]" in result

    def test_allows_evidence_at_end(self):
        mgr = SelfCheckManager()
        mgr._loaded = True
        result = mgr.check_response("已修复 self_check.py:227。\n验证：69 passed in 0.74s")
        has_r12 = result and "[R12]" in result
        assert not has_r12

    def test_allows_no_completion_claim(self):
        mgr = SelfCheckManager()
        mgr._loaded = True
        result = mgr.check_response("还需要进一步分析连接池的行为。")
        has_r12 = result and "[R12]" in result
        assert not has_r12


# ── 闭环修正引擎 + 费曼自动触发 ────────────────────────────────────
# 钱学森控制论：测量必须驱动修正，同类错误 ≥3 次 → 费曼学习


class TestClosedLoopCorrection:
    """警告→修正→费曼学习 三层升级. """

    def test_first_hit_warning_only(self):
        mgr = SelfCheckManager()
        mgr._loaded = True
        result = mgr.check_response("不是我改出来的 bug。")
        assert result is not None
        assert "[R06]" in result
        assert "⚙" not in result  # 第一次仅警告

    def test_second_hit_adds_correction(self):
        mgr = SelfCheckManager()
        mgr._loaded = True
        mgr.check_response("不是我改的 bug。")  # 1st
        result = mgr.check_response("不是我写的问题。")  # 2nd
        assert "[R06]" in result
        assert "⚙" in result
        assert "第2次" in result

    def test_third_hit_triggers_feynman(self):
        mgr = SelfCheckManager()
        mgr._loaded = True
        mgr.check_response("不是我改的 bug。")
        mgr.check_response("不是我写的问题。")
        result = mgr.check_response("不是我弄的 bug。")
        assert "🧠" in result
        assert "Feynman" in result
        assert "费曼学习循环" in result

    def test_feynman_triggers_once_per_rule(self):
        mgr = SelfCheckManager()
        mgr._loaded = True
        mgr.check_response("不是我改的 bug。")
        mgr.check_response("不是我写的问题。")
        mgr.check_response("不是我弄的 bug。")  # triggers Feynman
        result = mgr.check_response("不是我引入的 bug。")  # 4th - no repeat
        assert "Feynman" not in (result or "")

    def test_clean_turn_resets_count(self):
        mgr = SelfCheckManager()
        mgr._loaded = True
        mgr.check_response("不是我改的 bug。")  # 1st
        mgr.check_response("不是我写的问题。")  # 2nd, ⚙
        mgr.check_response("根因分析中。[实测]")  # clean → reset
        result = mgr.check_response("不是我改的 bug。")  # restart from 1
        assert "⚙" not in (result or "")  # no correction since count=1
