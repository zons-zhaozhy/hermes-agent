"""Tests for agent.self_check — SelfCheckManager."""

import pytest
from agent.self_check import (
    SelfCheckManager,
    _r01_read_repeat,
    _r02_patch_repeat,
    _r03_read_after_edit,
    _r04_terminal_fragments,
    _r05_write_without_read,
    _r10_chain_route,
    _r13_task_drift,
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
        history = {"_all_touched_before_update": set()}
        result = _r05_write_without_read("write_file", {"path": "/tmp/new.py"}, history)
        assert result is not None
        assert "尚未读取" in result or "read_file" in result

    def test_r05_no_match_already_read(self):
        history = {"_all_touched_before_update": {"/tmp/existing.py"}}
        result = _r05_write_without_read("write_file", {"path": "/tmp/existing.py"}, history)
        assert result is None


class TestIntegrationCheckPipeline:
    """集成路径测试——走 check()/check_response() 完整管道。"""

    def test_r05_fires_through_check_pipeline(self):
        mgr = SelfCheckManager()
        mgr._loaded = True
        result = mgr.check("write_file", {"path": "/tmp/integration_r05.py", "content": "x"})
        assert result is not None, "R05 应通过 check() 管道命中"
        assert "[R05]" in result

    def test_r05_no_fire_through_check_pipeline(self):
        mgr = SelfCheckManager()
        mgr._loaded = True
        mgr.check("read_file", {"path": "/tmp/integration_r05_read.py"})
        result = mgr.check("write_file", {"path": "/tmp/integration_r05_read.py", "content": "x"})
        assert not result or "[R05]" not in result

    def test_r13_fires_through_check_pipeline(self):
        mgr = SelfCheckManager()
        mgr._loaded = True
        mgr.check("read_file", {"path": "/src/a.py"})
        mgr.check("read_file", {"path": "/src/b.py"})
        mgr.check("patch", {"path": "/src/c.py", "old_string": "x", "new_string": "y"})
        result = mgr.check("write_file", {"path": "/docs/drift.md", "content": "x"})
        assert result is not None, "R13 应通过 check() 管道命中"
        assert "[R13]" in result

    def test_r14_fires_through_response_pipeline(self):
        mgr = SelfCheckManager()
        mgr._loaded = True
        mgr.record_tool_result("terminal", "Traceback (most recent call last)\nError: x")
        result = mgr.check_response("已修复，测试通过。")
        assert result is not None, "R14 应通过 check_response() 管道命中"
        assert "[R14]" in result

    def test_r07_fires_through_response_pipeline(self):
        mgr = SelfCheckManager()
        mgr._loaded = True
        result = mgr.check_response("应该没问题。")
        assert result is not None, "R07 应通过 check_response() 管道命中"
        assert "[R07]" in result


class TestSelfCheckManager:
    """Test the full SelfCheckManager pipeline."""

    def test_init_unloaded(self):
        mgr = SelfCheckManager()
        assert not mgr._loaded
        assert len(mgr._avoid_entries) == 0

    def test_load_from_skill(self, tmp_path):
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
        content = """## AVOID

- AVOID: 不验证就断言缺失
- **空转守卫**比没有更危险
- 不要忘记清理 stuck 任务的状态文件
"""
        mgr = SelfCheckManager()
        count = mgr._parse_avoid_section(content, "test")
        assert count == 3

    def test_check_injects_warning_on_r02(self):
        mgr = SelfCheckManager()
        mgr._loaded = True
        mgr._call_history["patch:/tmp/x.py"] = 2
        mgr._update_history("patch", {"path": "/tmp/x.py"})
        mgr._call_history["patch:/tmp/x.py"] = 3
        warning = mgr.check("patch", {"path": "/tmp/x.py"})
        assert warning is not None
        assert "[R02]" in warning

    def test_check_returns_none_when_clean(self):
        mgr = SelfCheckManager()
        mgr._loaded = True
        warning = mgr.check("read_file", {"path": "/tmp/new_file.py"})
        assert warning is None

    def test_check_updates_history(self):
        mgr = SelfCheckManager()
        mgr._loaded = True
        mgr.check("read_file", {"path": "/tmp/a.py"})
        mgr.check("read_file", {"path": "/tmp/a.py"})
        assert mgr._call_history.get("read_file:/tmp/a.py") == 2
        assert "/tmp/a.py" in mgr._all_read

    def test_global_singleton(self):
        mgr = SelfCheckManager()
        set_self_check(mgr)
        assert get_self_check() is mgr
        set_self_check(None)
        assert get_self_check() is None

    def test_stats(self):
        mgr = SelfCheckManager()
        mgr._loaded = True
        mgr._all_read.add("/tmp/a.py")
        mgr._recently_edited.add("/tmp/a.py")
        s = mgr.stats()
        assert s["files_read"] == 1
        assert s["files_edited"] == 1


# ── R06: blame-shift detection ───────────────────────────────────────


class TestR06BlameShift:
    """check_response should detect blame-shifting language."""

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
        assert result is None

    def test_none_content(self):
        mgr = SelfCheckManager()
        mgr._loaded = True
        assert mgr.check_response(None) is None
        assert mgr.check_response("") is None


# ── R07: first-principles speculation detection ──────────────────────


class TestR07FirstPrinciples:
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
        assert result is None

    def test_r08_clean_text_passes(self):
        mgr = SelfCheckManager()
        mgr._loaded = True
        result = mgr.check_response("已修复 self_check.py 的 AVOID 匹配逻辑。不需要用户操作。")
        assert result is None or "[R08]" not in result


# ── R09 removed: evidence-tagging enforcement was too noisy. ────────


# ── R10: tool chain routing ────────────────────────────────────────


class TestR10ChainRoute:
    def test_detects_search_read_patch_chain(self):
        mgr = SelfCheckManager()
        mgr._loaded = True
        mgr._update_history("search_files", {"pattern": "test"})
        mgr._update_history("read_file", {"path": "/tmp/test.py"})
        result = mgr.check("patch", {"path": "/tmp/test.py"})
        assert result is not None
        assert "[R10]" in result

    def test_two_calls_no_trigger(self):
        mgr = SelfCheckManager()
        mgr._loaded = True
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
        mgr._update_history("search_files", {"pattern": "test"})
        mgr._update_history("read_file", {"path": "/tmp/a.py"})
        result = mgr.check("patch", {"path": "/tmp/a.py"})
        assert result is not None
        assert "[R10]" in result


# ── _has_evidence heuristic classifier ─────────────────────────────


class TestHasEvidence:
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


# ── R11: judgment stage ─────────────────────────────────────────────


class TestR11JudgmentGate:
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
        has_r11 = result and "[R11]" in result
        assert not has_r11


# ── R10 extended ───────────────────────────────────────────────────


class TestR10ExtendedChains:
    def test_terminal_streak_3(self):
        mgr = SelfCheckManager()
        mgr._loaded = True
        mgr._update_history("terminal", {"command": "ls"})
        mgr._update_history("terminal", {"command": "pwd"})
        mgr._update_history("terminal", {"command": "whoami"})
        mgr._call_history["_recent_tools"] = mgr._recent_tools
        result = mgr.check("terminal", {"command": "date"})
        assert "[R10]" in (result or "")
        assert "terminal" in result.lower()

    def test_search_read_write_chain(self):
        mgr = SelfCheckManager()
        mgr._loaded = True
        mgr._update_history("search_files", {"pattern": "test"})
        mgr._update_history("read_file", {"path": "/tmp/a.py"})
        result = mgr.check("write_file", {"path": "/tmp/a.py"})
        assert "[R10]" in (result or "")


# ── R13: task drift detection ─────────────────────────────────────


class TestR13TaskDrift:
    def test_drift_detected(self):
        mgr = SelfCheckManager()
        mgr._loaded = True
        mgr._update_history("read_file", {"path": "/src/a.py"})
        mgr._update_history("read_file", {"path": "/src/b.py"})
        mgr._update_history("patch", {"path": "/src/c.py"})
        result = mgr.check("write_file", {"path": "/docs/manual.md"})
        assert result is not None
        assert "[R13]" in result

    def test_same_directory_no_drift(self):
        mgr = SelfCheckManager()
        mgr._loaded = True
        mgr._update_history("read_file", {"path": "/src/a.py"})
        mgr._update_history("read_file", {"path": "/src/b.py"})
        mgr._update_history("patch", {"path": "/src/c.py"})
        result = mgr.check("write_file", {"path": "/src/d.py"})
        has_r13 = result and "[R13]" in result
        assert not has_r13

    def test_few_files_no_drift(self):
        mgr = SelfCheckManager()
        mgr._loaded = True
        mgr._update_history("read_file", {"path": "/src/a.py"})
        result = mgr.check("write_file", {"path": "/docs/x.md"})
        has_r13 = result and "[R13]" in result
        assert not has_r13


# ── R14: tool result ignored ─────────────────────────────────────


class TestR14ResultIgnored:
    def test_error_ignored(self):
        mgr = SelfCheckManager()
        mgr._loaded = True
        mgr.record_tool_result("terminal", "Traceback (most recent call last)\nError: x")
        result = mgr.check_response("已修复，测试通过。")
        assert result is not None
        assert "[R14]" in result

    def test_error_acknowledged(self):
        mgr = SelfCheckManager()
        mgr._loaded = True
        mgr.record_tool_result("terminal", "exit_code=1\nFAILED: test")
        result = mgr.check_response("上次命令失败了，分析原因是路径不对。")
        has_r14 = result and "[R14]" in result
        assert not has_r14

    def test_no_error_no_trigger(self):
        mgr = SelfCheckManager()
        mgr._loaded = True
        mgr.record_tool_result("terminal", "All tests passed")
        result = mgr.check_response("完成。")
        has_r14 = result and "[R14]" in result
        assert not has_r14


# ── Self-audit: canary testing for all rules ──────────────────────


class TestSelfAudit:
    """audit() should verify every rule is alive via canary tests."""

    def test_all_rules_alive(self):
        mgr = SelfCheckManager()
        report = mgr.audit()
        assert report["broken"] == 0, (
            "Broken rules detected: %s" % {
                k: v for k, v in report["status"].items() if v != "alive"
            }
        )
        assert report["alive"] == report["total"]
        assert report["total"] >= 10  # at least 10 rules (R09+R12 removed)

    def test_audit_detects_broken_rule(self):
        mgr = SelfCheckManager()
        import agent.self_check as sc_mod
        original = sc_mod._R07_SPECULATION_PATTERNS[:]
        sc_mod._R07_SPECULATION_PATTERNS = [
            (sc_mod.re.compile(r"THIS_WILL_NEVER_MATCH_ANYTHING"), "broken")
        ]
        try:
            report = mgr.audit()
            assert report["status"].get("R07") == "broken"
        finally:
            sc_mod._R07_SPECULATION_PATTERNS = original
