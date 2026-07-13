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
