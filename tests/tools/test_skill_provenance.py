# -*- coding: utf-8 -*-
"""skill provenance（证据强度）+ curator 有效性回测单测。

根因背景：skill 沉淀"记录即完成"，无"证据强度"结构化承载（J-SPACE 三态
[实测]/[文档]/[推断]/[未查证] 在 skill 层缺失），也无"沉淀后是否真被检索
执行"的回测。机制：frontmatter 的 metadata.hermes.provenance.evidence
（observed/documented/inferred/unverified，缺省 unverified）作为数据，curator
candidate list 加 [NEVER-INHERITED] 信号（use=0 且 view=0 且 evidence=unverified）
作为回测输出。

期望值独立推导（不读实现凑数）：
1. evidence 缺省 unverified——新 skill 未声明证据不得冒充 observed
2. 非法 evidence 值回退 unverified——只认四态
3. use=0 且 view=0 且 evidence=unverified → [NEVER-INHERITED]
4. use>0 或 evidence≠unverified → 无 [NEVER-INHERITED] 标记
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest


@pytest.fixture
def skills_home(tmp_path: Any, monkeypatch: Any) -> Any:
    """Isolated HERMES_HOME with a clean skills/ dir for each test (mirrors test_skill_usage.py)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "skills").mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    import importlib

    import tools.skill_usage as mod

    importlib.reload(mod)
    monkeypatch.setattr(mod, "_prune_builtins_enabled", lambda: False)
    return home


class TestExtractSkillProvenance:
    def test_empty_frontmatter_defaults_unverified(self):
        from agent.skill_utils import extract_skill_provenance

        prov = extract_skill_provenance({})
        assert prov == {"source": "", "evidence": "unverified", "applies_when": ""}

    def test_declared_evidence_parsed(self):
        from agent.skill_utils import extract_skill_provenance

        fm = {"metadata": {"hermes": {"provenance": {
            "source": "agent-session", "evidence": "observed", "applies_when": "goal 验收",
        }}}}
        prov = extract_skill_provenance(fm)
        assert prov["source"] == "agent-session"
        assert prov["evidence"] == "observed"
        assert prov["applies_when"] == "goal 验收"

    def test_invalid_evidence_falls_back_to_unverified(self):
        from agent.skill_utils import extract_skill_provenance

        fm = {"metadata": {"hermes": {"provenance": {"evidence": "garbage"}}}}
        assert extract_skill_provenance(fm)["evidence"] == "unverified"

    def test_case_insensitive_evidence(self):
        from agent.skill_utils import extract_skill_provenance

        fm = {"metadata": {"hermes": {"provenance": {"evidence": "OBSERVED"}}}}
        assert extract_skill_provenance(fm)["evidence"] == "observed"

    def test_malformed_provenance_dict_treated_as_empty(self):
        from agent.skill_utils import extract_skill_provenance

        assert extract_skill_provenance({"metadata": {"hermes": {"provenance": "not-a-dict"}}})["evidence"] == "unverified"


class TestSkillEvidence:
    @staticmethod
    def _write_provenance_skill(skills_dir: Any, name: str, evidence: str) -> Any:
        from pathlib import Path

        d = Path(skills_dir) / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: test\nmetadata:\n  hermes:\n"
            f"    provenance:\n      evidence: {evidence}\n---\n\n# body\n",
            encoding="utf-8",
        )
        return d

    def test_reads_declared_evidence(self, skills_home):
        from tools import skill_usage

        self._write_provenance_skill(skills_home / "skills", "obs-skill", "observed")
        assert skill_usage.skill_evidence("obs-skill") == "observed"

    def test_missing_skill_defaults_unverified(self, skills_home):
        from tools import skill_usage

        assert skill_usage.skill_evidence("no-such-skill") == "unverified"

    def test_missing_evidence_field_defaults_unverified(self, skills_home):
        from pathlib import Path

        from tools import skill_usage

        d = Path(skills_home) / "skills" / "plain-skill"
        d.mkdir(parents=True, exist_ok=True)
        (d / "SKILL.md").write_text("---\nname: plain-skill\ndescription: test\n---\n\n# body\n", encoding="utf-8")
        assert skill_usage.skill_evidence("plain-skill") == "unverified"


class TestCandidateListNeverInherited:
    @staticmethod
    def _rows() -> List[Dict[str, Any]]:
        return [
            {"name": "dead-lesson", "provenance": "agent", "state": "active", "pinned": False,
             "activity_count": 0, "use_count": 0, "view_count": 0, "patch_count": 0, "last_activity_at": None},
            {"name": "used-lesson", "provenance": "agent", "state": "active", "pinned": False,
             "activity_count": 3, "use_count": 3, "view_count": 0, "patch_count": 0,
             "last_activity_at": "2026-09-15T00:00:00+00:00"},
            {"name": "observed-but-unused", "provenance": "agent", "state": "active", "pinned": False,
             "activity_count": 0, "use_count": 0, "view_count": 0, "patch_count": 0, "last_activity_at": None},
        ]

    @staticmethod
    def _evidence(name: str) -> str:
        return {"dead-lesson": "unverified", "used-lesson": "unverified",
                "observed-but-unused": "observed"}[name]

    def test_never_inherited_flag_only_on_dead_skill(self, monkeypatch):
        import agent.curator as curator

        monkeypatch.setattr(curator.skill_usage, "curated_report", self._rows)
        monkeypatch.setattr(curator.skill_usage, "skill_evidence", self._evidence)
        monkeypatch.setattr(curator, "_cron_referenced_skills", lambda: set())

        out = curator._render_candidate_list()
        assert "[NEVER-INHERITED]" in out
        # 标记只挂在 dead-lesson 上，不挂在 used/observed 上
        dead_line = next(ln for ln in out.splitlines() if ln.startswith("- dead-lesson"))
        used_line = next(ln for ln in out.splitlines() if ln.startswith("- used-lesson"))
        obs_line = next(ln for ln in out.splitlines() if ln.startswith("- observed-but-unused"))
        assert "[NEVER-INHERITED]" in dead_line
        assert "[NEVER-INHERITED]" not in used_line
        assert "[NEVER-INHERITED]" not in obs_line

    def test_evidence_column_present(self, monkeypatch):
        import agent.curator as curator

        monkeypatch.setattr(curator.skill_usage, "curated_report", self._rows)
        monkeypatch.setattr(curator.skill_usage, "skill_evidence", self._evidence)
        monkeypatch.setattr(curator, "_cron_referenced_skills", lambda: set())

        out = curator._render_candidate_list()
        assert "evidence=unverified" in out
        assert "evidence=observed" in out

    def test_no_rows_message(self, monkeypatch):
        import agent.curator as curator

        monkeypatch.setattr(curator.skill_usage, "curated_report", lambda: [])
        assert curator._render_candidate_list() == "No curator-managed skills to review."
