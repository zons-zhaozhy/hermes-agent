"""The bundled verification-before-completion workflow is discoverable."""

import json
from pathlib import Path

from tools import skills_tool


REPO = Path(__file__).resolve().parents[2]


def test_verification_before_completion_loads_from_bundled_tree(monkeypatch):
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", REPO / "skills")
    skills_tool._SKILLS_CACHE.clear()

    payload = json.loads(skills_tool.skill_view("verification-before-completion"))

    assert payload["success"], payload
    assert payload["name"] == "verification-before-completion"

    listing = json.loads(skills_tool.skills_list(query="completion fresh output"))
    assert any(skill["name"] == "verification-before-completion" for skill in listing["skills"])
