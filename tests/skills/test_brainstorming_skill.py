"""The bundled brainstorming workflow is discoverable through the real loader."""

import json
from pathlib import Path

from tools import skills_tool


REPO = Path(__file__).resolve().parents[2]


def test_brainstorming_loads_from_bundled_tree(monkeypatch):
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", REPO / "skills")
    skills_tool._SKILLS_CACHE.clear()

    payload = json.loads(skills_tool.skill_view("brainstorming"))

    assert payload["success"], payload
    assert payload["name"] == "brainstorming"

    listing = json.loads(skills_tool.skills_list(query="architecture code"))
    assert any(skill["name"] == "brainstorming" for skill in listing["skills"])
