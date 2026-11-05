"""The bundled executing-plans workflow is discoverable through the real loader."""

import json
from pathlib import Path

from tools import skills_tool


REPO = Path(__file__).resolve().parents[2]


def test_executing_plans_loads_from_bundled_tree(monkeypatch):
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", REPO / "skills")
    skills_tool._SKILLS_CACHE.clear()

    payload = json.loads(skills_tool.skill_view("executing-plans"))

    assert payload["success"], payload
    assert payload["name"] == "executing-plans"

    listing = json.loads(skills_tool.skills_list(query="written plans checkpoints evidence"))
    assert any(skill["name"] == "executing-plans" for skill in listing["skills"])
