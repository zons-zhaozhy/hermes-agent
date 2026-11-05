"""The bundled receiving-code-review workflow is discoverable."""

import json
from pathlib import Path

from tools import skills_tool


REPO = Path(__file__).resolve().parents[2]


def test_receiving_code_review_loads_from_bundled_tree(monkeypatch):
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", REPO / "skills")
    skills_tool._SKILLS_CACHE.clear()

    payload = json.loads(skills_tool.skill_view("receiving-code-review"))

    assert payload["success"], payload
    assert payload["name"] == "receiving-code-review"

    listing = json.loads(skills_tool.skills_list(query="review feedback code"))
    assert any(skill["name"] == "receiving-code-review" for skill in listing["skills"])
