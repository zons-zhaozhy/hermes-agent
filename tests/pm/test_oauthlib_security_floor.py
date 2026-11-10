"""Security floor for the Google OAuth dependency closure."""

import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version


ROOT = Path(__file__).resolve().parents[2]


def test_google_extra_carries_patched_oauthlib_floor() -> None:
    """Unlocked Google installs cannot fall back to vulnerable oauthlib."""

    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    google = project["project"]["optional-dependencies"]["google"]
    requirements = {Requirement(item).name: Requirement(item) for item in google}

    oauthlib = requirements["oauthlib"]
    assert oauthlib.specifier.contains("4.0.1")
    assert not oauthlib.specifier.contains("5.0.0")


def test_locked_google_closure_resolves_patched_oauthlib() -> None:
    """uv.lock must agree with the declared Google OAuth security floor."""

    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    google = project["project"]["optional-dependencies"]["google"]
    oauthlib = next(Requirement(item) for item in google if Requirement(item).name == "oauthlib")

    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    locked = next(package for package in lock["package"] if package["name"] == "oauthlib")
    locked_version = Version(locked["version"])

    assert locked_version in oauthlib.specifier
    assert locked_version >= Version("4.0.1")
