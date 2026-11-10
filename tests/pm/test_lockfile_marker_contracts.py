"""Platform-marker contracts for platform-gated optional dependencies."""

import tomllib
from pathlib import Path

import pytest
from packaging.markers import Marker


ROOT = Path(__file__).resolve().parents[2]

# uv.lock markers decide which wheels an extra installs per platform, and a
# relock can rewrite them while `uv lock --check` still passes (2026-11-10:
# av/faster-whisper flipped from "Android excluded" to "Android included").
# The heavyweight voice surface is therefore pinned explicitly here.
PLATFORM_CASES = [
    ("macos-arm64", {"sys_platform": "darwin", "platform_machine": "arm64"}, True),
    ("macos-x86_64", {"sys_platform": "darwin", "platform_machine": "x86_64"}, False),
    ("windows-x86_64", {"sys_platform": "win32", "platform_machine": "x86_64"}, True),
    ("linux-aarch64", {"sys_platform": "linux", "platform_machine": "aarch64"}, True),
    ("android-aarch64", {"sys_platform": "android", "platform_machine": "aarch64"}, True),
]

VOICE_EXTRAS = ("voice", "stt-whisper")
VOICE_PACKAGES = ("av", "faster-whisper")


def _root_optional_dependencies() -> dict[str, list[dict]]:
    """Return the root project's locked extras from uv.lock."""

    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    root = next(package for package in lock["package"] if package["name"] == "hermes-agent")
    return root["optional-dependencies"]


@pytest.mark.parametrize(
    ("label", "environment", "expected"),
    PLATFORM_CASES,
    ids=[case[0] for case in PLATFORM_CASES],
)
@pytest.mark.parametrize("extra", VOICE_EXTRAS)
@pytest.mark.parametrize("package", VOICE_PACKAGES)
def test_voice_extra_platform_surface_is_pinned(
    package: str, extra: str, label: str, environment: dict, expected: bool
) -> None:
    """av/faster-whisper install exactly on the platforms the lock now admits."""

    entries = _root_optional_dependencies()[extra]
    entry = next(item for item in entries if item["name"] == package)

    assert Marker(entry["marker"]).evaluate(environment) is expected, (
        f"{extra}/{package} on {label} drifted from the pinned platform surface"
    )
