"""Canonical Hermes-home identity comparisons used by gateway status records."""
from __future__ import annotations

import os
from pathlib import Path

from hermes_constants import hermes_home_key


def canonical_hermes_home(path: Path | str) -> Path:
    """Stable absolute HERMES_HOME path for persisted identity data."""
    return Path(hermes_home_key(Path(path).expanduser()))


def same_hermes_home(left: Path | str, right: Path | str) -> bool:
    """Compare HERMES_HOME paths with the host platform's case semantics."""
    left_c = os.path.normcase(str(canonical_hermes_home(left)))
    return left_c == os.path.normcase(str(canonical_hermes_home(right)))
