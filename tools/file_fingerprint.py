"""Content fingerprints for read-before-write freshness checks.

Hashline-style protocol (adapted from oh-my-pi's hashline, Apache-2.0):
``read_file`` stamps every successful read with a short content hash; the
model echoes it back on ``patch`` as ``expected_fingerprint``; the writer
rejects the edit when the file on disk no longer matches — catching
"edited from a stale mental copy" BEFORE the fuzzy matcher can graft a
patch onto the wrong content.

Contract:
  Preconditions:  path exists and is readable text (callers check first).
  Postconditions: return is a stable 12-hex string — same bytes always
                  hash identically; line-ending/BOM normalization is
                  applied so editor re-serialization never false-rejects.
"""
from __future__ import annotations

import hashlib

_FP_LEN = 12


def content_fingerprint(text: str) -> str:
    """Stable short hash of normalized file content.

    Normalization: strip a leading UTF-8 BOM, unify CRLF/CR to LF — the
    same normalization patch_replace applies before matching, so the
    fingerprint tracks the bytes the writer actually compares against.
    """
    if text.startswith("\ufeff"):
        text = text[1:]
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:_FP_LEN]


def fingerprint_matches(text: str, expected: str) -> bool:
    """True when ``expected`` is the fingerprint of ``text``'s current content."""
    return content_fingerprint(text) == expected
