"""Measured accuracy contract for the read-think gate's user-intent classifier.

The gate decides whether a turn may write by classifying the user's latest
message. Until this corpus existed that decision was only covered by
hand-written per-sentence tests, which pin "these sentences must classify this
way" but cannot answer "how often is it wrong, and which way". A sentence
chosen by the author is exactly the class of input the classifier already
handles; the failures live in the long tail nobody thought to write down.

Two layers are measured separately, because they have different failure modes:

* **The deterministic fallback** runs when the semantic classifier is
  unavailable. It may only refuse on evidence a table can actually read (an
  explicit prohibition), so its contract is: never refuse a request that was
  not read-only, and never miss an explicit prohibition.
* **The semantic classifier** is an LLM call, so its verdicts are replayed from
  ``read_think_intent_recorded.jsonl`` (regenerate with
  ``scripts/regen_read_think_intent_fixture.py``). The recording is a
  reviewable artifact: an accuracy change shows up as a commit diff instead of
  a drifting number.

The consequence that matters is the asymmetric one. The gate blocks a whole
turn when it predicts ``read_only``, so a wrong ``read_only`` is a deadlock the
user has to retype their way out of, while a wrong ``modify`` only costs an
unrequested investigation. Every row in ``_KNOWN_FALSE_BLOCKS`` is a message
that asked for work and would have been refused.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from plugins.read_think_gate_host import (
    INTENT_AMBIGUOUS,
    INTENT_MODIFY,
    INTENT_READ_ONLY,
    _fallback_detect_intent,
    _intent_from_labels,
)

_FIXTURES = Path(__file__).resolve().parent / "fixtures"
CORPUS = _FIXTURES / "read_think_intent_corpus.jsonl"
RECORDED = _FIXTURES / "read_think_intent_recorded.jsonl"

LABELS = (INTENT_READ_ONLY, INTENT_MODIFY, INTENT_AMBIGUOUS)

# Messages the classifier reads as read_only while the user was asking for a
# change. Each entry is a turn the gate would have deadlocked. This set may only
# shrink: a new id is a regression, and removing one means the fix landed and
# the baseline moved with it.
_KNOWN_FALSE_BLOCKS = frozenset(
    {
        "amb-vague-11",   # 「it feels slow」— 只陈述现象
        "amb-mixed-02",   # 「看看这个怎么改比较好，改完再说」— 先要方案，动作推迟
        "amb-phatic-06",  # 「我先看看」— 用户自述行为
    }
)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.fixture(scope="module")
def corpus() -> list[dict]:
    return _read_jsonl(CORPUS)


@pytest.fixture(scope="module")
def recorded() -> dict[str, dict]:
    return {row["id"]: row for row in _read_jsonl(RECORDED)}


def _counts(pairs: list[tuple[str, str]]) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    for truth, predicted in pairs:
        counts[(truth, predicted)] = counts.get((truth, predicted), 0) + 1
    return counts


def _recall(counts, label: str) -> float:
    support = sum(counts.get((label, other), 0) for other in LABELS)
    return counts.get((label, label), 0) / support if support else 1.0


def _precision(counts, label: str) -> float:
    predicted = sum(counts.get((other, label), 0) for other in LABELS)
    return counts.get((label, label), 0) / predicted if predicted else 1.0


def test_recording_covers_every_corpus_row(corpus, recorded) -> None:
    """A stale recording silently measures an older corpus, so the ids must match."""
    assert set(recorded) == {row["id"] for row in corpus}


def test_corpus_covers_the_decisions_the_gate_acts_on(corpus) -> None:
    """Both refusal-relevant classes and the undecided class must be represented."""
    labels = {row["label"] for row in corpus}
    assert labels == set(LABELS)
    for label in LABELS:
        assert sum(1 for row in corpus if row["label"] == label) >= 20


def test_fallback_never_refuses_a_request_that_was_not_read_only(corpus) -> None:
    """The fallback's only safe refusal is an explicit prohibition."""
    wrong = [
        (row["id"], row["label"], _fallback_detect_intent(row["text"], row.get("context") or []))
        for row in corpus
        if _fallback_detect_intent(row["text"], row.get("context") or []) == INTENT_READ_ONLY
        and row["label"] != INTENT_READ_ONLY
    ]
    assert wrong == []


def test_fallback_refuses_every_explicit_prohibition(corpus) -> None:
    """A user who wrote "do not modify" must never be talked into a write."""
    missed = [
        row["id"]
        for row in corpus
        if any(
            term in row["text"].lower()
            for term in ("不要修改", "别改", "别修", "只分析", "不要创建", "do not modify", "read only")
        )
        and _fallback_detect_intent(row["text"]) != INTENT_READ_ONLY
    ]
    assert missed == []


def test_fallback_treats_an_interjection_as_undecided(corpus) -> None:
    """Encouragement must never be read as authorization, nor as a refusal.

    「加油」 was blocked four times in one turn: no table row matched it, the
    undecided class was still a refusal, and nothing bounded the retry. An
    interjection is exactly the input a verb table cannot read, so the fallback
    leaves it undecided and lets the investigation contract decide.
    """
    phatic = [row for row in corpus if row["tag"] == "phatic"]
    assert phatic, "the corpus must keep covering interjections"
    for row in phatic:
        verdict = _fallback_detect_intent(row["text"], row.get("context") or [])
        assert verdict == INTENT_AMBIGUOUS, (row["id"], verdict)


def test_fallback_still_grants_on_an_imperative(corpus) -> None:
    """Permissive by design: granting costs an investigation, refusing costs the turn."""
    granted = [
        row["id"]
        for row in corpus
        if row["label"] == INTENT_MODIFY
        and _fallback_detect_intent(row["text"], row.get("context") or []) == INTENT_MODIFY
    ]
    assert len(granted) >= 10, granted


def test_semantic_classifier_accuracy_floor(corpus, recorded) -> None:
    """The replayed verdicts must keep the measured accuracy it was recorded at."""
    pairs = [
        (row["label"], _intent_from_labels(recorded[row["id"]]["read_only"], recorded[row["id"]]["modify"]))
        for row in corpus
    ]
    assert all(predicted is not None for _, predicted in pairs)
    counts = _counts(pairs)  # type: ignore[arg-type]
    correct = sum(counts.get((label, label), 0) for label in LABELS)
    assert correct / len(pairs) >= 0.90, counts


def test_semantic_classifier_keeps_each_class_usable(corpus, recorded) -> None:
    """Per-class recall/precision, because the aggregate hides a dead class."""
    pairs = [
        (row["label"], _intent_from_labels(recorded[row["id"]]["read_only"], recorded[row["id"]]["modify"]))
        for row in corpus
    ]
    counts = _counts(pairs)  # type: ignore[arg-type]
    assert _recall(counts, INTENT_READ_ONLY) >= 0.90, counts
    assert _recall(counts, INTENT_MODIFY) >= 0.98, counts
    assert _precision(counts, INTENT_READ_ONLY) >= 0.92, counts
    assert _precision(counts, INTENT_MODIFY) >= 0.90, counts


def test_false_blocks_do_not_grow(corpus, recorded) -> None:
    """Rows the gate would have refused while the user was asking for work."""
    false_blocks = {
        row["id"]
        for row in corpus
        if _intent_from_labels(recorded[row["id"]]["read_only"], recorded[row["id"]]["modify"])
        == INTENT_READ_ONLY
        and row["label"] != INTENT_READ_ONLY
    }
    assert false_blocks <= _KNOWN_FALSE_BLOCKS, sorted(false_blocks - _KNOWN_FALSE_BLOCKS)


def test_contradictory_labels_are_not_silently_resolved() -> None:
    """Both keys true is the prompt's forbidden answer; picking a side would hide it."""
    assert _intent_from_labels(True, True) is None
    assert _intent_from_labels(None, True) is None
    assert _intent_from_labels(False, False) == INTENT_AMBIGUOUS
