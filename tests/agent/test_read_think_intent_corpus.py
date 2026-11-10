"""Measured accuracy contract for the read-think gate's user-intent classifier.

The gate decides whether a turn may write by classifying the user's latest
message. Until this corpus existed that decision was only covered by
hand-written per-sentence tests, which pin "these sentences must classify this
way" but cannot answer "how often is it wrong, and which way". A sentence
chosen by the author is exactly the class of input the classifier already
handles; the failures live in the long tail nobody thought to write down.

## The label standard

The labels are not intuitions about what the user meant; they are a consequence
of the one question the gate asks: **would a write in this turn be the right
response?** Three rules, applied in order, first match wins. Each corpus row
carries the `rule` that produced its label, so a label can be audited against
the standard instead of re-argued row by row.

* ``R1`` **nothing to act on** — interjections, praise, acknowledgements, bare
  fragments (「加油」「好的知道了」「改」). Nothing is asked and nothing is
  refused: **ambiguous**.
* ``R2`` **a task is handed over** — a named change (「把 X 改成 180」), a
  proposal confirmation (「可以」), a bare operational directive (「提交推送」),
  or the decision delegated to the agent (「你决定」). A named change is
  **modify**; delegation that never says what to change is **ambiguous**.
* ``R3`` **speech only** — asking (fact, cause, method, feasibility), asking for
  a plan (including "we'll discuss before doing"), or stating an observation.
  A write here would be unrequested even after a complete investigation, so
  **read_only**. Explicit prohibition (「不要改」) is the case the user words.

The distinction between ``modify`` and ``ambiguous`` is annotation, not
behaviour: both let the write proceed through the investigation contract, and
the difference only shows up as session stickiness. The gate's decision is
therefore binary — **refuse or allow** — and that is the metric this file
defends. Three-way accuracy is a secondary guard against label drift.

## Why two layers

* **The deterministic fallback** runs when the semantic classifier is
  unavailable. It may refuse only on evidence a table can read (an explicit
  prohibition) and may grant on an imperative, because a deserved refusal
  deadlocks the whole turn while a grant only costs an investigation.
* **The semantic classifier** is an LLM call, so its verdicts are replayed from
  ``read_think_intent_recorded.jsonl`` (regenerate with
  ``scripts/regen_read_think_intent_fixture.py``; the recording is a reviewable
  artifact, so an accuracy change is a commit diff rather than a drifting
  number).

## The consequence that matters

The gate blocks a whole turn when it predicts ``read_only``. A wrong refusal is
a deadlock the user has to retype their way out of; a wrong ``modify`` only
costs an unrequested investigation they can still see and stop. 「加油」 was
refused four times in one turn before this corpus existed.
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

# What each rule is allowed to produce. A row whose label contradicts its own
# rule is a mislabelled row, not a classifier failure — this is what keeps the
# standard from drifting one arguable row at a time.
_RULE_LABELS = {
    "R1": frozenset({INTENT_AMBIGUOUS}),
    "R2": frozenset({INTENT_MODIFY, INTENT_AMBIGUOUS}),
    "R3": frozenset({INTENT_READ_ONLY}),
}


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.fixture(scope="module")
def corpus() -> list[dict]:
    return _read_jsonl(CORPUS)


@pytest.fixture(scope="module")
def recorded() -> dict[str, dict]:
    return {row["id"]: row for row in _read_jsonl(RECORDED)}


def _verdict(recorded: dict[str, dict], row_id: str, text: str,
             context: list[dict] | None = None) -> str:
    """The intent production would resolve for this message.

    A semantic verdict that is unusable (no answer, or the contradictory
    both-true the prompt forbids) is not an outcome of its own: the gate falls
    back to the deterministic layer, so the corpus scores that layer instead of
    pretending the row was never classified.
    """
    row = recorded[row_id]
    semantic = _intent_from_labels(row["read_only"], row["modify"])
    if semantic is not None:
        return semantic
    return _fallback_detect_intent(text, context or [])


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


def test_every_row_cites_a_rule_that_can_produce_its_label(corpus) -> None:
    """The standard is the contract; a label that breaks its own rule is a bug."""
    for row in corpus:
        rule = row.get("rule")
        assert rule in _RULE_LABELS, (row["id"], rule)
        assert row["label"] in _RULE_LABELS[rule], (row["id"], rule, row["label"])


def test_corpus_covers_every_rule_and_class(corpus) -> None:
    """Each rule and each class must stay represented, or a whole arm goes dark."""
    assert {row["label"] for row in corpus} == set(LABELS)
    assert {row["rule"] for row in corpus} == set(_RULE_LABELS)
    for label in LABELS:
        assert sum(1 for row in corpus if row["label"] == label) >= 10


def test_no_message_that_asked_for_work_is_refused(corpus, recorded) -> None:
    """The failure this corpus exists for: a write the user asked for, refused.

    A refusal deadlocks the turn, so this set is empty and a new entry fails the
    test outright — no ratchet, because the baseline is zero.
    """
    false_blocks = [
        (row["id"], row["text"])
        for row in corpus
        if row["label"] != INTENT_READ_ONLY
        and _verdict(recorded, row["id"], row["text"], row.get("context") or []) == INTENT_READ_ONLY
    ]
    assert false_blocks == []


def test_every_speech_only_message_is_refused(corpus, recorded) -> None:
    """The other direction: an unrequested write is what the gate exists to stop."""
    missed = [
        (row["id"], row["text"])
        for row in corpus
        if row["label"] == INTENT_READ_ONLY and _verdict(recorded, row["id"], row["text"], row.get("context") or []) != INTENT_READ_ONLY
    ]
    assert missed == []


def test_three_way_accuracy_floor(corpus, recorded) -> None:
    """Secondary guard against label drift; behaviour is the binary decision."""
    pairs = [(row["label"], _verdict(recorded, row["id"], row["text"], row.get("context") or [])) for row in corpus]
    assert all(predicted is not None for _, predicted in pairs)
    counts = _counts(pairs)  # type: ignore[arg-type]
    correct = sum(counts.get((label, label), 0) for label in LABELS)
    assert correct / len(pairs) >= 0.90, counts


def test_each_class_stays_usable(corpus, recorded) -> None:
    """Per-class recall/precision, because the aggregate hides a dead class."""
    pairs = [(row["label"], _verdict(recorded, row["id"], row["text"], row.get("context") or [])) for row in corpus]
    counts = _counts(pairs)  # type: ignore[arg-type]
    assert _recall(counts, INTENT_READ_ONLY) >= 0.98, counts
    assert _precision(counts, INTENT_READ_ONLY) >= 0.98, counts
    assert _recall(counts, INTENT_MODIFY) >= 0.80, counts
    assert _precision(counts, INTENT_MODIFY) >= 0.90, counts


def test_fallback_never_refuses_a_request_that_was_not_read_only(corpus) -> None:
    """The fallback's only safe refusal is an explicit prohibition."""
    wrong = [
        (row["id"], _fallback_detect_intent(row["text"], row.get("context") or []))
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
    """Encouragement must never be read as authorization, nor as a refusal."""
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


def test_contradictory_labels_are_not_silently_resolved() -> None:
    """Both keys true is the prompt's forbidden answer; picking a side would hide it."""
    assert _intent_from_labels(True, True) is None
    assert _intent_from_labels(None, True) is None
    assert _intent_from_labels(False, False) == INTENT_AMBIGUOUS
