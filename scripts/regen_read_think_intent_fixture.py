#!/usr/bin/env python3
"""Regenerate the recorded verdicts behind the read-think gate intent corpus.

The gate's semantic intent classifier is an LLM call, so its accuracy cannot be
measured in CI without a recording: `tests/agent/test_read_think_intent_corpus.py`
replays ``read_think_intent_recorded.jsonl`` instead of reaching the network.

This script rebuilds that recording from the labelled corpus. Run it by hand
when the prompt (``_INTENT_CLASSIFY_PROMPT``) or the corpus changes, then commit
the diff — the recorded verdicts are a reviewable artifact, and any accuracy
shift shows up as a changed commit rather than a drifting CI number.

    .venv/bin/python scripts/regen_read_think_intent_fixture.py

Contract:
    Preconditions: the corpus fixture exists; HERMES_HOME points at a home whose
                   ``auxiliary`` model resolves (the real one by default).
    Postconditions: ``read_think_intent_recorded.jsonl`` holds one row per corpus
                    row, keyed by the corpus id, carrying the raw per-key verdicts
                    plus the probability-free source channel. Rows the judge could
                    not answer are recorded as nulls, never dropped: a run that
                    leaves nulls is a failed recording, and the test says so.
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CORPUS = ROOT / "tests" / "agent" / "fixtures" / "read_think_intent_corpus.jsonl"
RECORDED = ROOT / "tests" / "agent" / "fixtures" / "read_think_intent_recorded.jsonl"
WORKERS = 4


def _load_corpus() -> list[dict]:
    rows = []
    for line in CORPUS.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _record(row: dict) -> dict:
    from plugins._llm_judge import llm_judge_multi
    from plugins import read_think_gate_host as host

    gate = host._gate  # the loaded implementation module: same prompt/context as runtime
    context = gate._build_intent_context(row["text"], row.get("context") or [])
    verdict = llm_judge_multi(
        task="read_think_intent",
        system=gate._INTENT_CLASSIFY_PROMPT,
        text=context,
        keys=["read_only", "modify"],
    )
    return {
        "id": row["id"],
        "text": row["text"],
        "label": row["label"],
        "read_only": verdict.get("read_only"),
        "modify": verdict.get("modify"),
    }


RETRY_PAUSE_SECONDS = 120


def _unanswered(recorded: list[dict]) -> list[dict]:
    return [row for row in recorded if row["read_only"] is None and row["modify"] is None]


def _refused(recorded: list[dict]) -> list[dict]:
    """Rows the endpoint never answered (as opposed to a neutral "both false")."""
    return _unanswered(recorded)


def main() -> int:
    rows = _load_corpus()
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        recorded = list(pool.map(_record, rows))

    # A judge outage answers nothing for a whole batch. Retry once after the
    # endpoint circuit breaker has had time to close, rather than recording a
    # hole: a half-recorded fixture would silently measure the missing rows as
    # "no verdict" and quietly deflate every metric.
    for attempt in (1, 2):
        missing = _refused(recorded)
        if not missing:
            break
        print(f"attempt {attempt}: {len(missing)} rows unanswered; retrying in {RETRY_PAUSE_SECONDS}s")
        time.sleep(RETRY_PAUSE_SECONDS)
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            retried = {row["id"]: row for row in pool.map(_record, missing)}
        recorded = [retried.get(row["id"], row) for row in recorded]

    body = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in recorded)
    unanswered = [row["id"] for row in _refused(recorded)]
    if unanswered:
        # Never clobber a good recording with a hole: keep the served rows for
        # inspection under .partial, leave the committed fixture untouched.
        partial = RECORDED.with_suffix(".partial.jsonl")
        partial.write_text(body, encoding="utf-8")
        print(f"FAILED: judge answered neither key for {len(unanswered)} rows "
              f"({', '.join(unanswered[:10])}{'…' if len(unanswered) > 10 else ''}); "
              f"wrote {partial.relative_to(ROOT)} and left {RECORDED.name} untouched")
        return 1
    RECORDED.write_text(body, encoding="utf-8")
    print(f"recorded {len(recorded)} rows -> {RECORDED.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
