"""Consumption governors for autonomous goal loops.

Measured 2026-11-11 00:40-02:40: three open-ended goals ran in parallel, each turn issuing
dozens of model calls with ~150k-token contexts, and drained a 5h provider quota in ~2h
(1,090 zai calls / 118.5M input tokens). ``evaluate_after_turn`` is the single continuation
decision shared by the CLI and the gateway, so both governors live there:

* ``goals.max_api_calls`` — per-goal API-call budget anchored on the session's cumulative
  ``api_call_count``; over budget the goal parks with a visible reason (``/goal resume``
  re-anchors it).
* ``goals.max_concurrent`` — at most N active goals continue autonomously; the newest parks.

Both default ON (120 calls / 2 concurrent), 0 disables, and a ``done``/``blocked`` verdict
still wins: governors only gate continuation, never completion.
"""
import pytest


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    from pathlib import Path

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    from hermes_cli import goals

    goals._DB_CACHE.clear()
    yield home
    goals._DB_CACHE.clear()


@pytest.fixture(autouse=True)
def _stub_judge(monkeypatch):
    """No LLM in tests: the judge always says continue unless a test overrides it."""
    from hermes_cli import goals

    monkeypatch.setattr(
        goals, "judge_goal", lambda *a, **k: ("continue", "work remains", False, None, False)
    )


def _make_db(home):
    from hermes_state import SessionDB

    return SessionDB(db_path=home / "state.db")


def _new_session(db, sid):
    db.create_session(sid, source="test")
    return sid


def _bump_calls(db, sid, calls):
    db.queue_token_counts(sid, api_call_count=calls)
    # Exact-reader contract: the goals layer reads through a separate registry handle, so the
    # queued deltas must be drained to the DB before the evaluation runs.
    assert db.flush_token_counts()


def _set_goal(sid, **governors):
    from hermes_cli.goals import GoalManager

    return GoalManager(session_id=sid, **governors)


# ── API-call budget ──────────────────────────────────────────────────────────


def test_api_call_budget_parks_goal_at_limit(hermes_home):
    db = _make_db(hermes_home)
    sid = _new_session(db, "s-budget")
    mgr = _set_goal(sid, max_api_calls=3, max_concurrent=0)
    mgr.set("ship it")
    _bump_calls(db, sid, 3)

    decision = mgr.evaluate_after_turn("did some work")

    assert decision["status"] == "paused"
    assert decision["should_continue"] is False
    assert "API-call budget" in mgr.state.paused_reason
    assert "3/3" in mgr.state.paused_reason


def test_api_call_budget_below_limit_continues(hermes_home):
    db = _make_db(hermes_home)
    sid = _new_session(db, "s-under")
    mgr = _set_goal(sid, max_api_calls=3, max_concurrent=0)
    mgr.set("ship it")
    _bump_calls(db, sid, 1)

    decision = mgr.evaluate_after_turn("did some work")

    assert decision["should_continue"] is True
    assert mgr.state.status == "active"
    assert mgr.state.api_calls_baseline is not None


def test_zero_disables_api_call_budget(hermes_home):
    db = _make_db(hermes_home)
    sid = _new_session(db, "s-unlimited")
    mgr = _set_goal(sid, max_api_calls=0, max_concurrent=0)
    mgr.set("ship it")
    _bump_calls(db, sid, 10_000)

    decision = mgr.evaluate_after_turn("did some work")

    assert decision["should_continue"] is True


def test_default_budget_is_enabled_without_config(hermes_home):
    """No config.yaml: the guardrail is still on — an unattended goal cannot loop forever."""
    db = _make_db(hermes_home)
    sid = _new_session(db, "s-default")
    mgr = _set_goal(sid, max_concurrent=0)  # no explicit budget
    mgr.set("ship it")
    _bump_calls(db, sid, 120)

    decision = mgr.evaluate_after_turn("did some work")

    assert decision["status"] == "paused"
    assert "API-call budget" in mgr.state.paused_reason


def test_resume_reanchors_api_call_budget(hermes_home):
    db = _make_db(hermes_home)
    sid = _new_session(db, "s-resume")
    mgr = _set_goal(sid, max_api_calls=3, max_concurrent=0)
    mgr.set("ship it")
    _bump_calls(db, sid, 3)
    assert mgr.evaluate_after_turn("work")["status"] == "paused"

    _bump_calls(db, sid, 2)
    assert mgr.resume(reset_budget=True) is not None

    decision = mgr.evaluate_after_turn("more work")
    assert decision["should_continue"] is True


def test_legacy_goal_row_anchors_baseline_lazily(hermes_home):
    """Goals persisted before the governor existed carry no baseline: the first evaluation
    anchors to the session's current counter instead of counting all of history."""
    import json

    db = _make_db(hermes_home)
    sid = _new_session(db, "s-legacy")
    db.set_meta(f"goal:{sid}", json.dumps({
        "goal": "legacy work", "status": "active", "turns_used": 1, "max_turns": 20,
        "created_at": 1.0, "last_turn_at": 1.0,
    }))
    _bump_calls(db, sid, 500)

    mgr = _set_goal(sid, max_api_calls=3, max_concurrent=0)
    decision = mgr.evaluate_after_turn("work")

    assert decision["should_continue"] is True
    assert mgr.state.api_calls_baseline == 500


def test_config_yaml_overrides_budget(hermes_home):
    (hermes_home / "config.yaml").write_text(
        "goals:\n  max_api_calls: 2\n  max_concurrent: 1\n", encoding="utf-8"
    )
    db = _make_db(hermes_home)
    sid = _new_session(db, "s-config")
    mgr = _set_goal(sid)  # resolution happens inside GoalManager
    mgr.set("ship it")
    _bump_calls(db, sid, 2)

    decision = mgr.evaluate_after_turn("did some work")

    assert decision["status"] == "paused"
    assert "2/2" in mgr.state.paused_reason


# ── Concurrency cap ──────────────────────────────────────────────────────────


def test_concurrency_cap_parks_newest_goal(hermes_home):
    db = _make_db(hermes_home)
    for sid in ("g1", "g2", "g3"):
        _new_session(db, sid)
        _set_goal(sid, max_api_calls=0, max_concurrent=2).set(f"goal {sid}")

    newest = _set_goal("g3", max_api_calls=0, max_concurrent=2)
    decision = newest.evaluate_after_turn("work")
    assert decision["status"] == "paused"
    assert "concurrency" in newest.state.paused_reason

    oldest = _set_goal("g1", max_api_calls=0, max_concurrent=2)
    assert oldest.evaluate_after_turn("work")["should_continue"] is True


def test_concurrency_cap_zero_disables(hermes_home):
    db = _make_db(hermes_home)
    for sid in ("c1", "c2", "c3"):
        _new_session(db, sid)
        _set_goal(sid, max_api_calls=0, max_concurrent=0).set(f"goal {sid}")

    decision = _set_goal("c3", max_api_calls=0, max_concurrent=0).evaluate_after_turn("work")
    assert decision["should_continue"] is True


# ── Governors never steal a completion ────────────────────────────────────────


def test_done_verdict_wins_over_governors(hermes_home, monkeypatch):
    from hermes_cli import goals

    monkeypatch.setattr(
        goals, "judge_goal", lambda *a, **k: ("done", "criteria met", False, None, False)
    )
    db = _make_db(hermes_home)
    sid = _new_session(db, "s-done")
    mgr = _set_goal(sid, max_api_calls=1, max_concurrent=1)
    mgr.set("ship it")
    _bump_calls(db, sid, 50)

    decision = mgr.evaluate_after_turn("all done")

    assert decision["status"] == "done"
    assert mgr.state.status == "done"
