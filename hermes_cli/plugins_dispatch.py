"""Plugin hook / middleware / event-bus / system-prompt-section dispatch.

Mixed into :class:`hermes_cli.plugins.PluginManager`. ``_resolve_hook_callback_timeout`` stays on
the origin (tests patch it there) and is looked up lazily.
"""

from __future__ import annotations

import asyncio
import contextvars
import copy
import inspect
import logging
import queue
import re
import threading
import time
import types
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Set, Union

from hermes_cli.gate_plane import GATE_EVIDENCE_HOOKS, GateStatus, result_status
from hermes_cli.middleware import OBSERVER_SCHEMA_VERSION

logger = logging.getLogger("hermes_cli.plugins")

# Allowlist of agent-turn hot-path hooks bounded by plugins.hook_callback_timeout (fail-open:
# abandon without join — joining reintroduced a shutdown hang). Unlisted hooks run synchronously.
# Intentionally unbounded: on_session_finalize/reset (last-chance flush — abandon can lose state);
# subagent_start (observer); pre_gateway_dispatch (policy gate — neither fail mode is acceptable);
# pre/post_approval_* (approval UX has its own timeout); kanban_* (own heartbeat/stale reclaim).
# The goal is to stop a hung Python plugin callback from wedging the conversation loop (#76821) without
# joining the worker (avoids the #6622 ThreadPoolExecutor shutdown hang). Hooks not listed below run
# synchronously to completion. (on_session_start/end stay bounded — they sit on the common session-boundary
# path.) - subagent_start — observer only; blocking delegation belongs in pre_tool_call. Lower frequency
# than tool/LLM hooks. Abandoning is unsafe either way (fail-open skips auth-like checks; fail-closed can
# drop legitimate messages). Prefer finish-or-exception fallthrough. - pre_approval_request /
# post_approval_response — observers only (cannot veto); the approval UX already has its own timeout; not on
# the tool loop hot path. - kanban_task_* — fire after the board DB commit, observers only, in
# dispatcher/worker processes; kanban has its own heartbeat/stale reclaim. Abandon-without-join also leaves
# a daemon thread that may still mutate shared state — safer for value-returning observers than for
# gates/flushes.
_HOOK_TIMEOUT_BOUNDED_HOOKS: set[str] = {
    "post_tool_call", "transform_terminal_output", "transform_tool_result", "transform_llm_output",
    "pre_llm_call", "post_llm_call", "pre_api_request", "post_api_request", "api_request_error",
    "pre_auxiliary_call", "post_auxiliary_call", "pre_verify", "on_session_start", "on_session_end",
    # Fail-open consumer on every inbound gateway message: a hung plugin must not stall the profile.
    "post_gateway_admission",
    # Observer on every local session's commands, fired by the gateway (gateway/session_commands.py).
    "pre_command",
    # Fires inside the compression commit window (lease held): a hung plugin must not wedge the session.
    "post_compaction",
}
# Hooks whose calls are told apart by another payload field: one session's commands are one
# sequence, so a still-running callback skips that session's next command, not another session's.
_HOOK_CALL_IDENTITY_FIELDS: dict[str, tuple[str, ...]] = {
    "pre_command": ("session_key",), "post_compaction": ("session_id",),
}

# Policy hooks: timeout / still-running are not a verdict, so they go through
# ``_HOOK_TIMEOUT_DEGRADE_HOOKS`` below before any directive is produced. A hook listed here either
# fails the unit of work closed or degrades to the callback's own no-verdict contract.
_HOOK_TIMEOUT_FAIL_CLOSED_HOOKS: set[str] = {
    "pre_tool_call", "pre_tool_batch", "pre_llm_call", "pre_verify",
}
# Of those, the ones that decide ONCE per unit of user-visible work and whose timeout therefore
# costs that whole unit: a turn (``pre_llm_call``) or a completion claim (``pre_verify``).
# Only a VETO-GRADE callback here earns the automatic retry below. The per-tool-call gates do not:
# they fail closed into an error result the agent can adapt to, and retrying would add another
# full budget to every blocked tool call for the length of an outage.
_HOOK_TIMEOUT_RETRY_HOOKS: set[str] = {"pre_llm_call", "pre_verify"}
# Hooks whose timeout must NOT become a veto by itself (``_hook_timeout_directive``). Their
# callbacks' normal output is additive — a context injection — and every registration in this
# tree already answers "no verdict" with fail-open (``guards.scope_gate``: 「判定失败或不可用 →
# 放行」, ``declare_act_guard``: 「judge 失败(None=放行)」, ``read_think_gate``'s ``pre_llm_call``:
# always returns None). A 30 s budget expiring is not a verdict about the turn; turning it into
# one inverts the plugin's own contract and costs the user the whole turn for zero information.
# Denial stays where it is load-bearing and cheap: ``pre_tool_call`` (adaptable error result),
# ``pre_verify`` (don't certify the claim), or an explicit ``veto_on_timeout=True`` registration.
_HOOK_TIMEOUT_DEGRADE_HOOKS: set[str] = {"pre_llm_call"}
# One turn's total ``pre_llm_call`` budget, as a multiple of the per-callback budget. Without it
# 20+ registered callbacks can each pay a full budget + retry, which is a many-minute turn even
# once nothing blocks. The remainder is spent callback by callback; past it, callbacks degrade
# unstarted instead of queueing more wall clock.
_HOOK_TURN_BUDGET_FACTOR = 1.5
# Documented parent-thread serialization contract — never run on a timeout worker (hooks.md).
_HOOK_CALLER_THREAD_HOOKS: set[str] = {"subagent_stop"}
# After a timeout, suppress the same callback this long so a hung hook cannot pile up threads.
_HOOK_TIMEOUT_SUPPRESSION_SECONDS = 60.0
# Live workers a hung callback may accumulate before it is skipped outright (#105223 / #98382).
_HOOK_MAX_ABANDONED_WORKERS = 3
# A veto-grade (``veto_on_timeout=True``) policy callback that blew its budget gets ONE automatic
# retry under this cap. Without it a transient infrastructure hiccup ends the user's whole turn: a
# guard's auxiliary judge stalled in DNS/connect, the abandoned worker never returns, and a retry
# inside the 60s suppression window is skipped — which veto-grade turns into a second identical
# veto. The first attempt's budget is already spent, so the retry is capped to bound the added
# latency. A degrading callback is never retried: there is no verdict to recover.
_HOOK_RETRY_TIMEOUT_SECS = 15.0
_PRE_TOOL_CALL_TIMEOUT_BLOCK_MESSAGE = "pre_tool_call plugin callback timed out or is still running"

# The one hook whose callback rewrites the reply the user would otherwise see streaming.
STREAM_HOLDING_HOOK = "transform_llm_output"
_TRANSFORM_FAILURE_MARKER = (
    "\n\n[插件门禁不可用 / UNKNOWN] 回复改写链中的 {callbacks} 超时或失败；"
    "该门禁未完成判定，不能视为通过。"
)


def stream_held() -> bool:
    """True when a plugin loaded in the active profile registered ``transform_llm_output`` with
    ``hold_stream=True``: that profile's replies reach the user only as the rewritten final.
    Lazy-discovers first, like ``has_hook``."""
    from hermes_cli.plugins import _delivery_manager

    return _delivery_manager().holds_stream()


def held_stream_transforms() -> tuple[Callable, ...]:
    """:meth:`PluginDispatchMixin.held_transforms` of the active profile's manager (lazy-discovers)."""
    from hermes_cli.plugins import _delivery_manager

    return _delivery_manager().held_transforms()


def _hook_timeout_block_message(
    hook_name: str, callback: Optional[Callable] = None, *, timeout: float = 0.0,
    cooling_down: bool = False, retry_in: float = 0.0, retried: bool = False,
) -> str:
    """Fail-closed block message for a timed-out/still-running policy-hook callback.

    Names the callback and its plugin, not just the hook: a bare ``pre_llm_call plugin
    callback timed out`` leaves both the operator and the user unable to tell WHICH guard
    hung, whether it already retried, and whether waiting will help — every one of those
    changes what they should do next.
    """
    if hook_name == "pre_tool_call":
        return _PRE_TOOL_CALL_TIMEOUT_BLOCK_MESSAGE  # exact string asserted by existing tests
    who = ""
    if callback is not None:
        who = (f"插件回调 {getattr(callback, '__name__', repr(callback))} "
               f"({getattr(callback, '__module__', '') or '未知插件'}) ")
    if cooling_down:
        wait = f"，约 {max(retry_in, 0.0):.0f}s 后解除" if retry_in > 0 else ""
        state = f"仍在冷却（上一轮超时的回调尚未返回{wait}）"
    else:
        state = f"超出 {timeout:g}s 预算未返回" + ("，自动重试一次仍无判定" if retried else "")
    return (f"[{hook_name} 门禁暂不可用 / UNKNOWN] {who}{state}——本轮未发起任何模型调用；"
            "稍后重发即可。")


def _hook_timeout_unknown_note(
    hook_name: str, callback: Optional[Callable] = None, *, timeout: float = 0.0,
    cooling_down: bool = False,
) -> str:
    """Degradation note for a timed-out hook whose verdict is not load-bearing alone.

    Loud, not silent: the agent is told which guard did not judge this turn, that its context
    injection is missing, and where denial still applies. That keeps the failure visible without
    letting a slow judge delete the user's turn.
    """
    who = ""
    if callback is not None:
        who = (f"插件回调 {getattr(callback, '__name__', repr(callback))} "
               f"({getattr(callback, '__module__', '') or '未知插件'}) ")
    state = ("本轮未启动（该回调仍在超时冷却、已达放弃线程上限，或本轮钩子预算已用尽）"
             if cooling_down else f"超出 {timeout:g}s 预算未返回")
    return (f"[{hook_name} 门禁未判定 / UNKNOWN] {who}{state}——本轮按该守卫自己的「无判定」"
            "契约继续：它没有否决本轮，但它本该注入的上下文/纪律提示本轮缺失，相关自检请自行补上。"
            "动作级否决（pre_tool_call / pre_verify）不受影响，仍按原语义拦截。")


def _hook_error_unknown_note(hook_name: str, callback: Callable, exc: BaseException) -> str:
    """Degradation note for a callback that raised: same contract as a timeout, with the error."""
    who = (f"插件回调 {getattr(callback, '__name__', repr(callback))} "
           f"({getattr(callback, '__module__', '') or '未知插件'}) ")
    return (f"[{hook_name} 门禁未判定 / UNKNOWN] {who}抛出 {type(exc).__name__}: {str(exc)[:200]}"
            "——本轮按该守卫自己的「无判定」契约继续：它没有否决本轮，但它本该注入的上下文/"
            "纪律提示本轮缺失，相关自检请自行补上。动作级否决（pre_tool_call / pre_verify）"
            "不受影响，仍按原语义拦截；该守卫本身需要修。")


def _policy_error_block_directive(hook_name: str, cb: Callable, exc: BaseException) -> dict[str, str]:
    """Block directive for a fail-closed hook whose callback raised: names the callback and the
    error (truncated — a hook that embeds tool args in its exception must not grow the tool
    result) so the operator can tell a crashing guard from a slow one."""
    callback_name = getattr(cb, "__name__", repr(cb))
    return {"action": "block",
            "message": f"{hook_name} plugin callback {callback_name} raised {type(exc).__name__}: {str(exc)[:200]}"}

# System-prompt sections are tightly bounded: they become high-trust prompt bytes charged every turn.
SYSTEM_PROMPT_SECTION_POSITIONS = frozenset({"after_memory"})
DEFAULT_SYSTEM_PROMPT_SECTION_MAX_CHARS = 4_000
MAX_SYSTEM_PROMPT_SECTION_CHARS = 4_000
MAX_SYSTEM_PROMPT_SECTIONS = 32
MAX_SYSTEM_PROMPT_SECTIONS_TOTAL_CHARS = 8_000
_SYSTEM_PROMPT_SECTION_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_SYSTEM_PROMPT_SECTION_HEADING_PREFIX = "## Plugin Context: "
PLUGIN_SECTIONS_START = "<!-- hermes-plugin-sections:start -->"
PLUGIN_SECTIONS_END = "<!-- hermes-plugin-sections:end -->"


def is_valid_system_prompt_section_id(value: Any) -> bool:
    """Return whether *value* is a stable, heading-safe section identifier."""
    return isinstance(value, str) and bool(_SYSTEM_PROMPT_SECTION_ID_RE.fullmatch(value))


def format_system_prompt_section(section_id: str, content: str) -> str:
    """Render an auditable, length-framed block recoverable from the full prompt."""
    return (
        f"{_SYSTEM_PROMPT_SECTION_HEADING_PREFIX}{section_id}\n"
        f"<!-- hermes-plugin-section-chars:{len(content)} -->\n\n{content}")


def format_system_prompt_sections(sections: list) -> str:
    """Render the canonical container used for persistence recovery."""
    if not sections:
        return ""
    blocks = [format_system_prompt_section(item.id, item.content) for item in sections]
    return f"{PLUGIN_SECTIONS_START}\n" + "\n\n".join(blocks) + f"\n{PLUGIN_SECTIONS_END}"


# Reserved event namespace prefix — only core may publish ``hermes:<event>``.
HERMES_EVENT_NAMESPACE = "hermes"
# Event recursion depth cap (subscribers may emit); over-deep emits are dropped with a warning.
_EVENT_EMIT_DEPTH_CAP = 8
# Max queued + running events per manager generation; emit never waits — a full budget drops.
_EVENT_PENDING_CAP = 64
_EVENT_WORKER_STOP = object()


@dataclass(frozen=True)
class PluginSystemPromptSection:
    """A plugin-owned section rendered once for each new session."""

    id: str
    content: str | Callable[[Mapping[str, Any]], str]
    position: str
    max_chars: int
    plugin: str


@dataclass(frozen=True)
class RenderedPluginSystemPromptSection:
    """Validated prompt bytes frozen on the owning AIAgent."""

    id: str
    content: str
    position: str
    plugin: str


@dataclass(frozen=True)
class _EventSubscription:
    """Host-owned subscription ledger entry."""

    owner: str
    callback: Callable


@dataclass(frozen=True)
class _QueuedPluginEvent:
    """Immutable dispatch envelope consumed by the event worker."""

    event: str
    payload: dict[str, Any]
    subscriptions: tuple[_EventSubscription, ...]
    depth: int
    generation: int
    # The emitter's contextvars: the single worker thread serves every profile, so each delivery
    # runs under the profile scope the emit happened in (#118538).
    context: contextvars.Context


# Hook callback timeout (non-blocking abandon). Default cap per Python hook callback; overridden by
# ``plugins.hook_callback_timeout``. Shell hooks enforce their own subprocess timeout.
_HOOK_CALLBACK_TIMEOUT_SECS = 30.0
_MAX_HOOK_CALLBACK_TIMEOUT_SECS = 600.0
# Returned by _run_hook_callback_bounded when the callback produced no decision: a worker ran
# and blew its budget (or could not be started) — "timed out", vs _HOOK_SKIPPED_UNSTARTED below.
_HOOK_SKIPPED = object()
# ...and when nothing was invoked for this call at all (suppression window, abandoned-worker cap).
# Kept apart so the fail-closed block can tell the user "retried, still no verdict" from "still
# cooling down, wait" instead of re-reading the timeout bookkeeping at the call site.
_HOOK_SKIPPED_UNSTARTED = object()


def _hook_call_identity(kwargs: dict[str, Any], hook_name: str = "") -> Optional[str]:
    """Identity of the call this callback fires for, or ``None`` when the event has none.

    Concurrent invocations of the same tool in one session must not collapse into one
    gate key: they are different work, and treating the second as a duplicate drops the
    hook as if a callback had timed out (upstream #98382). The identity is already in the
    payload; nothing new is plumbed. Deliberately not ``api_request_id`` — one API request
    carries many tool calls, which would re-collapse the keys.
    """
    for field in _HOOK_CALL_IDENTITY_FIELDS.get(hook_name, ("tool_call_id", "turn_id")):
        value = kwargs.get(field)
        if isinstance(value, str) and value:
            return value
    return None


def _hook_uses_callback_timeout(hook_name: str, timeout: float) -> bool:
    """Whether *hook_name* should run under the non-blocking timeout path."""
    if timeout <= 0 or hook_name in _HOOK_CALLER_THREAD_HOOKS:
        return False
    return hook_name in _HOOK_TIMEOUT_BOUNDED_HOOKS or hook_name in _HOOK_TIMEOUT_FAIL_CLOSED_HOOKS


_PLUGIN_COMMAND_AWAIT_TIMEOUT_SECS = 30.0


def resolve_plugin_command_result(result: Any) -> Any:
    """Resolve a plugin command result, awaiting async handlers: ``asyncio.run`` when no loop is
    running, else a helper thread with its own loop (30s bound so a hung handler cannot wedge the
    terminal)."""
    if not inspect.isawaitable(result):
        return result
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(result)
    outcome: dict[str, Any] = {}
    failure: dict[str, BaseException] = {}
    done = threading.Event()

    def _runner() -> None:
        try:
            outcome["value"] = asyncio.run(result)
        except BaseException as exc:  # pragma: no cover - re-raised below
            failure["exc"] = exc
        finally:
            done.set()

    # copy_context: the helper thread must see the caller's profile/secret scope, else an
    # async hook under a running loop reads the default HERMES_HOME and get_secret raises.
    threading.Thread(target=contextvars.copy_context().run, args=(_runner,),
                     name="hermes-plugin-command-await", daemon=True).start()
    if not done.wait(timeout=_PLUGIN_COMMAND_AWAIT_TIMEOUT_SECS):
        raise TimeoutError("Plugin command async handler did not complete within "
                           f"{_PLUGIN_COMMAND_AWAIT_TIMEOUT_SECS:.0f}s")
    if "exc" in failure:
        raise failure["exc"]
    return outcome.get("value")


def get_pre_verify_continue_message(
    *, session_id: str = "", platform: str = "", model: str = "", coding: bool = False, attempt: int = 0,
    final_response: str = "", changed_paths: Optional[list[str]] = None, cwd: str = "",
) -> Optional[str]:
    """Check ``pre_verify`` hooks for a continuation directive.

    ``{"action": "continue", "message"}`` (or Claude-Code Stop ``{"decision": "block",
    "reason"}``) keeps the turn going; first non-empty message wins, any other return lets the
    turn finish. ``coding``/``attempt`` let hooks scope and self-throttle.
    """
    from hermes_cli.plugins import invoke_hook

    hook_results = invoke_hook(
        "pre_verify", session_id=session_id, platform=platform, model=model, coding=coding,
        attempt=attempt, final_response=final_response, changed_paths=list(changed_paths or []), cwd=cwd,
    )
    for result in hook_results:
        if not isinstance(result, dict):
            continue
        action = str(result.get("action") or result.get("decision") or "").strip().lower()
        message = result.get("message") or result.get("reason")
        if action in ("continue", "block") and isinstance(message, str) and message.strip():
            return message.strip()
    return None


class PluginDispatchMixin:
    @staticmethod
    def _hook_callback_kwargs(callback: Callable, payload: dict[str, Any]) -> dict[str, Any]:
        """The slice of *payload* a callback accepts: everything for ``**kwargs`` (or
        un-introspectable) callbacks, only declared names for narrow legacy signatures."""
        try:
            parameters = inspect.signature(callback).parameters
        except (TypeError, ValueError):
            return dict(payload)  # no introspectable signature
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
            return dict(payload)
        keyword_kinds = {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
        return {
            name: value for name, value in payload.items()
            if name in parameters and parameters[name].kind in keyword_kinds
        }

    @classmethod
    def _invoke_hook_callback(cls, callback: Callable, payload: dict[str, Any]) -> Any:
        """Invoke a hook while withholding additive fields from narrow legacy callbacks.

        An ``async def`` callback returns a coroutine; resolve it the way plugin slash commands
        are (loop-safe), otherwise the bare coroutine object is appended to the results and the
        plugin's body never runs (#12449).
        """
        return resolve_plugin_command_result(callback(**cls._hook_callback_kwargs(callback, payload)))

    def _record_gate_event(
        self,
        hook_name: str,
        callback: Callable,
        kwargs: dict[str, Any],
        status: GateStatus,
        *,
        detail: str = "",
        duration_ms: float = 0.0,
    ) -> None:
        """Store safe policy-hook evidence; diagnostics must never alter enforcement."""
        if hook_name not in GATE_EVIDENCE_HOOKS:
            return
        try:
            self.gate_plane.record(
                hook_name,
                callback,
                session_id=str(kwargs.get("session_id") or ""),
                call_id=_hook_call_identity(kwargs, hook_name) or "",
                status=status,
                detail=detail,
                duration_ms=duration_ms,
            )
        except Exception:
            logger.warning("Gate Plane failed to record '%s' callback", hook_name, exc_info=True)

    def invoke_hook(self, hook_name: str, **kwargs: Any) -> list[Any]:
        """Call all callbacks for *hook_name*; return their non-``None`` results.

        Payloads evolve additively: ``**kwargs`` callbacks get everything, narrow signatures only
        what they declare. Each callback is isolated. Bounded hooks and ``pre_tool_call`` run under
        ``plugins.hook_callback_timeout`` (worker abandoned, never joined); ``pre_tool_call`` fails
        closed with a block directive, others skip. ``_HOOK_CALLER_THREAD_HOOKS`` always run on the
        caller thread. ``pre_llm_call`` may return ``{"context": "..."}`` (or a str) to inject, or
        ``{"action": "block", "message": "..."}`` to end the turn before any inference request.
        """
        from agent.safe_worker_policy import safe_worker_enabled

        if safe_worker_enabled():
            return []
        # Gateway platform events define event-local envelopes; a bus-wide version here would turn
        # unrelated adapter payloads into one monolithic compatibility contract.
        if hook_name != "gateway_platform_event":
            kwargs.setdefault("telemetry_schema_version", OBSERVER_SCHEMA_VERSION)
        if hook_name == STREAM_HOLDING_HOOK:
            return self._run_output_transform_chain(kwargs)
        return self._run_hook_callbacks(hook_name, self._hooks.get(hook_name, []), kwargs)

    def _run_output_transform_chain(self, kwargs: dict[str, Any]) -> list[Any]:
        """Compose reply rewrites in registration order.

        Every callback sees the current text and the immutable ``original_response_text``.
        The next callback therefore sees the previous gate's rewrite instead of competing
        for a single first-non-empty slot. A failed transform contributes a visible UNKNOWN
        marker; silently dropping that gate would turn infrastructure failure into success.
        """
        original = str(kwargs.get("response_text") or "")
        current = original
        failed_callbacks: list[Callable] = []
        for cb in self._hooks.get(STREAM_HOLDING_HOOK, []):
            payload = {
                **kwargs,
                "response_text": current,
                "original_response_text": original,
            }
            callback_failures: list = []
            results = self._run_hook_callbacks(
                STREAM_HOLDING_HOOK, [cb], payload, callback_failures,
            )
            failed_callbacks.extend(callback_failures)
            rewrite = next(
                (r for r in results if isinstance(r, str) and r),
                None,
            )
            if rewrite is not None:
                current = rewrite
        return self._finish_output_transform_chain(original, current, failed_callbacks)

    @staticmethod
    def _finish_output_transform_chain(
        original: str, current: str, failed_callbacks: list[Callable],
    ) -> list[Any]:
        """Make transform failures visible and expose only a changed final text."""
        if failed_callbacks:
            names = ", ".join(
                getattr(cb, "__name__", repr(cb)) for cb in failed_callbacks
            )
            current += _TRANSFORM_FAILURE_MARKER.format(callbacks=names)
        return [current] if current != original else []

    def _hook_cooldown_remaining(self, hook_name: str, cb: Callable, kwargs: dict[str, Any]) -> Optional[float]:
        """Seconds until *cb* may run again, or ``None`` when it is free to run now.

        Read-only view of the timeout bookkeeping so the fail-closed block message can say
        "timed out just now" instead of "still cooling down": those two need different advice
        (retry vs wait), and the callback name alone never told the user which it was."""
        suppression_key = (hook_name, id(cb))
        gate_key = (*suppression_key, _hook_call_identity(kwargs, hook_name))
        with self._hook_timeout_lock:
            suppressed_until = self._hook_timeout_suppressed_until.get(suppression_key)
            now = time.monotonic()
            if gate_key in self._hook_running_callbacks:
                return max(0.0, (suppressed_until or now) - now)
            if suppressed_until is not None and suppressed_until > now:
                return suppressed_until - now
        return None

    def _record_hook_policy(
        self, hook_name: str, callback: Callable, *, veto_on_timeout: bool, policy: str,
    ) -> None:
        """Remember what a fail-closed hook registration is allowed to do when it cannot decide."""
        self._hook_gate_policy[(hook_name, id(callback))] = (
            callback, bool(veto_on_timeout), str(policy or "required"))

    def _forget_hook_policy(self, hook_name: str, callback: Callable) -> None:
        self._hook_gate_policy.pop((hook_name, id(callback)), None)

    def _hook_policy(self, hook_name: str, callback: Callable) -> tuple[bool, str]:
        """``(veto_on_timeout, gate_policy)`` for one registration; the fail-open default for
        callbacks registered outside ``track_hook`` (memory-provider fallback hooks, shell hooks)."""
        entry = self._hook_gate_policy.get((hook_name, id(callback)))
        if entry is None or entry[0] is not callback:
            return False, "required"
        return entry[1], entry[2]

    def _hook_timeout_directive(
        self, hook_name: str, cb: Callable, kwargs: dict[str, Any], *, timeout: float,
        cooling_down: bool, retried: bool,
    ) -> Optional[dict[str, str]]:
        """What a timed-out / skipped callback contributes to this call.

        A veto-grade registration blocks, as before. Everything else degrades to the plugin's own
        no-verdict contract: a ``required`` gate's absence is injected as an explicit UNKNOWN note
        (the agent has to cover the missing self-check itself), an ``advisory`` gate's absence is
        logged and recorded and contributes nothing — an advisory gate owes no turn-level claim.
        """
        veto_on_timeout, policy = self._hook_policy(hook_name, cb)
        if hook_name not in _HOOK_TIMEOUT_DEGRADE_HOOKS or veto_on_timeout:
            return {"action": "block", "message": _hook_timeout_block_message(
                hook_name, cb, timeout=timeout, cooling_down=cooling_down,
                retry_in=self._hook_cooldown_remaining(hook_name, cb, kwargs) or 0.0,
                retried=retried and not cooling_down)}
        if policy == "advisory":
            return None
        return {"context": _hook_timeout_unknown_note(
            hook_name, cb, timeout=timeout, cooling_down=cooling_down)}

    def _hook_error_directive(
        self, hook_name: str, cb: Callable, exc: BaseException,
    ) -> Optional[dict[str, str]]:
        """What a callback that RAISED contributes: a veto only for a veto-grade registration.

        A guard that crashes on every turn must not delete every turn; it degrades like a timeout
        (the note names the exception) and the failure is still reported once at WARNING.
        """
        veto_on_timeout, policy = self._hook_policy(hook_name, cb)
        if hook_name not in _HOOK_TIMEOUT_DEGRADE_HOOKS or veto_on_timeout:
            return _policy_error_block_directive(hook_name, cb, exc)
        if policy == "advisory":
            return None
        return {"context": _hook_error_unknown_note(hook_name, cb, exc)}

    def _run_hook_callbacks(self, hook_name: str, callbacks: Any, kwargs: dict[str, Any],
                            failed: Optional[list] = None) -> list[Any]:
        """:meth:`invoke_hook`'s loop over ``callbacks``; each one that raised or timed out is
        appended to ``failed`` when given."""
        from hermes_cli.plugins import _resolve_hook_callback_timeout
        results: list[Any] = []
        timeout = _resolve_hook_callback_timeout()
        use_timeout = _hook_uses_callback_timeout(hook_name, timeout)
        fail_closed = hook_name in _HOOK_TIMEOUT_FAIL_CLOSED_HOOKS
        retryable_hook = hook_name in _HOOK_TIMEOUT_RETRY_HOOKS
        turn_deadline = (
            time.monotonic() + timeout * _HOOK_TURN_BUDGET_FACTOR
            if use_timeout and hook_name in _HOOK_TIMEOUT_DEGRADE_HOOKS else None
        )
        for cb in callbacks:
            started = time.perf_counter()
            try:
                if use_timeout:
                    veto_grade = fail_closed and (
                        hook_name not in _HOOK_TIMEOUT_DEGRADE_HOOKS
                        or self._hook_policy(hook_name, cb)[0])
                    budget = timeout
                    if turn_deadline is not None:
                        budget = min(timeout, turn_deadline - time.monotonic())
                    ret = (self._run_hook_callback_bounded(hook_name, cb, kwargs, budget)
                           if budget > 0 else _HOOK_SKIPPED_UNSTARTED)
                    retried = False
                    if (ret is _HOOK_SKIPPED or ret is _HOOK_SKIPPED_UNSTARTED) \
                            and retryable_hook and veto_grade:
                        # One retry, past the suppression window: the first attempt's budget is
                        # already spent, so an unretried skip costs the whole unit of work. Only
                        # a veto-grade callback buys that latency — a degrading one has nothing
                        # to protect, so a retry would only stretch the turn.
                        retried = True
                        ret = self._run_hook_callback_bounded(
                            hook_name, cb, kwargs, min(timeout, _HOOK_RETRY_TIMEOUT_SECS),
                            retry_attempt=True)
                    if ret is _HOOK_SKIPPED or ret is _HOOK_SKIPPED_UNSTARTED:
                        self._record_gate_event(
                            hook_name, cb, kwargs, GateStatus.UNKNOWN,
                            detail="callback skipped or timed out",
                            duration_ms=(time.perf_counter() - started) * 1000,
                        )
                        if failed is not None:
                            failed.append(cb)
                        directive = self._hook_timeout_directive(
                            hook_name, cb, kwargs, timeout=timeout,
                            cooling_down=ret is _HOOK_SKIPPED_UNSTARTED, retried=retried,
                        ) if fail_closed else None
                        if directive is not None:
                            results.append(directive)
                        continue
                else:
                    ret = self._invoke_hook_callback(cb, kwargs)
                status = result_status(ret)
                action = str(ret.get("action", "")).lower() if isinstance(ret, Mapping) else ""
                self._record_gate_event(
                    hook_name, cb, kwargs, status,
                    detail=f"action={action}" if action else "",
                    duration_ms=(time.perf_counter() - started) * 1000,
                )
                if ret is not None:
                    results.append(ret)
            except (Exception, SystemExit) as exc:  # health: allow BLE001 -- moved from invoke_hook: plugin callbacks are untrusted; each failure is reported and isolated
                self._report_hook_failure(hook_name, cb, kwargs, exc)
                self._record_gate_event(
                    hook_name, cb, kwargs, GateStatus.UNKNOWN,
                    detail=f"{type(exc).__name__}: {exc}",
                    duration_ms=(time.perf_counter() - started) * 1000,
                )
                if failed is not None:
                    failed.append(cb)
                if fail_closed:  # a guard that raised made no decision: veto only if it asked to
                    directive = self._hook_error_directive(hook_name, cb, exc)
                    if directive is not None:
                        results.append(directive)
        return results

    def _report_hook_failure(
        self, hook_name: str, cb: Callable, kwargs: dict[str, Any], exc: BaseException, *, surface: str = "Hook"
    ) -> None:
        """One WARNING per distinct (hook, callback, error); identical repeats at DEBUG.

        A callback whose signature names a parameter the hook never sends (``tool_data`` instead
        of ``tool_name``/``args``) fails identically on every tool call — ~1700 WARNING lines an
        hour that bury real signals (#111922). The first report names the fields the hook does
        provide so the plugin author can fix the signature. The key names the callback by
        module/qualname (not ``id()``, which CPython recycles across plugin reloads) and
        truncates the message so a hook that embeds tool args in its error cannot grow the set
        per call; the set is cleared on unload alongside the timeout-suppression map.
        """
        callback_name = getattr(cb, "__name__", repr(cb))
        key = (hook_name, getattr(cb, "__module__", ""), getattr(cb, "__qualname__", callback_name),
               type(exc).__name__, str(exc)[:200])
        if key in self._hook_failures_reported:
            logger.debug("%s '%s' callback %s raised again: %s", surface, hook_name, callback_name, exc)
            return
        self._hook_failures_reported.add(key)
        logger.warning(
            "%s '%s' callback %s raised: %s (%s provides: %s; identical failures are logged at DEBUG from now on)",
            surface, hook_name, callback_name, exc, surface.lower(), ", ".join(sorted(kwargs)) or "no fields")

    def _run_hook_callback_bounded(
        self, hook_name: str, cb: Callable, kwargs: dict[str, Any], timeout: float, *,
        retry_attempt: bool = False,
    ) -> Any:
        """Run one callback on a daemon worker with a wall-clock cap; ``_HOOK_SKIPPED`` when it
        timed out (worker abandoned, never joined) or its worker could not be started, and
        ``_HOOK_SKIPPED_UNSTARTED`` when it was suppressed, still running for this call id, or
        over the abandoned-worker cap. Exceptions propagate.

        ``retry_attempt`` marks the one automatic retry of a fail-closed callback: it is new
        work, not a duplicate of the worker it retries, so it takes its own gate token. It still
        honours the abandoned-worker cap — the pile-up bound, not the time back-off, is what
        keeps a permanently hung guard from leaking a thread per turn."""
        callback_name = getattr(cb, "__name__", repr(cb))
        # Suppression is a fact about the CALLBACK — a hung one must keep its back-off —
        # so that key stays coarse. The gate must instead tell CONCURRENT CALLS apart.
        suppression_key = (hook_name, id(cb))
        gate_key = (*suppression_key, _hook_call_identity(kwargs, hook_name))
        if retry_attempt:
            gate_key = (*gate_key, "retry")
        token = object()
        with self._hook_timeout_lock:
            suppressed_until = self._hook_timeout_suppressed_until.get(suppression_key)
            if not retry_attempt and (
                    gate_key in self._hook_running_callbacks
                    or (suppressed_until is not None and suppressed_until > time.monotonic())):
                logger.warning(
                    "Hook '%s' callback %s skipped after previous "
                    "timeout or while still running", hook_name, callback_name)
                return _HOOK_SKIPPED_UNSTARTED
            # Workers abandoned on timeout still hold threads. Once the suppression window has
            # passed, a fresh call id may start a new worker (a hung guard must not fail every
            # later tool call closed until restart, #105223), but only up to a small cap per
            # callback — expiring the bookkeeping while the hung worker lives must not leak a
            # thread per call (#98382). At the cap the callback keeps being skipped (fail-closed
            # for pre_tool_call) until one of its workers finishes and releases its slot.
            abandoned = self._hook_abandoned.get(suppression_key)
            if abandoned and len(abandoned) >= _HOOK_MAX_ABANDONED_WORKERS:
                logger.warning(
                    "Hook '%s' callback %s (%s) skipped: %d abandoned worker(s) still running — "
                    "the plugin is hung; fix or disable it (retried when a worker finishes)",
                    hook_name, callback_name, getattr(cb, "__module__", "unknown plugin"), len(abandoned))
                return _HOOK_SKIPPED_UNSTARTED
            if suppressed_until is not None:
                self._hook_timeout_suppressed_until.pop(suppression_key, None)
            self._hook_running_callbacks[gate_key] = token

        context = contextvars.copy_context()
        done = threading.Event()
        outcome: dict[str, Any] = {}
        failure: dict[str, BaseException] = {}

        def _release_token() -> None:
            with self._hook_timeout_lock:
                if self._hook_running_callbacks.get(gate_key) is token:
                    self._hook_running_callbacks.pop(gate_key, None)
                    abandoned = self._hook_abandoned.get(suppression_key)
                    if abandoned is not None:
                        abandoned.discard(gate_key)
                        if not abandoned:
                            self._hook_abandoned.pop(suppression_key, None)

        def _runner() -> None:
            try:
                outcome["value"] = context.run(self._invoke_hook_callback, cb, kwargs)
            except BaseException as exc:
                failure["exc"] = exc
            finally:
                _release_token()
                done.set()

        thread = threading.Thread(target=_runner, name=f"hermes-hook-{callback_name}"[:40], daemon=True)
        try:
            thread.start()
        except RuntimeError as exc:
            _release_token()  # the runner's finally never runs when OS thread creation fails
            logger.warning(
                "Hook '%s' callback %s worker failed to start: %s — skipping",
                hook_name, callback_name, exc)
            return _HOOK_SKIPPED
        if not done.wait(timeout=timeout):  # do not join — that would reintroduce the hang
            with self._hook_timeout_lock:
                # See #6622.
                self._hook_timeout_suppressed_until[suppression_key] = (
                    time.monotonic() + self._hook_timeout_suppression_seconds)
                # The worker may have finished (and released its token) between the wait
                # expiring and this lock; recording it as abandoned then would block the
                # callback for that call id until reload with no thread behind it.
                if self._hook_running_callbacks.get(gate_key) is token:
                    self._hook_abandoned.setdefault(suppression_key, set()).add(gate_key)
            logger.warning(
                "Hook '%s' callback %s (%s) timed out after %gs — skipping",
                hook_name, callback_name, getattr(cb, "__module__", "unknown plugin"), timeout)
            return _HOOK_SKIPPED
        if "exc" in failure:
            raise failure["exc"]
        return outcome.get("value")

    def _subscribe_event(self, owner: str, event: str, callback: Callable) -> None:
        """Add an owner-tagged event subscription in registration order."""
        if not callable(callback):
            raise TypeError("Event subscriber callback must be callable")
        with self._event_lock:
            self._subscriptions.setdefault(event, []).append(_EventSubscription(owner, callback))

    def _remove_plugin_subscriptions(self, owner: str) -> int:
        """Remove every subscription owned by *owner*; return the count. Queued envelopes re-check
        membership per callback, so this also cancels already-snapshotted deliveries.

        TODO(#64229): when the central plugin ownership ledger / registration handles land, route this
        owner-tagged bookkeeping through that ledger so per-plugin unload cancels event subscriptions
        alongside every other registration surface. This method is the integration seam.
        """
        removed = 0
        with self._event_lock:
            for event in list(self._subscriptions):
                entries = self._subscriptions[event]
                retained = [entry for entry in entries if entry.owner != owner]
                removed += len(entries) - len(retained)
                if retained:
                    self._subscriptions[event] = retained
                else:
                    del self._subscriptions[event]
        return removed

    def _ensure_event_worker_locked(self) -> None:
        worker = self._event_worker
        if worker is not None and worker.is_alive():
            return
        worker = threading.Thread(
            target=self._event_worker_loop, args=(self._event_queue,), name="hermes-plugin-events",
            daemon=True,
        )
        self._event_worker = worker
        worker.start()

    def _event_worker_loop(self, dispatch_queue: queue.Queue[Any]) -> None:
        while True:
            item = dispatch_queue.get()
            try:
                if item is _EVENT_WORKER_STOP:
                    return
                self._deliver_event(item)
            finally:
                if item is not _EVENT_WORKER_STOP:
                    self._mark_event_done(item.generation)
                dispatch_queue.task_done()

    def _mark_event_done(self, generation: int) -> None:
        with self._event_idle:
            pending = self._event_pending_by_generation.get(generation, 0)
            if pending > 0:
                self._event_pending_by_generation[generation] = pending - 1
            self._event_idle.notify_all()

    def _deliver_event(self, item: _QueuedPluginEvent) -> None:
        """Deliver one queued event on the host-owned worker thread."""
        with self._event_lock:
            if item.generation != self._event_generation:
                return
        previous_depth = getattr(self._emit_depth, "value", 0)
        self._emit_depth.value = item.depth
        try:
            for subscription in item.subscriptions:
                with self._event_lock:
                    if item.generation != self._event_generation:
                        break
                    # Owner unload may have removed this entry after the event was queued.
                    if not any(cur is subscription for cur in self._subscriptions.get(item.event, [])):
                        continue
                callback = subscription.callback
                try:
                    # Fresh deep copy per subscriber: no callback can mutate what the next sees.
                    resolve_plugin_command_result(
                        item.context.copy().run(callback, **copy.deepcopy(item.payload)))
                except (Exception, SystemExit) as exc:
                    # A subscriber that fails identically on every emit is reported once (#111922).
                    self._report_hook_failure(item.event, callback, item.payload, exc, surface="Event")
        finally:
            self._emit_depth.value = previous_depth

    def _wait_for_event_dispatch(self, timeout: float = 2.0) -> bool:
        """Wait for the current event generation to become idle (test helper)."""
        with self._event_idle:
            generation = self._event_generation
            return self._event_idle.wait_for(
                lambda: self._event_pending_by_generation.get(generation, 0) == 0, timeout=timeout)

    def _dispatch_event(self, event: str, payload: dict[str, Any]) -> int:
        """Queue *event* without blocking; return the subscriber count scheduled. Pending work is
        bounded per generation so a blocking subscriber costs one worker and later emits drop."""
        depth = getattr(self._emit_depth, "value", 0)
        if depth >= _EVENT_EMIT_DEPTH_CAP:
            logger.warning(
                "Event bus recursion cap (%d) exceeded while dispatching '%s' "
                "— dropping this emit to prevent an infinite loop", _EVENT_EMIT_DEPTH_CAP, event)
            return 0
        budget_msg = "Event bus pending budget (%d) exhausted while dispatching '%s' — dropping this emit"
        with self._event_lock:
            subscriptions = tuple(self._subscriptions.get(event, []))
            if not subscriptions:
                return 0
            generation = self._event_generation
            pending = self._event_pending_by_generation.get(generation, 0)
            if pending >= _EVENT_PENDING_CAP:
                logger.warning(budget_msg, _EVENT_PENDING_CAP, event)
                return 0
            item = _QueuedPluginEvent(
                event=event, payload=dict(payload), subscriptions=subscriptions, depth=depth + 1,
                generation=generation, context=contextvars.copy_context())
            try:
                self._event_queue.put_nowait(item)
            except queue.Full:
                logger.warning(budget_msg, _EVENT_PENDING_CAP, event)
                return 0
            self._event_pending_by_generation[generation] = pending + 1
            self._ensure_event_worker_locked()
            return len(subscriptions)

    def has_hook(self, hook_name: str) -> bool:
        """Return True when at least one callback is registered for a hook."""
        from agent.safe_worker_policy import safe_worker_enabled

        if safe_worker_enabled():
            return False
        return bool(self._hooks.get(hook_name))

    def track_hook(self, context: Any, hook_name: str, callback: Callable, *, hold_stream: bool = False,
                   veto_on_timeout: bool = False):
        """Register ``callback`` for ``hook_name`` on behalf of ``context`` (unknown names warn but are
        still stored). ``hold_stream=True`` on ``transform_llm_output`` declares the rewrite must reach
        the user first, so this profile's replies are not streamed (:meth:`holds_stream`).
        ``veto_on_timeout=True`` on a fail-closed hook keeps the old semantics for that callback: a
        budget timeout fails the unit of work closed instead of degrading to its no-verdict path."""
        from hermes_cli.plugins import VALID_HOOKS

        handle = context._track_callback("hook", hook_name, callback, self._hooks, VALID_HOOKS)
        if hold_stream and hook_name != STREAM_HOLDING_HOOK:
            logger.warning("Plugin '%s' set hold_stream on '%s'; only %s rewrites the reply, ignored",
                           context.manifest.name, hook_name, STREAM_HOLDING_HOOK)
        else:
            handle.hold_stream = hold_stream
        if hook_name in _HOOK_TIMEOUT_FAIL_CLOSED_HOOKS:
            policy = str(getattr(context.manifest, "gate_policy", "") or "required")
            self._record_hook_policy(
                hook_name, callback, veto_on_timeout=veto_on_timeout, policy=policy)
            handle.veto_on_timeout = bool(veto_on_timeout)
            # The policy entry outlives the callback otherwise: keep it exactly as long as the
            # registration (a reload re-registers, and an id() may be recycled after unload).
            previous_on_dispose = handle._on_dispose

            def _forget_policy(disposed, _hook=hook_name, _cb=callback, _previous=previous_on_dispose):
                self._forget_hook_policy(_hook, _cb)
                if _previous is not None:
                    _previous(disposed)

            handle._on_dispose = _forget_policy
        elif veto_on_timeout:
            logger.warning("Plugin '%s' set veto_on_timeout on '%s'; only fail-closed policy hooks "
                           "(%s) can veto a unit of work, ignored",
                           context.manifest.name, hook_name,
                           ", ".join(sorted(_HOOK_TIMEOUT_FAIL_CLOSED_HOOKS)))
        return handle

    def holds_stream(self) -> bool:
        """True while a loaded plugin's ``transform_llm_output`` registration declared ``hold_stream``.
        Read under the discovery lock: a reload empties the ledger before plugins re-register, and a
        turn deciding in that window would stream text the plugin must rewrite first."""
        with self._discovery_lock:
            return any(r.hold_stream for owned in self._ownership_ledger.values() for r in owned)

    def held_transforms(self) -> tuple[Callable, ...]:
        """The ``transform_llm_output`` callbacks a turn that holds its stream must run, captured with
        the hold decision under the discovery lock; ``()`` when nothing holds. A reload during the turn
        empties ``_hooks`` before plugins re-register, so the held turn keeps the set it started with
        (:meth:`run_held_transforms`)."""
        with self._discovery_lock:
            if not any(r.hold_stream for owned in self._ownership_ledger.values() for r in owned):
                return ()
            return tuple(self._hooks.get(STREAM_HOLDING_HOOK, []))

    def run_held_transforms(self, callbacks: tuple[Callable, ...], **kwargs: Any) -> tuple[Optional[str], bool]:
        """Run a held turn's snapshotted transforms: ``(rewrite, failed)``.

        Rewrites compose in registration order. ``failed`` is True when any callback raised or
        timed out, so the caller must not deliver the original untransformed reply.
        """
        from agent.safe_worker_policy import safe_worker_enabled

        if safe_worker_enabled():
            return None, False
        kwargs.setdefault("telemetry_schema_version", OBSERVER_SCHEMA_VERSION)
        original = str(kwargs.get("response_text") or "")
        current = original
        failed_callbacks: list[Callable] = []
        for cb in callbacks:
            payload = {
                **kwargs,
                "response_text": current,
                "original_response_text": original,
            }
            callback_failures: list = []
            results = self._run_hook_callbacks(
                STREAM_HOLDING_HOOK, [cb], payload, callback_failures,
            )
            failed_callbacks.extend(callback_failures)
            rewrite = next(
                (r for r in results if isinstance(r, str) and r),
                None,
            )
            if rewrite is not None:
                current = rewrite
        return (current if current != original else None), bool(failed_callbacks)

    async def _await_hook_callback(
        self, hook_name: str, cb: Callable, kwargs: dict[str, Any], timeout: float, *, retry: bool,
    ) -> tuple[Any, bool]:
        """Await one callback under *timeout*; ``(value, timed_out)``.

        A coroutine callback is cancellable, so a timed-out one leaves nothing running: unlike
        the threaded path, the retry needs no gate bypass. ``retry`` buys the fail-closed
        callback one more attempt under ``_HOOK_RETRY_TIMEOUT_SECS`` — the same reason the sync
        path retries: a transient stall in a guard's auxiliary judge must not veto the turn."""
        value = cb(**self._hook_callback_kwargs(cb, kwargs))
        if not inspect.isawaitable(value):
            return value, False
        try:
            return await asyncio.wait_for(value, timeout), False
        except TimeoutError:
            if not retry:
                return None, True
        second = cb(**self._hook_callback_kwargs(cb, kwargs))
        if not inspect.isawaitable(second):
            return second, False
        try:
            return await asyncio.wait_for(second, min(timeout, _HOOK_RETRY_TIMEOUT_SECS)), False
        except TimeoutError:
            return None, True

    async def ainvoke_hook(self, hook_name: str, **kwargs: Any) -> list[Any]:
        """:meth:`invoke_hook` for callers that are already on an event loop.

        Same payload narrowing, per-callback isolation and result contract. The difference is
        where an ``async def`` callback runs: here it is awaited on the caller's own loop, so a
        callback that awaits anything scheduled on that loop can make progress. Through the
        sync path it runs on a helper thread while the caller blocks in ``done.wait()`` — on the
        gateway that stalls the whole event loop for the callback's duration. Sync callbacks
        run inline. Bounded hooks keep ``plugins.hook_callback_timeout`` via ``asyncio.wait_for``
        (the coroutine is cancelled, not abandoned); a timed-out ``pre_tool_call`` fails closed.
        """
        from hermes_cli.plugins import _resolve_hook_callback_timeout
        if hook_name != "gateway_platform_event":
            kwargs.setdefault("telemetry_schema_version", OBSERVER_SCHEMA_VERSION)
        if hook_name == STREAM_HOLDING_HOOK:
            return await self._arun_output_transform_chain(kwargs)
        results: list[Any] = []
        timeout = _resolve_hook_callback_timeout()
        use_timeout = _hook_uses_callback_timeout(hook_name, timeout)
        fail_closed = hook_name in _HOOK_TIMEOUT_FAIL_CLOSED_HOOKS
        retryable_hook = hook_name in _HOOK_TIMEOUT_RETRY_HOOKS
        turn_deadline = (
            time.monotonic() + timeout * _HOOK_TURN_BUDGET_FACTOR
            if use_timeout and hook_name in _HOOK_TIMEOUT_DEGRADE_HOOKS else None
        )
        for cb in self._hooks.get(hook_name, []):
            callback_name = getattr(cb, "__name__", repr(cb))
            started = time.perf_counter()
            try:
                timed_out = False
                if use_timeout:
                    veto_grade = fail_closed and (
                        hook_name not in _HOOK_TIMEOUT_DEGRADE_HOOKS
                        or self._hook_policy(hook_name, cb)[0])
                    budget = timeout
                    if turn_deadline is not None:
                        budget = min(timeout, turn_deadline - time.monotonic())
                    if budget <= 0:
                        ret, timed_out = None, True
                    else:
                        ret, timed_out = await self._await_hook_callback(
                            hook_name, cb, kwargs, budget,
                            retry=retryable_hook and veto_grade)
                else:
                    ret = cb(**self._hook_callback_kwargs(cb, kwargs))
                    if inspect.isawaitable(ret):
                        ret = await ret
                if timed_out:
                    logger.warning(
                        "Hook '%s' callback %s timed out after %.0fs",
                        hook_name, callback_name, timeout,
                    )
                    self._record_gate_event(
                        hook_name, cb, kwargs, GateStatus.UNKNOWN,
                        detail="callback skipped or timed out",
                        duration_ms=(time.perf_counter() - started) * 1000,
                    )
                    if fail_closed:
                        directive = self._hook_timeout_directive(
                            hook_name, cb, kwargs, timeout=timeout, cooling_down=False,
                            retried=retryable_hook and veto_grade)
                        if directive is not None:
                            results.append(directive)
                    continue
                status = result_status(ret)
                action = str(ret.get("action", "")).lower() if isinstance(ret, Mapping) else ""
                self._record_gate_event(
                    hook_name, cb, kwargs, status,
                    detail=f"action={action}" if action else "",
                    duration_ms=(time.perf_counter() - started) * 1000,
                )
                if ret is not None:
                    results.append(ret)
            except TimeoutError:
                # An awaitable that itself raises TimeoutError is handled by the outer
                # exception clause below; only wait_for expiry reached this branch before.
                raise
            except (Exception, SystemExit) as exc:
                # Same isolation + failure contract as the sync path (#111922 warn-once, #109624
                # a raising policy guard fails closed).
                self._report_hook_failure(hook_name, cb, kwargs, exc)
                self._record_gate_event(
                    hook_name, cb, kwargs, GateStatus.UNKNOWN,
                    detail=f"{type(exc).__name__}: {exc}",
                    duration_ms=(time.perf_counter() - started) * 1000,
                )
                if fail_closed:
                    directive = self._hook_error_directive(hook_name, cb, exc)
                    if directive is not None:
                        results.append(directive)
        return results

    async def _arun_output_transform_chain(self, kwargs: dict[str, Any]) -> list[Any]:
        """Async counterpart of the ordered reply-transform chain."""
        from hermes_cli.plugins import _resolve_hook_callback_timeout

        original = str(kwargs.get("response_text") or "")
        current = original
        failed_callbacks: list[Callable] = []
        timeout = _resolve_hook_callback_timeout()
        use_timeout = _hook_uses_callback_timeout(STREAM_HOLDING_HOOK, timeout)
        for cb in self._hooks.get(STREAM_HOLDING_HOOK, []):
            payload = {
                **kwargs,
                "response_text": current,
                "original_response_text": original,
            }
            callback_name = getattr(cb, "__name__", repr(cb))
            started = time.perf_counter()
            try:
                ret = cb(**self._hook_callback_kwargs(cb, payload))
                if inspect.isawaitable(ret):
                    try:
                        ret = await (asyncio.wait_for(ret, timeout) if use_timeout else ret)
                    except TimeoutError:
                        self._record_gate_event(
                            STREAM_HOLDING_HOOK, cb, payload, GateStatus.UNKNOWN,
                            detail="callback skipped or timed out",
                            duration_ms=(time.perf_counter() - started) * 1000,
                        )
                        failed_callbacks.append(cb)
                        continue
                status = result_status(ret)
                self._record_gate_event(
                    STREAM_HOLDING_HOOK, cb, payload, status,
                    duration_ms=(time.perf_counter() - started) * 1000,
                )
            except TimeoutError:
                logger.warning(
                    "Hook '%s' callback %s timed out after %.0fs",
                    STREAM_HOLDING_HOOK, callback_name, timeout,
                )
                self._record_gate_event(
                    STREAM_HOLDING_HOOK, cb, payload, GateStatus.UNKNOWN,
                    detail="callback skipped or timed out",
                    duration_ms=(time.perf_counter() - started) * 1000,
                )
                failed_callbacks.append(cb)
                continue
            except (Exception, SystemExit) as exc:  # health: allow BLE001 -- plugin callbacks are untrusted; failures are isolated and surfaced as UNKNOWN
                self._report_hook_failure(STREAM_HOLDING_HOOK, cb, payload, exc)
                self._record_gate_event(
                    STREAM_HOLDING_HOOK, cb, payload, GateStatus.UNKNOWN,
                    detail=f"{type(exc).__name__}: {exc}",
                    duration_ms=(time.perf_counter() - started) * 1000,
                )
                failed_callbacks.append(cb)
                continue
            if isinstance(ret, str) and ret:
                current = ret
        return self._finish_output_transform_chain(original, current, failed_callbacks)

    def iter_hook_callbacks(self, hook_name: str) -> tuple[Callable, ...]:
        """Return a stable snapshot of callbacks registered for a hook."""
        from agent.safe_worker_policy import safe_worker_enabled

        if safe_worker_enabled():
            return ()
        return tuple(self._hooks.get(hook_name, ()))

    def render_system_prompt_sections(
        self, session_info: Mapping[str, Any]
    ) -> list[RenderedPluginSystemPromptSection]:
        """Render all registered sections deterministically and fail open."""
        from agent.safe_worker_policy import safe_worker_enabled

        if safe_worker_enabled():
            return []
        frozen_info = types.MappingProxyType(dict(session_info))
        rendered: list[RenderedPluginSystemPromptSection] = []
        total_chars = len(PLUGIN_SECTIONS_START) + len(PLUGIN_SECTIONS_END) + 2
        for _section_id, section in sorted(self._system_prompt_sections.items()):
            if len(rendered) >= MAX_SYSTEM_PROMPT_SECTIONS:
                logger.warning(
                    "Plugin system prompt section %s exceeded the section-count "
                    "budget (%d) and was skipped", section.id, MAX_SYSTEM_PROMPT_SECTIONS)
                continue
            text = self._render_prompt_section_text(section, frozen_info)
            if text is None:
                continue
            rendered_chars = len(format_system_prompt_section(section.id, text))
            if rendered:
                rendered_chars += 2  # canonical ``\n\n`` separator
            if total_chars + rendered_chars > MAX_SYSTEM_PROMPT_SECTIONS_TOTAL_CHARS:
                logger.warning(
                    "Plugin system prompt section %s (%s) exceeded the aggregate "
                    "session budget (%d chars) and was skipped", section.id, section.plugin,
                    MAX_SYSTEM_PROMPT_SECTIONS_TOTAL_CHARS)
                continue
            rendered.append(
                RenderedPluginSystemPromptSection(
                    id=section.id, content=text, position=section.position, plugin=section.plugin))
            total_chars += rendered_chars
            logger.info(
                "Session plugin prompt section: id=%s plugin=%s position=%s chars=%d", section.id,
                section.plugin, section.position, len(text))
        return rendered

    @staticmethod
    def _render_prompt_section_text(
        section: PluginSystemPromptSection, frozen_info: Mapping[str, Any]
    ) -> Optional[str]:
        """Evaluate one section; return its stripped text or None (with a warning) when skipped."""
        def _skip(detail: str, *args: Any) -> None:
            logger.warning(
                "Plugin system prompt section %s (%s) " + detail, section.id, section.plugin, *args)

        try:
            value = section.content(frozen_info) if callable(section.content) else section.content
        except (Exception, SystemExit) as exc:
            _skip("raised and was skipped: %s", exc)
            return None
        if not isinstance(value, str):
            _skip("returned %s, not str; skipped", type(value).__name__)
            return None
        text = value.strip()
        if not text:
            return None
        if PLUGIN_SECTIONS_START in text or PLUGIN_SECTIONS_END in text:
            _skip("contained a reserved persistence marker and was skipped")
            return None
        if len(text) > section.max_chars:
            _skip("exceeded max_chars (%d > %d) and was skipped", len(text), section.max_chars)
            return None
        return text

    def has_middleware(self, kind: str) -> bool:
        """Return True when at least one callback is registered for middleware."""
        from agent.safe_worker_policy import safe_worker_enabled

        if safe_worker_enabled():
            return False
        return bool(self._middleware.get(kind))

    def invoke_middleware(
        self, kind: str, *, _payload_key: Optional[str] = None, **kwargs: Any
    ) -> list[Any]:
        """Call middleware callbacks for *kind* (each isolated); return non-``None`` results.

        Request middleware passes ``_payload_key``: a dict returned under it becomes the payload the
        next callback sees, so rewrites compose, and each callback gets its own copy of the payload and
        of ``original_<key>`` so an in-place edit cannot leak.
        """
        from agent.safe_worker_policy import safe_worker_enabled
        from hermes_cli.middleware import _safe_copy

        if safe_worker_enabled():
            return []
        results: list[Any] = []
        for cb in self._middleware.get(kind, []):
            call_kwargs = kwargs
            if _payload_key:
                original_key = "original_" + _payload_key
                call_kwargs = {**kwargs, _payload_key: _safe_copy(kwargs[_payload_key]),
                               original_key: _safe_copy(kwargs[original_key])}
            try:
                ret = cb(**self._hook_callback_kwargs(cb, call_kwargs))  # narrow signatures, as hooks
                if ret is not None:
                    results.append(ret)
                    if _payload_key and isinstance(ret, dict) and isinstance(ret.get(_payload_key), dict):
                        kwargs[_payload_key] = ret[_payload_key]
            except (Exception, SystemExit) as exc:
                # Runs once per tool call like a hook, so a mis-declared callback floods identically.
                self._report_hook_failure(kind, cb, kwargs, exc, surface="Middleware")
        return results
