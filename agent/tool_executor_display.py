"""Pure presentation helpers for tool execution."""


def preview(text: str, limit: int) -> str:
    return text[:limit] + "..." if len(text) > limit else text


def delegate_spinner_label(function_args: dict) -> str:
    action = str(function_args.get("action") or "").strip().lower()
    tasks = function_args.get("tasks")
    if action in ("list", "steer", "stop"):
        return f"🔀 subagent {action}"
    if tasks and isinstance(tasks, list):
        return f"🔀 delegating {len(tasks)} tasks · (/agents to monitor)"
    goal_preview = (function_args.get("goal") or "")[:30]
    return f"🔀 {goal_preview} · (/agents to monitor)" if goal_preview else "🔀 delegating · (/agents to monitor)"
