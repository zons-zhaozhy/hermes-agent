#!/usr/bin/env python3
"""Difficulty-classifier evaluation harness for the adaptive-reasoning plugin.

Ground truth definition (measured, not guessed): a message's *sufficiency
level* is the LOWEST effort at which the eval model answers it correctly.
- routing below sufficiency  -> accuracy loss (underthinking)
- routing above sufficiency  -> wasted reasoning tokens (overthinking)

The harness runs each benchmark item at each effort level, records
correctness + reasoning tokens, and saves the grid to a results file so
classifier iterations can be evaluated offline (free, reproducible).

Env-driven (no hardcoded credentials, endpoints, or models):
    ADAPTIVE_EVAL_API_KEY   API key (required)
    ADAPTIVE_EVAL_BASE_URL  OpenAI-compatible base URL (required)
    ADAPTIVE_EVAL_MODEL     model id (required)
    ADAPTIVE_EVAL_RESULTS   results file path (optional)
    ADAPTIVE_EVAL_PLUGIN    path to the plugin __init__.py (optional;
                            defaults to <repo>/plugins/adaptive-reasoning/
                            __init__.py relative to this file)

Usage:
    python3 eval_benchmark.py run     # measure (real API calls)
    python3 eval_benchmark.py eval    # score classifiers offline
"""

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
RESULTS = Path(os.environ.get(
    "ADAPTIVE_EVAL_RESULTS", str(HERE / "eval_results.json")))
PLUGIN = Path(os.environ.get(
    "ADAPTIVE_EVAL_PLUGIN",
    str(HERE.parent / "plugins" / "adaptive-reasoning" / "__init__.py")))

# ── Benchmark: 8 categories × known-answer items ──────────────────────────
BENCH = [
    ("brevity", "ok", ["ok", "okay"]),
    ("brevity", "continue", ["continu"]),
    ("brevity", "好的，继续", ["继续", "好的"]),
    ("trivial-fact", "What is the capital of France? Answer with one word.", ["paris"]),
    ("trivial-fact", "Which planet is the largest in the solar system? One word.", ["jupiter"]),
    ("arithmetic", "What is 17 * 23? Answer with just the number.", ["391"]),
    ("arithmetic", "Compute 128 / 8 + 3. Answer with just the number.", ["19"]),
    ("crt-trap", "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How many cents does the ball cost? Answer with just the number.", ["5"]),
    ("crt-trap", "A farmer has 17 sheep. All but 9 run away. How many are left? Answer with just the number.", ["9"]),
    ("crt-trap", "If you're running a race and you pass the person in 2nd place, what place are you in? Answer with just the ordinal.", ["2nd", "second"]),
    ("short-hard", "Compute det([[2,3,1],[4,7,2],[3,5,9]]). Answer with just the number.", ["15"]),
    ("short-hard", "How many trailing zeros does 375! have? Answer with just the number.", ["93"]),
    ("short-hard", "Alice is twice as old as Bob was when Alice was as old as Bob is now. Alice is 30. How old is Bob? Answer with just the number.", ["22.5", "22½", "45/2"]),
    ("short-hard", "Sort by orbital period, shortest first: Mars, Venus, Europa, Titan. Answer with commas.", ["europa, titan, venus, mars"]),
    ("agent-task", "Read plugins/adaptive-reasoning/__init__.py, find the middleware function, and summarize in one sentence what it rewrites.", ["effort", "reasoning"]),
    ("agent-task", "帮我排查这个问题：为什么下面的 Python 循环在删除列表元素时跳过了某些项？items = [1,2,2,3]; [items.remove(x) for x in items[:]]。请给出根因和修法。", ["迭代", "remove", "跳过", "copy", "切片", "快照"]),
    ("long-easy", ("Hi team! Just wanted to check in quickly and see how everyone is doing this week. "
                   "It's been a while since our last catch-up and I figured a quick hello was in order. "
                   "Nothing urgent on my side at all - everything is going smoothly and there are no blockers "
                   "or issues or problems or concerns of any kind whatsoever. I just wanted to say hello, "
                   "share a friendly greeting, and wish everyone a wonderful rest of the week. Take care all! "
                   "Hope the weather is nice wherever you are and that you all have great days ahead of you."), ["hello", "hi team", "thank", "你好", "likewise", "you too", "care"]),
    ("long-easy", ("I was thinking about lunch earlier today. There is a noodle place around the corner "
                   "that I quite like - they do a very good spicy beef noodle soup and the portions are "
                   "generous. Last time I went there I ordered the soup and some dumplings and it was all "
                   "very tasty and filling and reasonably priced too. Anyway I just wanted to mention it "
                   "in case anyone is interested in going there sometime this week. No pressure at all."), ["noodle", "soup", "dumpling", "sounds", "好", "interested"]),
    ("hard-long", ("A warehouse has three robots. Robot A picks 40 items/hour, B picks 60, C picks 50. "
                   "A picks for the first 2 hours, then C replaces A for 1 hour, 2 hours total are lost "
                   "to a lunch break during which nobody picks, then A and C work together "
                   "for 90 minutes, then B works alone for 45 minutes. Half of all picked items were "
                   "mis-scanned and had to be re-picked at 50% speed by B alone. What is the total number "
                   "of GOOD items? Answer with just the number."), ["202"]),
    ("hard-long", ("Prove or disprove: for all positive integers n, the number of divisors of n is odd "
                   "if and only if n is a perfect square. Then state the smallest n > 100 for which the "
                   "divisor count is odd, and give its divisor count. Answer as: verdict, n, count."), ["true", "121", "9"]),
]

LEVELS = ["low", "medium", "high"]


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(
            f"{name} not set — export it before running the eval harness")
    return value


def run() -> None:
    from openai import OpenAI
    client = OpenAI(
        api_key=_require_env("ADAPTIVE_EVAL_API_KEY"),
        base_url=_require_env("ADAPTIVE_EVAL_BASE_URL"),
    )
    model = _require_env("ADAPTIVE_EVAL_MODEL")
    grid = {}
    for cat, prompt, accepts in BENCH:
        row = {}
        for level in LEVELS:
            t0 = time.time()
            try:
                r = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    reasoning_effort=level,
                    max_tokens=2048,
                )
                text = (r.choices[0].message.content or "").strip()
                det = getattr(r.usage, "completion_tokens_details", None)
                row[level] = {
                    "ok": any(a in text.lower() for a in accepts),
                    "reasoning": getattr(det, "reasoning_tokens", None) if det else None,
                    "latency": round(time.time() - t0, 1),
                }
            except Exception as exc:  # noqa: BLE001 - record and continue
                row[level] = {"error": str(exc)[:120]}
            time.sleep(0.6)
        grid[prompt] = {"cat": cat, "accepts": accepts, "runs": row}
        print(f"[done] {cat}: {prompt[:40]!r}", flush=True)
    RESULTS.write_text(json.dumps(grid, ensure_ascii=False, indent=1))
    print(f"saved -> {RESULTS}")


def sufficiency(entry: dict) -> str:
    """Lowest level answering correctly; None if model never correct."""
    runs = entry["runs"]
    for lv in LEVELS:
        if runs.get(lv, {}).get("ok"):
            return lv
    return None


def eval_classifiers() -> None:
    import importlib.util
    spec = importlib.util.spec_from_file_location("ar", PLUGIN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    grid = json.loads(RESULTS.read_text())
    print(f"{'category':<12} {'sufficiency':<11} {'plugin':<7} verdict")
    under = over = 0
    wasted = saved = 0
    for prompt, entry in grid.items():
        suff = sufficiency(entry)
        if suff is None:
            print(f"{entry['cat']:<12} {'NEVER':<11} - model wrong at all levels, excluded")
            continue
        pred = mod.classify_effort(prompt)
        # minimal/low share the lowest rung: wire maps both to native low
        if pred == "minimal":
            pred = "low"
        idx = {"low": 0, "medium": 1, "high": 2}
        delta = idx[pred] - idx[suff]
        rt = lambda lv: entry["runs"].get(lv, {}).get("reasoning") or 0  # noqa: E731
        if delta < 0:
            verdict = "UNDERTHINK (accuracy risk)"
            under += 1
        elif delta > 0:
            verdict = f"overthink (+{rt(pred) - rt(suff)} tok)"
            over += 1
            wasted += rt(pred) - rt(suff)
        else:
            verdict = "exact"
        saved_rt = rt("medium") - rt(pred) if idx[pred] < 1 else 0
        saved += max(0, saved_rt)
        print(f"{entry['cat']:<12} {suff:<11} {pred:<7} {verdict}")
    print(f"\nsummary: underthink={under}  overthink={over}  "
          f"wasted_tok_vs_sufficiency={wasted}  saved_tok_vs_static_medium={saved}")


if __name__ == "__main__":
    {"run": run, "eval": eval_classifiers}[sys.argv[1]]()
