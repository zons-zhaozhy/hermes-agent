"""sandwich-inject plugin — 三明治架构：确定性代码包抄概率 LLM。

上层（Pre-hook 强制注入）：不让 Agent 做选择题。
    pre_llm_call 每次请求前把场景 SOP + 知识库内容注入 user message，
    并附"禁止自行猜测"工作守则。知识已注入，Agent 无需自行检索。

下层（Post-hook 工具契约校验）：不做完别想走。
    从 conversation_history 校验上轮是否调用了该场景必需的工具，
    未调用则追加【校验报错】指令要求重做；连续 N 次违规输出
    NEED_HUMAN_INTERVENTION 转人工。

零核心改动：仅挂现有 pre_llm_call hook 点（返回值拼入 user message，
框架自带超大输出 spill 保护，不会膨胀每轮 prompt）。

ACTIVATION MODEL
================
插件默认关闭，仅在匹配到场景配置时激活。

配置：~/.hermes/sandwich.yaml（或 HERMES_SANDWICH_CONFIG 指定路径）

    scenes:
      - name: aml-review
        match_keywords: ["反洗钱", "可疑交易", "AML"]
        sop: |
          1. 必须先查数据库获取原始数据
          2. 必须引用监管条款（央行令〔2016〕第3号）
          3. 输出必须包含合规声明
        knowledge: |
          最新产品能力说明（业务语言，不含技术架构细节）
        knowledge_files:
          - /abs/path/to/kb.md
        required_tools:
          - mcp__dbhub__execute_sql_aml_v7
          - terminal
        max_retries: 3

匹配规则：场景按 match_keywords 命中 user_message 激活。激活后该会话
每轮 pre_llm_call 都注入 SOP+知识，并校验必需工具。

无场景匹配 = 插件零行为 = 正常 agent。
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── 会话级违规计数（进程内，不落盘）───────────────────────────
# key: session_id, value: {scene_name: {"violations": int, "active": bool}}
_VIOLATION_TRACKER: Dict[str, Dict[str, Dict[str, Any]]] = {}
_TRACKER_LOCK = threading.Lock()

_CONFIG_CACHE: Dict[str, tuple[float, Optional[dict]]] = {}  # path -> (mtime, data)
_CONFIG_LOCK = threading.Lock()

# knowledge_files 内容缓存: path -> (mtime, content)
_KB_CACHE: Dict[str, tuple[float, str]] = {}
_KB_LOCK = threading.Lock()

# 会话级场景锁定: session_id -> scene dict (首个命中后锁定, 防多轮失活)
_SESSION_SCENES: Dict[str, dict] = {}
_SESSION_LOCK = threading.Lock()

_MAX_RETRIES_DEFAULT = 3

# 长跑进程（gateway）会话数无界增长防护：超过上限时清理最旧条目
_MAX_TRACKED_SESSIONS = 512


def _bounded_insert(d: Dict[str, Any], key: str, value: Any) -> None:
    """Insert into a session-keyed dict, evicting oldest when over capacity.

    Contract:
      Preconditions: d is a dict, key is a str, value is Any
      Postconditions: d[key] == value; len(d) <= _MAX_TRACKED_SESSIONS;
        oldest-inserted keys evicted first
    """
    if key in d:
        d[key] = value
        return
    if len(d) >= _MAX_TRACKED_SESSIONS:
        # dict 保持插入序：弹出最旧 key
        d.pop(next(iter(d)))
    d[key] = value


def _config_path() -> Path:
    """Resolve the sandwich config file path.

    Contract:
      Preconditions: none
      Postconditions: returns a Path (HERMES_SANDWICH_CONFIG override, else
        HERMES_HOME/sandwich.yaml); never raises
    """
    override = os.environ.get("HERMES_SANDWICH_CONFIG")
    if override:
        return Path(override).expanduser()
    home = Path(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes")))
    return home / "sandwich.yaml"


def _load_config() -> Optional[dict]:
    """Load sandwich.yaml with mtime-based cache. None if missing/unparseable.

    Contract:
      Preconditions: none
      Postconditions: returns the parsed config dict or None; caches by
        mtime; never raises (load failures log a warning and return None)
    """
    path = _config_path()
    try:
        if not path.exists():
            return None
        mtime = path.stat().st_mtime
        with _CONFIG_LOCK:
            cached = _CONFIG_CACHE.get(str(path))
            if cached and cached[0] == mtime:
                return cached[1]
        import yaml  # type: ignore

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        with _CONFIG_LOCK:
            _CONFIG_CACHE[str(path)] = (mtime, data)
        return data
    except Exception as exc:
        logger.warning("[Sandwich] config load failed: %s", exc)
        return None


def _load_knowledge_files(files: Optional[List[str]]) -> str:
    """Read knowledge files with mtime cache, join contents.

    Contract:
      Preconditions: files is None or a list of path strings
      Postconditions: returns joined file contents ("" when empty/missing);
        cached by file mtime so unchanged files are read once per change;
        missing or unreadable files are skipped with a warning, never raised
    """
    if not files:
        return ""
    parts: List[str] = []
    for f in files:
        try:
            p = Path(f).expanduser()
            if not p.exists():
                logger.warning("[Sandwich] knowledge file missing: %s", f)
                continue
            mtime = p.stat().st_mtime
            with _KB_LOCK:
                cached = _KB_CACHE.get(str(p))
                if cached and cached[0] == mtime:
                    parts.append(cached[1])
                    continue
            content = p.read_text(encoding="utf-8")
            with _KB_LOCK:
                _KB_CACHE[str(p)] = (mtime, content)
            parts.append(content)
        except Exception as exc:
            logger.warning("[Sandwich] knowledge file read failed %s: %s", f, exc)
    return "\n\n".join(parts)


def _match_scene(cfg: Optional[dict], user_message: str) -> Optional[dict]:
    """First scene whose keywords hit user_message. None if no match.

    Contract:
      Preconditions: cfg is None or a dict with optional "scenes" list
      Postconditions: returns the first matching scene dict or None;
        scenes with empty match_keywords never match; never raises
    """
    if not cfg or not isinstance(cfg, dict):
        return None
    for scene in cfg.get("scenes", []) or []:
        if not isinstance(scene, dict):
            continue
        keywords = scene.get("match_keywords") or []
        if not keywords:
            continue
        if any(str(k) in user_message for k in keywords):
            return scene
    return None


def _last_turn_tool_calls(conversation_history: List[dict]) -> List[str]:
    """Tool names called in the most recent assistant turn.

    Walks the tail of conversation_history; returns the tool names of the
    last assistant message that made tool calls. Empty if none.

    Contract:
      Preconditions: conversation_history is a list of message dicts
      Postconditions: returns only the most recent assistant turn's tool
        names; never raises on malformed messages; returns [] when no
        assistant turn exists
    """
    if not conversation_history:
        return []
    for msg in reversed(conversation_history):
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "assistant":
            continue
        # 只看最近一条 assistant 消息：它没有 tool_calls 就是没调用
        # （三明治语义：不回溯更早轮次，最后一轮不查库就作答 = 违规）
        tool_calls = msg.get("tool_calls") or []
        return [
            tc.get("function", {}).get("name", "")
            for tc in tool_calls
            if isinstance(tc, dict) and tc.get("function", {}).get("name")
        ]
    return []


def _render_injection(scene: dict, violations: int, max_retries: int) -> str:
    """Build the injected context block for a matched scene.

    Contract:
      Preconditions: scene is a dict with optional sop/knowledge/
        knowledge_files/required_tools; violations >= 0; max_retries >= 1
      Postconditions: returns a non-empty string containing the SOP,
        knowledge, required-tool contract, and work rules; missing files
        are skipped, never raised
    """
    sop = str(scene.get("sop") or "").strip()
    knowledge = str(scene.get("knowledge") or "").strip()
    kb_files = _load_knowledge_files(scene.get("knowledge_files"))
    required = scene.get("required_tools") or []
    req_list = "\n".join(f"- {t}" for t in required)

    parts: List[str] = []
    if sop:
        parts.append(f"【强制SOP - 必须严格遵循】\n{sop}")
    kb_all = "\n\n".join(p for p in (knowledge, kb_files) if p)
    if kb_all:
        parts.append(f"【已注入的最新知识 - 禁止自行猜测】\n{kb_all}")
    if required:
        parts.append(
            f"【必需工具契约 - 本轮必须调用】\n{req_list}\n"
            f"完成前必须调用上述全部工具，否则输出将被拒绝重试。"
        )
    if violations > 0:
        parts.append(
            f"【校验报错】你已连续 {violations}/{max_retries} 轮未满足工具契约，"
            f"请立即补上必需工具调用，不得再次跳过。"
        )
    parts.append(
        "工作守则：\n"
        "1. 你无需再检索任何内容，最新最准确的知识已注入。\n"
        "2. 严格按照 SOP 步骤执行。\n"
        "3. 如知识不足以回答，输出 NEED_HUMAN_INTERVENTION。"
    )
    return "\n\n".join(parts)


def on_pre_llm_call(**kwargs) -> Optional[Dict[str, Any]]:
    """Sandwich gate: inject SOP/knowledge + verify required-tool contract.

    Returns {"context": ...} consumed by the framework (injected into the
    user message). Returns None when no scene matches (zero behavior).

    Contract:
      Preconditions: kwargs contains user_message (str); conversation_history
        optional
      Postconditions: never raises; returns None when no scene matches the
        user message; returns a context dict otherwise; violation counter
        increments on missing required tools and resets on full coverage;
        NEED_HUMAN_INTERVENTION returned after max_retries consecutive
        violations
    """
    user_message = str(kwargs.get("user_message") or "")
    if not user_message:
        return None

    session_id = str(kwargs.get("session_id") or kwargs.get("task_id") or "default")

    # ── 场景解析：会话级锁定（首个命中后锁定，防多轮关键词失活）──
    with _SESSION_LOCK:
        scene = _SESSION_SCENES.get(session_id)
        if scene is None:
            cfg = _load_config()
            scene = _match_scene(cfg, user_message)
            if scene is not None:
                _bounded_insert(_SESSION_SCENES, session_id, scene)
    if scene is None:
        return None

    scene_name = str(scene.get("name") or "unnamed")
    max_retries = int(scene.get("max_retries") or _MAX_RETRIES_DEFAULT)

    # ── Post-hook 工具契约校验：上轮是否调用了必需工具 ──
    history = kwargs.get("conversation_history") or []
    required = set(scene.get("required_tools") or [])
    called = set(_last_turn_tool_calls(history))

    # 违规计数 read-modify-write 全程持锁（多会话并发安全）
    with _TRACKER_LOCK:
        state = _VIOLATION_TRACKER.get(session_id)
        if state is None:
            state = {}
            _bounded_insert(_VIOLATION_TRACKER, session_id, state)
        if scene_name not in state:
            state[scene_name] = {"violations": 0, "active": True}
        state = state[scene_name]
        if required:
            missing = required - called
            if missing and state["active"]:
                state["violations"] += 1
                if state["violations"] >= max_retries:
                    state["active"] = False
            elif not missing:
                # 本轮补齐了契约 → 清零违规计数
                state["violations"] = 0
        violations = state["violations"]
        active = state["active"]

    if required and missing and not active:
        # 已转人工且仍缺工具：保持 NEED_HUMAN_INTERVENTION 信号
        return {
            "context": (
                f"【强制SOP】\n{scene.get('sop') or '（无 SOP）'}\n\n"
                f"你已连续 {violations} 轮违反工具契约"
                f"（缺少: {sorted(missing)}）。\n"
                f"NEED_HUMAN_INTERVENTION — 转人工处理。"
            )
        }

    injection = _render_injection(scene, violations, max_retries)
    return {"context": injection}


def register(ctx) -> None:
    ctx.register_hook("pre_llm_call", on_pre_llm_call)
    logger.info(
        "sandwich-inject registered: inactive until ~/.hermes/sandwich.yaml "
        "defines a scene matching the user message"
    )
