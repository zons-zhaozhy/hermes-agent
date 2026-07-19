"""
Agent 自我检查管理器 (SelfCheckManager)

内核组件——在每次工具调用前检查是否匹配已知失败模式。
解析所有已加载 skill 的 AVOID 章节 + 冗余检测规则，
匹配时在工具结果中注入警告（不阻断调用，让模型自主修正）。

约束：不触碰系统提示词（prompt caching 安全），只在 tool result 中注入。
"""

from __future__ import annotations

import json
import logging
import re  # noqa: E402 — pattern parser fundamentally needs regex for AVOID section extraction
from pathlib import Path

logger = logging.getLogger("agent.self_check")

# ── 冗余检测规则函数 ────────────────────────────────────────────────


def _r01_read_repeat(tool_name: str, args: dict, history: dict) -> str | None:
    """R01: 同一文件被 read_file >= 3 次。"""
    file_path = args.get("path", "")
    if tool_name != "read_file":
        return None
    key = "read_file:" + file_path
    if history.get(key, 0) >= 3:
        return (
            "文件 %s 已被 read_file 了 %d 次。"
            "文件内容不会变，不需要反复读取。每次 read_file 都在消耗 token 预算。"
            % (file_path, history[key])
        )
    return None


def _r02_patch_repeat(tool_name: str, args: dict, history: dict) -> str | None:
    """R02: 同一文件被 patch >= 3 次。"""
    file_path = args.get("path", "")
    if tool_name != "patch":
        return None
    key = "patch:" + file_path
    if history.get(key, 0) >= 3:
        return (
            "文件 %s 已被 patch 了 %d 次。"
            "考虑用 write_file 一次性重写整个文件——"
            "反复 patch 意味着第一轮方案没选对，继续 patch 只会越改越乱。"
            % (file_path, history[key])
        )
    return None


def _r03_read_after_edit(tool_name: str, args: dict, history: dict) -> str | None:
    """R03: read_file 刚被编辑过的文件。"""
    file_path = args.get("path", "")
    if tool_name != "read_file":
        return None
    recently_edited = history.get("_recently_edited", set())
    if file_path in recently_edited:
        return (
            "文件 %s 刚被编辑过，不需要重新读取。"
            "上一轮编辑的结果就是当前状态。"
            % file_path
        )
    return None


def _r04_terminal_fragments(tool_name: str, args: dict, history: dict) -> str | None:
    """R04: 5+ terminal 调用。"""
    if tool_name != "terminal":
        return None
    cnt = history.get("_terminal_count", 0) + 1
    if cnt >= 5:
        return (
            "已调用 terminal %d 次。"
            "考虑用 execute_code 编写 Python 脚本替代多个碎片化的 shell 命令——"
            "execute_code 里的 terminal() 在同一进程中运行，结果可以继续加工。"
            % cnt
        )
    return None


def _r05_write_without_read(tool_name: str, args: dict, history: dict) -> str | None:
    """R05: 编码前未读文件（首次写入/修改陌生文件）。"""
    file_path = args.get("path", "")
    if tool_name not in ("write_file", "patch"):
        return None
    all_read = history.get("_all_read", set())
    if file_path and file_path not in all_read and not file_path.startswith("/dev/"):
        return (
            "准备写入 %s 但本 session 尚未读取过它。"
            "先 read_file 确认当前内容，防止覆盖未提交的改动或基于过时的记忆编写代码。"
            % file_path
        )
    return None


# ── R06: 推责/归属语言检测 ─────────────────────────────────────────
# 检测 assistant 回复中把问题归因于他人/历史代码的推责语言。
# 这些表述是行为层面的自我开脱，不是技术分析。
_BLAME_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"不是我改的|不是我改出来的|不是这次(改动|变更|修改)"), r"不推责——视野内破损即责任，不说归属"),
    (re.compile(r"这是(上游|历史|之前|其他人|别的)的(问题|代码|改动|bug)"), r"不推责——追究来源不如直接修复"),
    (re.compile(r"不属于本[次轮个](修改|改动|修复|任务)"), r"不推责——看到就修，不区分范围"),
    (re.compile(r"这个是.*留[下存]的|(从以前|旧版本|版本)就有的"), r"不推责——历史债务也是债，当场清"),
]


def _r06_blame_shift(_tool_name: str, _args: dict, _history: dict) -> str | None:
    """R06: 推责/归属语言。由 check_response 驱动，工具参数忽略。

    注：此规则通过 check_response() 接收 assistant_content 参数，
    不在 check() 中触发（check 只有一个工具调用上下文）。
    """
    return None  # 不作为工具级规则，走 check_response 专用路径


# ── R07: 第一性原理——跳结论/凭推断/没分析就动手 ──────────────────
# 检测 speculative language + 跳分析直接给方案。
_R07_SPECULATION_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"应该(是|可以|没问题)|大概(是|没问题)|一般来说|正常情况下|大概率"), r"第一性原理——'应该/大概/一般来说'是推断不是验证，加 [实测] 或 [文档] 标注"),
    (re.compile(r"(直接|简单|快速).{0,10}(修复|改|处理|解决)(一下|即可|就好)"), r"第一性原理——'直接修复'之前先拆解根因，不要跳分析就动手"),
]


# ── R08: 推责给用户——六类推活回用户的语句 ────────────────────────
_R08_USER_BLAME_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"需要你做的|需用户.{0,5}(操作|执行|运行|配置)"), r"禁止推责——自己排查修复，不把活推回用户"),
    (re.compile(r"^(要不要|是否需要|要不要我).{0,20}[？?]$", re.MULTILINE), r"禁止推责——分析清楚直接做，不问用户"),
    (re.compile(r"请确认一下|请确认是否|请帮我确认"), r"禁止推责——自己验证，不让用户确认"),
    (re.compile(r"我无法直接执行|不支持.{0,10}(这个|该)操作"), r"禁止推责——换一条路执行，不报告障碍"),
    (re.compile(r"(你要|你想)我修(哪个|哪|什么|哪些)"), r"禁止推责——主动修全部，不问用户选什么"),
    (re.compile(r"(还需|仍需|要)手动.{0,10}(操作|执行|处理|改|配|运行)"), r"禁止推责——找程序化替代方案，不把手工操作推给用户"),
]


# ── R09: 验证驱动——声称结论缺少证据标注 ──────────────────────────
# 检测断言式结论未搭配证据来源标注。
# 触发模式：声称某种状态/事实，但不跟任何 [实测]/[文档]/[推断]/[未查证] 标签。
_R09_EVIDENCE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"(根因|原因)(是|在于|出在).{0,40}[。，\n]"), r"验证驱动——声称根因但没有 [实测] [文档] [推断] 证据标注"),
    (re.compile(r"(已经|已)(修复|解决|完成|搞定)(?!.*\d+.*通过).{0,30}[。，\n]"), r"验证驱动——声称完成但没有验证证据（测试输出/日志/文件内容）"),
    (re.compile(r"(确认|确定|保证|肯定|绝对).{0,40}[。，\n]"), r"验证驱动——断言确认但没有 [实测] 证据，加一句验证命令"),
]


def _r07_first_principles(_tool_name: str, _args: dict, _history: dict) -> str | None:
    """R07: 第一性原理。由 check_response 驱动。"""
    return None


def _r08_user_delegation(_tool_name: str, _args: dict, _history: dict) -> str | None:
    """R08: 推责给用户。由 check_response 驱动。"""
    return None


def _r09_evidence_driven(_tool_name: str, _args: dict, _history: dict) -> str | None:
    """R09: 验证驱动。由 check_response 驱动。"""
    return None


# ── R10: 链式工具路由——3+ 步应合并为 execute_code ──────────────────
# 检测模式：search_files → read_file → patch/write_file/terminal 链
_CHAIN_WINDOW: int = 4  # 看最近 N 个工具调用
_CHAIN_TRIGGER_SEQUENCES: list[tuple[tuple[str, ...], str]] = [
    (
        ("search_files", "read_file", "patch", "terminal"),
        "工具链——检测到 search→read→patch→验证 模式，走 execute_code 一次性流水线更快更省 token",
    ),
    (
        ("search_files", "read_file", "write_file", "terminal"),
        "工具链——检测到 search→read→write→验证 模式，走 execute_code 一次性流水线",
    ),
    (
        ("search_files", "read_file", "patch"),
        "工具链——检测到 search→read→patch 三次往返，合并为 execute_code 一步跑完",
    ),
]


def _r10_chain_route(tool_name: str, _args: dict, history: dict) -> str | None:
    """R10: 链式工具路由。检测工具调用链并提示合并。

    通过 history["_recent_tools"] 追踪最近 N 次工具调用，
    匹配预定义链式模式后注入 execute_code 提示。
    """
    recent: list[str] = history.get("_recent_tools", [])  # type: ignore[assignment]
    if len(recent) < 3:
        return None

    for sequence, hint in _CHAIN_TRIGGER_SEQUENCES:
        seq_len = len(sequence)
        if len(recent) < seq_len:
            continue
        # 检查最近 seq_len 个调用是否匹配序列
        window = recent[-seq_len:]
        if tuple(window) == sequence[:seq_len]:
            return hint

    return None


# 规则注册表
# 只读查询类工具——不对其参数做 AVOID 关键词匹配。
# AVOID 条目是行为警示（"不要硬编码/不要吞异常"），只对执行类工具有意义。
# 对 search_files/read_file/web_search 等只读工具匹配就是误报。
_READ_ONLY_TOOLS: frozenset[str] = frozenset({
    "search_files",
    "read_file",
    "web_search",
    "web_extract",
    "browser_snapshot",
    "browser_console",
    "browser_back",
    "browser_scroll",
    "browser_get_images",
    "session_search",
    "skill_view",
    "skills_list",
    "memory",
    "vision_analyze",
})

_RULES = [
    ("R01", "medium", _r01_read_repeat),
    ("R02", "high", _r02_patch_repeat),
    ("R03", "medium", _r03_read_after_edit),
    ("R04", "low", _r04_terminal_fragments),
    ("R05", "medium", _r05_write_without_read),
    ("R06", "high", _r06_blame_shift),
    ("R07", "medium", _r07_first_principles),
    ("R08", "high", _r08_user_delegation),
    ("R09", "medium", _r09_evidence_driven),
    ("R10", "medium", _r10_chain_route),
]

_SEVERITY_EMOJI = {"high": "\U0001f534", "medium": "\U0001f7e1", "low": "\U0001f7e2"}
_SEVERITY_EMOJI["high"] = "\U0001f534"  # 🔴
_SEVERITY_EMOJI["medium"] = "\U0001f7e1"  # 🟡
_SEVERITY_EMOJI["low"] = "\U0001f7e2"  # 🟢


class SelfCheckManager:
    """内核自检组件——在 tool_executor 中作为 pre-tool 检查运行。

    指标：
    - 零缓存影响：警告注入到 tool result，不碰 system prompt
    - 零阻断：只警告不阻止，模型自主决定是否采纳
    - 零网络调用：所有检查是纯内存操作
    """

    def __init__(self):
        self._avoid_entries: list[tuple[str, str, str]] = []  # [(keyword, text, source)]
        self._call_history: dict[str, int | set] = {}
        self._all_read: set[str] = set()
        self._recently_edited: set[str] = set()
        self._recent_tools: list[str] = []  # 最近 N 个工具调用名链
        self._loaded = False

    # ── 加载 ────────────────────────────────────────────────────────

    def load(self, agent=None) -> None:
        """从所有已加载 skill 中解析 AVOID 章节。agent 参数可选。"""
        count = 0
        skill_dirs_seen: set[str] = set()

        try:
            from agent.skill_commands import scan_skill_commands
            commands = scan_skill_commands()
            for _slug, info in commands.items():
                skill_dir_str = info.get("skill_dir", "")
                if not skill_dir_str or skill_dir_str in skill_dirs_seen:
                    continue
                skill_dirs_seen.add(skill_dir_str)
                count += self._load_skill_avoid(Path(skill_dir_str), info.get("name", _slug))
        except Exception as e:
            logger.warning("SelfCheck: repo skills scan skipped: %s", e)

        # 用户安装的 skill
        try:
            from hermes_constants import get_hermes_home
            user_skills = get_hermes_home() / "skills"
            if user_skills.exists():
                for skill_md in user_skills.rglob("SKILL.md"):
                    skill_dir_str = str(skill_md.parent)
                    if skill_dir_str in skill_dirs_seen:
                        continue
                    skill_dirs_seen.add(skill_dir_str)
                    count += self._load_skill_avoid(skill_md.parent, skill_md.parent.name)
        except Exception as e:
            logger.warning("SelfCheck: user skills scan skipped: %s", e)

        self._loaded = True
        if count:
            logger.info("SelfCheckManager: loaded %d AVOID entries from skills", count)

    def _load_skill_avoid(self, skill_dir: Path, name: str) -> int:
        """加载单个 skill 的 AVOID 章节。返回条目数。"""
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            return 0
        try:
            content = skill_md.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning("SelfCheck: cannot read %s: %s", skill_md, e)
            return 0
        return self._parse_avoid_section(content, name)

    def _parse_avoid_section(self, content: str, source: str) -> int:
        """从 markdown 中解析 ## AVOID 章节。返回条目数。"""
        avoid_match = re.search(r'## AVOID\s*\n((?:(?:[-*]\s+).*\n?)+)', content)
        if not avoid_match:
            return 0
        avoid_text = avoid_match.group(1)
        count = 0
        for line in avoid_text.strip().split("\n"):
            line = line.strip()
            if not line or not re.match(r'^[-*]\s', line):
                continue
            item = line[2:].strip()
            if not item:
                continue
            # 清理格式
            clean = re.sub(r'^(AVOID|avoid):\s*', "", item)
            clean = re.sub(r"\*\*([^*]+)\*\*", r"\1", clean)
            clean = re.sub(r'[`"\']', "", clean)
            keywords = clean.lower()
            self._avoid_entries.append((keywords, item, source))
            count += 1
        return count

    # ── 检查 ────────────────────────────────────────────────────────

    def check(self, tool_name: str, args: dict) -> str | None:
        """检查当前工具调用是否匹配已知失败模式。

        Returns:
            None if clean, or a warning string to prepend to the tool result.
        """
        if not self._loaded:
            self.load()

        self._update_history(tool_name, args)
        warnings: list[str] = []

        # 1. 冗余检测规则（确定性匹配）
        for rule_id, severity, check_fn in _RULES:
            try:
                warning = check_fn(tool_name, args, self._call_history)
                if warning:
                    emoji = _SEVERITY_EMOJI.get(severity, "\u26aa")
                    warnings.append("%s [%s] %s" % (emoji, rule_id, warning))
            except Exception as e:
                logger.warning("SelfCheck: rule %s check raised: %s", rule_id, e)

        # 2. AVOID 关键词匹配（模糊匹配，需重合度）
        # 只读查询类工具跳过——它们的参数是搜索词不是行为，匹配 AVOID 是误报
        if tool_name not in _READ_ONLY_TOOLS:
            try:
                tool_desc = (tool_name + " " + json.dumps(args, ensure_ascii=False)).lower()
            except (TypeError, ValueError) as e:
                logger.warning("SelfCheck: json.dumps(args) failed for %s: %s", tool_name, e)
                tool_desc = tool_name.lower()
            for keywords, item, source in self._avoid_entries:
                kw_parts = keywords.split()
                if len(kw_parts) >= 3:
                    matches = sum(1 for kw in kw_parts if kw in tool_desc)
                    if matches >= 2:
                        warnings.append("\U0001f4cb [%s] \U0001f4a1 %s" % (source, item))

        if warnings:
            return "[SelfCheck]\n" + "\n".join(warnings)
        return None

    def check_response(self, assistant_content: str | None) -> str | None:
        """检查 assistant 回复文本是否包含推责/归属语言。

        与 check() 不同：check() 检查工具调用参数，check_response() 检查回复文本。
        在 tool_executor 中，对每个 assistant_message 先调 check_response()，
        再对每个 tool_call 调 check()。

        Returns:
            None if clean, or a warning string to prepend to the first tool result.
        """
        if not assistant_content:
            return None

        warnings: list[str] = []
        # R06: 推责/归属
        for pattern, hint in _BLAME_PATTERNS:
            if pattern.search(assistant_content):
                warnings.append("\U0001f534 [R06] %s" % hint)
        # R07: 第一性原理——跳结论/凭推断
        for pattern, hint in _R07_SPECULATION_PATTERNS:
            if pattern.search(assistant_content):
                warnings.append("\U0001f7e1 [R07] %s" % hint)
        # R08: 推责给用户
        for pattern, hint in _R08_USER_BLAME_PATTERNS:
            if pattern.search(assistant_content):
                warnings.append("\U0001f534 [R08] %s" % hint)
        # R09: 验证驱动——缺证据
        for pattern, hint in _R09_EVIDENCE_PATTERNS:
            if pattern.search(assistant_content):
                warnings.append("\U0001f7e1 [R09] %s" % hint)

        if warnings:
            return "[SelfCheck]\n" + "\n".join(warnings)
        return None

    def _update_history(self, tool_name: str, args: dict) -> None:
        """更新调用计数器。"""
        file_path = args.get("path", "")
        if file_path:
            if tool_name in ("read_file", "patch", "write_file"):
                key = tool_name + ":" + file_path
                self._call_history[key] = self._call_history.get(key, 0) + 1  # type: ignore[operator]
            if tool_name == "read_file":
                self._all_read.add(file_path)
            if tool_name in ("write_file", "patch"):
                self._recently_edited.add(file_path)
                self._all_read.add(file_path)

        if tool_name == "terminal":
            cnt = self._call_history.get("_terminal_count", 0)
            self._call_history["_terminal_count"] = int(cnt) + 1  # type: ignore[arg-type]

        # 传递引用给检查函数
        self._call_history["_recently_edited"] = self._recently_edited  # type: ignore[index]
        self._call_history["_all_read"] = self._all_read  # type: ignore[index]

        # R10 链式工具路由：追踪最近 N 个工具调用名
        self._recent_tools.append(tool_name)
        if len(self._recent_tools) > _CHAIN_WINDOW:
            self._recent_tools = self._recent_tools[-_CHAIN_WINDOW:]
        self._call_history["_recent_tools"] = self._recent_tools  # type: ignore[index]

    # ── 查询 ────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """返回自检统计。"""
        return {
            "avoid_entries": len(self._avoid_entries),
            "call_history_keys": len(self._call_history),
            "files_read": len(self._all_read),
            "files_edited": len(self._recently_edited),
        }


# 单例 —— 由 AIAgent 持有
_global_manager: SelfCheckManager | None = None


def get_self_check() -> SelfCheckManager | None:
    return _global_manager


def set_self_check(manager: SelfCheckManager | None) -> None:
    global _global_manager
    _global_manager = manager
