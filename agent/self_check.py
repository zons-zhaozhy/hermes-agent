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
    (re.compile(r"不是我(这次|的)?(改的|改出来的|写的|弄的|引入的)"), r"不推责——视野内破损即责任，不说归属"),
    (re.compile(r"这是(上游|历史|之前|其他人|别的).{0,20}的(问题|代码|改动|bug)"), r"不推责——追究来源不如直接修复"),
    (re.compile(r"不属于本[次轮个](修改|改动|修复|任务)"), r"不推责——看到就修，不区分范围"),
    (re.compile(r"这个是.*留[下存]的|(从以前|旧版本|版本)就有的"), r"不推责——历史债务也是债，当场清"),
    (re.compile(r"不是这次(改动|变更|修改|改的)"), r"不推责——视野内破损即责任，不说归属"),
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
# 启发式证据分类器：不依赖大模型，用多个结构信号判断文本是否含验证证据。
# 设计原则：证据有 3 个维度 — ①显式标签 [实测]/[文档]/[推断]
# ②数据密度（数字+单位+结果）③外部引用（文件行号/日志时间戳/异常堆栈）

def _has_evidence(content: str) -> bool:
    """启发式判断段落是否包含可验证的证据。纯 Python，零依赖，毫秒级。"""
    # 1. 显式标签
    if re.search(r"\[实测\]|\[文档\]|\[推断\]", content):
        return True

    # 2. 测试/运行结果：N/N 通过、exit_code、状态码
    if re.search(r"\d+/\d+\s*(通过|pass|ok|green|passed)", content, re.I):
        return True
    if re.search(r"(exit.?code|返回码|status.?code)\s*[=:]\s*\d+", content, re.I):
        return True
    if re.search(r"(passed|failed|skipped|error)\s*[=:]\s*\d+", content, re.I):
        return True
    if re.search(r"\d+\s*(通过|失败|跳过|pass|fail|skip)", content, re.I):
        return True

    # 3. 文件引用：path/to/file.py:行号
    if re.search(r"[/\w]+\.(py|java|go|ts|tsx|js|rs|yaml|yml|sql):\d+", content):
        return True

    # 4. 日志/终端输出：时间戳、traceback、stdout/stderr
    if re.search(r"\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}", content):
        return True
    if re.search(r"Traceback\s*\(most recent call last\)", content, re.I):
        return True
    if re.search(r"(stdout|stderr|output)\s*:", content, re.I):
        return True

    # 5. 数量级数据：数字+单位，暗示实测
    if re.search(r"\d+\s*(个|条|次|行|s|ms|MB|KB|并发|qps|tps)", content):
        return True
    if re.search(r"\d+\s*(files|tests|errors|matches|results|records)", content, re.I):
        return True

    # 6. 文件/路径存在性声明
    if re.search(r"[/\w]+\s*(存在|不存在|已创建|已删除|is\s+at|located\s+at)", content):
        return True

    return False


# R09 模式：只检测"声称"的句子，实际判断由 _has_evidence 完成
_R09_EVIDENCE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"(根因|原因)(是|在于|出在).{0,60}[。，\n]"), r"验证驱动——声称根因但没有验证证据（测试输出/日志/文件内容）"),
    (re.compile(r"(已经|已)(修复|解决|完成|搞定).{0,40}[。，\n]"), r"验证驱动——声称完成但没有验证证据（测试输出/日志/文件内容）"),
    (re.compile(r"(确认|确定|保证|肯定|绝对).{0,40}[。，\n]"), r"验证驱动——断言确认但没有 [实测] 证据，加一句验证命令"),
]


# ── R11: 判断阶段——给方案但没列替代方案对比 ──────────────────────
# 决策框架要求"判断"阶段输出多方案对比（方案A/B/C + 优劣分析），
# 直接跳到"修复方案是X"而没有任何对比框架 = 跳过判断。
_R11_JUDGMENT_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"(方案|做法|方法)\s*(很简单|就是简单|没什么|非常容易)"), r"判断阶段——'方案很简单'=跳过了复杂性分析，decision-framework 要求先列替代方案再选"),
    (re.compile(r"(直接|只要|仅仅|简单)(改|修复|用|换|替换|写).{0,20}(就行|就好|即可|可以)"), r"判断阶段——直接跳到实施没有分析路径，先列2-3种替代方案再选"),
]


# ── R12: 验证阶段——声称完成但验证证据不在结尾 ────────────────────
# 决策框架要求"验证"是最后阶段。如果回复声称完成但验证证据
# 不在结尾附近，说明框架流程被破坏（先声称再事后验证）。
def _r12_verify_position(assistant_content: str, _has_evidence_fn=None) -> str | None:
    """R12: 验证阶段位置检测——两阶段逻辑。

    阶段①：全文任一句子声称完成（"完成/搞定/done/已修复"）
    阶段②：证据不在结尾附近
    两个都成立 → 验证流程被破坏（宣告和验证脱节）
    """
    if not assistant_content:
        return None

    # 阶段①：全局扫描——全文任一句子声称完成？
    if not re.search(r"(完成|搞定|done|fixed|resolved|已修复|已解决)", assistant_content, re.I):
        return None

    # 阶段②：局部检查——结尾是否有验证证据？
    sentences = [s.strip() for s in re.split(r"[。\n]", assistant_content) if s.strip()]
    if not sentences:
        return None

    end_chunk = " ".join(sentences[-2:])
    if _has_evidence_fn and _has_evidence_fn(end_chunk):
        return None  # 验证在结尾，正确

    return "验证阶段——声称完成但验证证据不在结尾，decision-framework 要求验证是最后一步"


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
    ("R11", "medium", _r07_first_principles),  # placeholder — response-level only
    ("R12", "medium", _r09_evidence_driven),   # placeholder — uses custom fn
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
        # R09: 验证驱动——缺证据（匹配整段文本，用 _has_evidence 判断）
        for pattern, hint in _R09_EVIDENCE_PATTERNS:
            match = pattern.search(assistant_content)
            if match:
                # 检查匹配到的句子所在上下文是否含证据
                # 取匹配位置前后各 200 字符做证据检测
                start = max(0, match.start() - 200)
                end = min(len(assistant_content), match.end() + 200)
                evidence_context = assistant_content[start:end]
                if not _has_evidence(evidence_context):
                    warnings.append("\U0001f7e1 [R09] %s" % hint)
        # R11: 判断阶段——给方案但没列替代方案对比
        for pattern, hint in _R11_JUDGMENT_PATTERNS:
            if pattern.search(assistant_content):
                # 检查整段文本是否有对比框架（方案A/B、①/②、对比/替代/权衡）
                has_comparison = bool(re.search(r"方案\s*[A-Za-z①②③]|①|②|③|\bvs\b|比较|权衡|对比|另一个|替代方案|哪个更好", assistant_content, re.I))
                if not has_comparison:
                    warnings.append("\U0001f7e1 [R11] %s" % hint)
        # R12: 验证阶段——声称完成但验证证据不在结尾
        r12_warning = _r12_verify_position(assistant_content, _has_evidence)
        if r12_warning:
            warnings.append("\U0001f7e1 [R12] %s" % r12_warning)

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
