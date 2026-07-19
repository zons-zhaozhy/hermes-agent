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
    """R05: 编码前未读文件（首次写入/修改陌生文件）。

    用 update_history 前的快照判断——否则新文件已被加入 _all_read 导致永远不触发。
    """
    file_path = args.get("path", "")
    if tool_name not in ("write_file", "patch"):
        return None
    all_read = history.get("_all_touched_before_update", set())
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
    (re.compile(r"不是我(这次|的)?(改的|改出来的|写的|弄的|引入的|代码)"), r"不推责——视野内破损即责任，不说归属"),
    (re.compile(r"这是(上游|历史|之前|其他人|别的).{0,20}的(问题|代码|改动|bug)"), r"不推责——追究来源不如直接修复"),
    (re.compile(r"不属于本[次轮个](修改|改动|修复|任务)"), r"不推责——看到就修，不区分范围"),
    (re.compile(r"这个是.{0,200}留[下存]的|(从以前|旧版本|版本)就有的"), r"不推责——历史债务也是债，当场清"),
    (re.compile(r"不是这次(改动|变更|修改|改的)"), r"不推责——视野内破损即责任，不说归属"),
]


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
    (
        ("search_files", "read_file", "write_file"),
        "工具链——检测到 search→read→write 三次往返，合并为 execute_code 一步跑完",
    ),
    (
        ("read_file", "patch", "terminal"),
        "工具链——读→改→验证可合并为 execute_code 一次性跑完",
    ),
    (
        ("read_file", "write_file", "terminal"),
        "工具链——读→写→验证可合并为 execute_code 一次性跑完",
    ),
]


def _r10_chain_route(tool_name: str, _args: dict, history: dict) -> str | None:
    """R10: 链式工具路由。检测工具调用链并提示合并。

    两类检测：
    1. 预定义序列匹配（search→read→patch 等）——exact match
    2. 连续 terminal 碎片化（≥3 个 terminal）——计数 match
    """
    recent: list[str] = history.get("_recent_tools", [])  # type: ignore[assignment]
    if len(recent) < 3:
        return None

    # 检测 1：预定义序列匹配
    for sequence, hint in _CHAIN_TRIGGER_SEQUENCES:
        seq_len = len(sequence)
        if len(recent) < seq_len:
            continue
        window = recent[-seq_len:]
        if tuple(window) == sequence[:seq_len]:
            return hint

    # 检测 2：连续 3+ terminal 调用（碎片化 shell 替代品）
    terminal_streak = 0
    for t in reversed(recent):
        if t == "terminal":
            terminal_streak += 1
        else:
            break
    if terminal_streak >= 3:
        return (
            "工具链——连续 %d 次 terminal 调用，合并为 execute_code 一次性脚本更快更可控"
            % terminal_streak
        )

    return None


# ── R13: 任务漂移——写入与已操作文件零目录重叠 ────────────────────
# 检测模式：write_file/patch 写入的文件路径与 session 中已读取/编辑的
# 文件零目录重叠 → 大概率是不相关的额外改动（drive-by edit）。
def _r13_task_drift(tool_name: str, args: dict, history: dict) -> str | None:
    """R13: 任务漂移检测。

    逻辑：写入文件的目录（parent dir）与所有已操作文件（update 前快照）
    零交集 → 任务漂移信号。排除首次写入（没有历史时不触发）。
    """
    if tool_name not in ("write_file", "patch"):
        return None

    file_path = args.get("path", "")
    if not file_path or file_path.startswith("/dev/"):
        return None

    # 用 update_history 前的快照——避免新文件已被加入集合导致漏检
    all_touched = history.get("_all_touched_before_update", set())
    if len(all_touched) < 3:
        return None

    if file_path in all_touched:
        return None

    from pathlib import PurePosixPath
    new_dir = str(PurePosixPath(file_path).parent)

    overlap = any(
        str(PurePosixPath(f).parent) == new_dir
        for f in all_touched
        if f and f != file_path
    )
    if overlap:
        return None  # 同目录 → 相关

    return (
        "任务漂移——写入 %s 与本次 session 已操作的文件零目录重叠。"
        "检查这是否是 drive-by edit：每行 git diff 必须能回答'用户要求改的吗'"
        % file_path
    )


# ── R14: 操作结果忽略——工具输出含错误但回复无引用 ─────────────────
# 检测模式：上次工具输出含 ERROR/FAILED/Traceback 但 assistant 回复
# 中没有任何对该错误的引用（"错误"/"失败"/"失败原因"等）。
_R14_ERROR_SIGNALS: list[re.Pattern] = [
    re.compile(r"Traceback\s*\(most recent call last\)", re.I),
    re.compile(r"\bERROR\b.*[:：]", re.I),
    re.compile(r"\bFAILED\b", re.I),
    re.compile(r"\bError:\s", re.I),
    re.compile(r"exit_code\s*[=:]\s*[1-9]"),
]

_R14_ACK_PATTERNS: re.Pattern = re.compile(
    r"错误|失败|异常|报错|exit.?code|错误码|error|failed|traceback|stack.?trace|出错|没通过|不通过|崩溃",
    re.I,
)


def _r14_result_ignored(tool_name: str, args: dict, history: dict) -> str | None:
    """R14: 操作结果忽略检测。

    逻辑：history["_last_tool_output"] 含错误信号，但当前 assistant_content
    中没有引用该错误 → 忽略了错误结果。

    注：此规则在 check_response() 中调用（需要 assistant_content），
    但放这里作为函数文档。工具级 check() 中不做检测。
    """
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
    # ── 工具级规则：check() 中调用 ──
    ("R01", "medium", _r01_read_repeat),
    ("R02", "high", _r02_patch_repeat),
    ("R03", "medium", _r03_read_after_edit),
    ("R04", "low", _r04_terminal_fragments),
    ("R05", "medium", _r05_write_without_read),
    ("R10", "medium", _r10_chain_route),
    ("R13", "medium", _r13_task_drift),
]
# 回复级规则（R06-R09, R11-R12, R14）：在 check_response() 中专有路径处理，
# 不走 check() 的 _RULES 循环——下面注释仅作文档用。
_RESPONSE_RULE_IDS: frozenset[str] = frozenset({"R06", "R07", "R08", "R09", "R11", "R12", "R14"})

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
        # 闭环修正引擎：钱学森控制论——测量必须驱动修正
        self._rule_fire_count: dict[str, int] = {}  # rule_id → 连续命中次数
        self._rule_last_turn: set[str] = set()  # 上一轮命中的规则
        self._feynman_triggered: set[str] = set()  # 已触发学习的规则(去重)
        # R13/R14：需要外部调用 record_tool_result() 传入工具输出
        self._last_tool_output: str = ""  # 最近一次工具输出文本
        self._all_touched_before_update: set[str] = set()  # update_history 前快照

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

        # R13 需要在 update_history 前做快照——否则新文件已被加入 all_read
        self._all_touched_before_update = (self._all_read | self._recently_edited).copy()
        self._call_history["_all_touched_before_update"] = self._all_touched_before_update  # type: ignore[index]

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

        try:
            return self._do_check_response(assistant_content)
        except Exception:
            logger.exception("SelfCheck: check_response raised — returning None")
            return None

    def _do_check_response(self, assistant_content: str) -> str | None:
        """check_response 的核心逻辑。包裹在 try/except 中防止单点崩溃。"""
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
        # R14: 操作结果忽略——上次工具输出含错误信号但回复无引用
        if self._last_tool_output:
            has_error = any(p.search(self._last_tool_output) for p in _R14_ERROR_SIGNALS)
            if has_error:
                acknowledges = bool(_R14_ACK_PATTERNS.search(assistant_content))
                if not acknowledges:
                    warnings.append(
                        "\U0001f534 [R14] 操作结果忽略——上次工具输出含 ERROR/FAILED/Traceback，"
                        "但回复中没有任何对该错误的引用。必须分析错误原因再继续"
                    )

        if warnings:
            # ── 闭环修正引擎升级逻辑 ─────────────────────────────────
            self._update_rule_fire_counts(warnings)
            self._inject_corrections(warnings)
            return "[SelfCheck]\n" + "\n".join(warnings)
        else:
            # 本轮无命中 → 重置命中计数（连续命中需要持续性）
            self._rule_fire_count.clear()
        return None

    # ── 闭环修正引擎方法 ───────────────────────────────────────────

    # 每条规则对应的强制修正指令（钱学森闭环：测量→修正）
    _CORRECTION_DIRECTIVES: dict[str, str] = {
        "R06": "修正动作：删除归属描述，只陈述技术事实不涉及谁改的",
        "R07": "修正动作：用 [实测] 或 [文档] 替换推断词（应该/大概/一般），或标注 [推断] 并补验证",
        "R08": "修正动作：删除请求句，替换为自己执行的方案——不说'需要你做的'，说'我来做'",
        "R09": "修正动作：补充证据——贴 terminal output / 测试结果 / 日志摘录，或加 [实测] 标签",
        "R11": "修正动作：补充替代方案——至少列出方案A/B并对比优劣后再选",
        "R12": "修正动作：将验证证据移到回复结尾——decision-framework 要求验证是最后一步",
        "R14": "修正动作：引用上次工具输出中的错误信息并分析原因，不要忽略错误结果",
    }

    def record_tool_result(self, tool_name: str, tool_output: str) -> None:
        """R14 支撑：记录最近一次工具输出文本，供 check_response() 做错误引用检测。

        由 tool_executor 在工具执行完成后调用。
        """
        if not tool_output:
            return
        # 只保留最近一次，截断到 4000 字符防止内存膨胀
        self._last_tool_output = str(tool_output)[:4000]

    def _update_rule_fire_counts(self, warnings: list[str]) -> None:
        """从警告中提取规则ID，更新连续命中计数。"""
        current_rules: set[str] = set()
        for w in warnings:
            for rule_id in self._CORRECTION_DIRECTIVES:
                if "[%s]" % rule_id in w:
                    current_rules.add(rule_id)
                    self._rule_fire_count[rule_id] = self._rule_fire_count.get(rule_id, 0) + 1

        # 上一轮命中但本轮未命中 → 重置（不再连续）
        for rule_id in self._rule_last_turn - current_rules:
            if rule_id in self._rule_fire_count:
                del self._rule_fire_count[rule_id]
        self._rule_last_turn = current_rules

    def _inject_corrections(self, warnings: list[str]) -> None:
        """按控制论闭环原则升级警告：命中1次=警告，2次=警告+修正建议，3次=触发费曼学习。"""
        for i, w in enumerate(warnings):
            for rule_id, directive in self._CORRECTION_DIRECTIVES.items():
                if "[%s]" % rule_id not in w:
                    continue
                count = self._rule_fire_count.get(rule_id, 0)

                # 第 2 次命中 → 加修正建议
                if count == 2:
                    warnings[i] = w + "\n  \u2699 [%s] %s（第%d次）" % (rule_id, directive, count)

                # 第 3 次命中 → 触发费曼学习循环（第 3 层自动触发）
                if count >= 3 and rule_id not in self._feynman_triggered:
                    self._feynman_triggered.add(rule_id)
                    warnings[i] = (
                        w + "\n  \u2699 [%s] %s（第%d次）\n"
                        "  \U0001f9e0 [Feynman] 同类规则连续命中%d次→触发费曼学习循环：\n"
                        "    ① 写出你对'%s'的理解 ② 用自己的话解释为什么反复触发\n"
                        "    ③ 找知识缺口 ④ 重构理解并写入 memory"
                    ) % (rule_id, directive, count, count, rule_id)

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

    # ── 自审计：验证每条规则确实在工作 ──────────────────────────────
    # 金丝雀测试：对每条规则注入已知应命中/不应命中的用例，
    # 验证规则被调用 + 正例命中 + 反例不命中。
    # 三类失败可被检测：
    # 1) 规则没注册（dead） 2) 数据没流入（stale） 3) 正则永不匹配（broken）

    _CANARY_TOOL_CASES: list[dict] = [
        # 工具级规则：通过 check() 注入
        {"rule": "R01", "type": "positive", "desc": "read_file×3 同文件",
         "ops": [("read_file", {"path": "/canary/r01.py"})] * 3,
         "final": ("read_file", {"path": "/canary/r01.py"})},
        {"rule": "R01", "type": "negative", "desc": "read_file×1",
         "ops": [("read_file", {"path": "/canary/r01_neg.py"})],
         "final": ("read_file", {"path": "/canary/r01_neg.py"})},
        {"rule": "R02", "type": "positive", "desc": "patch×3 同文件",
         "ops": [("patch", {"path": "/canary/r02.py", "old_string": "a", "new_string": "b"})] * 3,
         "final": ("patch", {"path": "/canary/r02.py", "old_string": "a", "new_string": "b"})},
        {"rule": "R03", "type": "positive", "desc": "编辑后重读",
         "ops": [("write_file", {"path": "/canary/r03.py", "content": "x"}), ("read_file", {"path": "/canary/r03.py"})],
         "final": ("read_file", {"path": "/canary/r03.py"})},
        {"rule": "R04", "type": "positive", "desc": "terminal×5",
         "ops": [("terminal", {"command": "ls"})] * 4,
         "final": ("terminal", {"command": "ls"})},
        {"rule": "R05", "type": "positive", "desc": "write 未读文件",
         "ops": [],
         "final": ("write_file", {"path": "/canary/r05_new.py", "content": "x"})},
        {"rule": "R05", "type": "negative", "desc": "write 已读文件",
         "ops": [("read_file", {"path": "/canary/r05_read.py"})],
         "final": ("write_file", {"path": "/canary/r05_read.py", "content": "x"})},
        {"rule": "R10", "type": "positive", "desc": "search→read→patch",
         "ops": [("search_files", {"pattern": "x"}), ("read_file", {"path": "/canary/r10.py"})],
         "final": ("patch", {"path": "/canary/r10.py", "old_string": "a", "new_string": "b"})},
        {"rule": "R13", "type": "positive", "desc": "3文件/src 写/docs",
         "ops": [("read_file", {"path": "/src/a.py"}), ("read_file", {"path": "/src/b.py"}), ("patch", {"path": "/src/c.py", "old_string": "x", "new_string": "y"})],
         "final": ("write_file", {"path": "/docs/drift.md", "content": "x"})},
        {"rule": "R13", "type": "negative", "desc": "同目录不漂移",
         "ops": [("read_file", {"path": "/src/a.py"}), ("read_file", {"path": "/src/b.py"}), ("patch", {"path": "/src/c.py", "old_string": "x", "new_string": "y"})],
         "final": ("write_file", {"path": "/src/d.py", "content": "x"})},
    ]

    _CANARY_RESPONSE_CASES: list[dict] = [
        # 回复级规则：通过 check_response() 注入
        {"rule": "R06", "type": "positive", "text": "不是我改的bug。"},
        {"rule": "R06", "type": "negative", "text": "根因在匹配逻辑。"},
        {"rule": "R07", "type": "positive", "text": "应该没问题。"},
        {"rule": "R07", "type": "negative", "text": "已修复。[实测]"},
        {"rule": "R08", "type": "positive", "text": "需要你做的：改配置。"},
        {"rule": "R08", "type": "negative", "text": "我来修复。"},
        {"rule": "R09", "type": "positive", "text": "根因是连接池太小。"},
        {"rule": "R09", "type": "negative", "text": "根因是连接池 [实测]：500并发通过。"},
        {"rule": "R11", "type": "positive", "text": "方案很简单，加个白名单就行。"},
        {"rule": "R11", "type": "negative", "text": "方案A加白名单；方案B加分类器。选A因为更简洁。"},
        {"rule": "R12", "type": "positive", "text": "已修复。本次改动不涉及配置。"},
        {"rule": "R12", "type": "negative", "text": "已修复。\n验证：69 passed in 0.74s"},
    ]

    def audit(self) -> dict:
        """自审计：验证每条规则的金丝雀测试是否通过。

        返回 {"alive": int, "dead": int, "broken": int, "details": [...]}
        - alive: 正例命中且反例不命中 = 规则正常
        - dead:  规则未被调用或数据未流入 = 空转
        - broken: 正例未命中或反例误命中 = 正则坏了
        """
        results: list[dict] = []
        rule_status: dict[str, str] = {}  # rule_id → "alive"/"dead"/"broken"

        # ── 工具级规则审计 ──
        for case in self._CANARY_TOOL_CASES:
            rule_id = case["rule"]
            ctype = case["type"]
            mgr = SelfCheckManager()
            mgr._loaded = True
            for tn, args in case["ops"]:
                mgr.check(tn, args)
            final_tn, final_args = case["final"]
            result = mgr.check(final_tn, final_args)
            hit = bool(result) and ("[%s]" % rule_id in result)
            results.append({"rule": rule_id, "type": ctype, "hit": hit, "desc": case["desc"]})

        # ── 回复级规则审计 ──
        for case in self._CANARY_RESPONSE_CASES:
            rule_id = case["rule"]
            ctype = case["type"]
            mgr = SelfCheckManager()
            mgr._loaded = True
            result = mgr.check_response(case["text"])
            hit = bool(result) and ("[%s]" % rule_id in result)
            results.append({"rule": rule_id, "type": ctype, "hit": hit, "desc": case["text"][:40]})

        # ── R14 特殊审计：需要 record_tool_result ──
        # 正例：ERROR 输出 + 无引用回复
        mgr14p = SelfCheckManager()
        mgr14p._loaded = True
        mgr14p.record_tool_result("terminal", "Traceback (most recent call last)\nError: x")
        r14p = mgr14p.check_response("已修复。")
        results.append({"rule": "R14", "type": "positive", "hit": bool(r14p) and "[R14]" in r14p, "desc": "ERROR无引用"})
        # 反例：ERROR 输出 + 有引用回复
        mgr14n = SelfCheckManager()
        mgr14n._loaded = True
        mgr14n.record_tool_result("terminal", "exit_code=1\nFAILED")
        r14n = mgr14n.check_response("上次命令失败了，原因分析中。")
        results.append({"rule": "R14", "type": "negative", "hit": bool(r14n) and "[R14]" in r14n, "desc": "ERROR有引用"})

        # ── 汇总判定 ──
        all_rules = set(r["rule"] for r in results)
        for rule_id in all_rules:
            positives = [r for r in results if r["rule"] == rule_id and r["type"] == "positive"]
            negatives = [r for r in results if r["rule"] == rule_id and r["type"] == "negative"]
            pos_ok = all(r["hit"] for r in positives) if positives else False
            neg_ok = all(not r["hit"] for r in negatives) if negatives else True
            if pos_ok and neg_ok:
                rule_status[rule_id] = "alive"
            elif not pos_ok:
                rule_status[rule_id] = "broken"
            else:
                rule_status[rule_id] = "broken"  # 反例误命中

        alive = sum(1 for v in rule_status.values() if v == "alive")
        broken = sum(1 for v in rule_status.values() if v == "broken")

        return {
            "alive": alive,
            "broken": broken,
            "total": len(rule_status),
            "status": rule_status,
            "details": results,
        }


# 单例 —— 由 AIAgent 持有
_global_manager: SelfCheckManager | None = None


def get_self_check() -> SelfCheckManager | None:
    return _global_manager


def set_self_check(manager: SelfCheckManager | None) -> None:
    global _global_manager
    _global_manager = manager
