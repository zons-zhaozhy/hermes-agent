"""
瞎猜根治闸门插件（用户 0826 拍板：纪律要机械强制，不要口头承诺）
=================================================================

针对的三种瞎猜（均有 0826 实锤案例）：
  1. 命令标识符未查证就使用——deploy.sh cloud loom / loom-frontend 连错两次，
     正名 loom-backend 在 build.sh --list 注册表里。规则：带服务名的 deploy/build
     命令，服务名必须出现在本会话成功执行过的 --list 输出里，否则 block 先查。
  2. 报错后原样重试——CDP 失联时同命令连发。规则：与上次失败命令相同的命令
     （归一化空白后逐字相同）→ block；同一命令累计失败 2 次后再发 → block 逼换策略。
  3. 「应该是/大概」当事实——由失败后 pre_llm_call 注入三条纪律提醒兜底
     （LLM 内部状态无法程序检测，只能失败时刻提醒）。

实现纪律：
  - 只用 shlex 切令牌 + 集合成员判断，无正则扫文本（防 CPU100% 陷阱）。
  - 状态经 plugins._shared_state 会话隔离，自动过期清理。
  - 只拦 terminal 工具；只读命令（--list/ps/ls/grep）永不拦。

Contract:
  Preconditions: plugin system provides pre_tool_call / post_tool_call hooks
                 with session_id, tool_name, args, result(exit_code).
  Postconditions:
    - identical-to-last-failed command blocked;
    - command failed twice blocked until its normalized form changes;
    - deploy/build service-name command blocked unless name appears in
      session's last successful --list output;
    - read-only commands never blocked.
  Invariants: plugin never raises out of hooks; never blocks --list itself.
"""

import json
import logging
import shlex

from plugins._shared_state import get_session_state

logger = logging.getLogger(__name__)

_STATE_PREFIX = "no_guessing"

# ============ 累教不改升级机制（0826 用户拍板：三级惩罚阶梯） ============
# L1 默认: 拦截+正解提示
# L2 累犯(30天窗口内同规则>=3次): 拦截消息前置累犯警告+注入针对性上下文
# L3 顽劣(30天窗口内同规则>=8次): 收窄放行面, 14天无新违规降级
_VIOLATION_WINDOW_DAYS = 30
_DEMOTION_WINDOW_DAYS = 14
_L2_THRESHOLD = 3
_L3_THRESHOLD = 8
_L3_SLEEP_LIMIT = 3  # L3 时 R5 的 sleep 上限从 10s 收窄到 3s


def _db_path():
    """Contract: 返回 outcomes.db 路径（复用既有库, 不新建）。"""
    import os
    from pathlib import Path
    home = os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))
    return Path(home) / "outcomes.db"


def _ensure_violations_table(conn):
    conn.execute(
        "CREATE TABLE IF NOT EXISTS violations ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " rule TEXT NOT NULL,"
        " command TEXT NOT NULL,"
        " level TEXT,"
        " session_id TEXT,"
        " timestamp TEXT NOT NULL)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_viol_rule_ts ON violations(rule, timestamp)"
    )


def _record_violation(rule, command, level, session_id):
    """写入前科档。Contract: 失败静默(库不可用不阻断拦截), 返回是否成功。"""
    import sqlite3
    import datetime
    try:
        conn = sqlite3.connect(_db_path(), timeout=5)
        try:
            _ensure_violations_table(conn)
            conn.execute(
                "INSERT INTO violations (rule, command, level, session_id, timestamp) "
                "VALUES (?,?,?,?,?)",
                (rule, command[:500], level, session_id,
                 datetime.datetime.now().isoformat(timespec="seconds")),
            )
            conn.commit()
            return True
        finally:
            conn.close()
    except Exception as e:  # noqa: D5 — 档案失败不阻断拦截主流程
        logger.warning("no-guessing: violations 记档失败 %s", e)
        return False


def _violation_stats(rule):
    """查询规则前科。Contract: 返回 (window_count_30d, last_ts, days_since_last); 库不可用返回 (0,None,None)。"""
    import sqlite3
    import datetime
    try:
        conn = sqlite3.connect(_db_path(), timeout=5)
        try:
            _ensure_violations_table(conn)
            cutoff = (
                datetime.datetime.now()
                - datetime.timedelta(days=_VIOLATION_WINDOW_DAYS)
            ).isoformat(timespec="seconds")
            row = conn.execute(
                "SELECT COUNT(*), MAX(timestamp) FROM violations "
                "WHERE rule=? AND timestamp>=?",
                (rule, cutoff),
            ).fetchone()
            cnt, last = row[0] or 0, row[1]
            days = None
            if last:
                days = (
                    datetime.datetime.now()
                    - datetime.datetime.fromisoformat(last)
                ).days
            return cnt, last, days
        finally:
            conn.close()
    except Exception as e:  # noqa: D5
        logger.warning("no-guessing: violations 查档失败 %s", e)
        return 0, None, None


def _current_level(rule):
    """Contract: 返回 'L1'|'L2'|'L3'——L3 需 30 天窗口计数>=8 且最近一次违规在 14 天内。"""
    cnt, _last, days_since = _violation_stats(rule)
    if cnt >= _L3_THRESHOLD and days_since is not None and days_since <= _DEMOTION_WINDOW_DAYS:
        return "L3"
    if cnt >= _L2_THRESHOLD:
        return "L2"
    return "L1"


def _block_with_escalation(rule, command, base_message, session_id):
    """统一拦截出口：升级消息 + 记档 + L2 注入上下文。
    Contract: 总是返回 block dict; 副作用=violations 表新增一行。"""
    level = _current_level(rule)
    cnt, last, _days = _violation_stats(rule)
    _record_violation(rule, command, level, session_id)
    msg = base_message
    if level != "L1":
        msg = (
            f"⚠ [累犯升级 {level}] 规则 {rule} 30 天内第 {cnt + 1} 次违规"
            f"（上次: {last}）。再犯将收窄放行面——这条规则你已证明靠不住，"
            f"必须从源头改习惯。\n\n{base_message}"
        )
    out = {"action": "block", "message": msg}
    if level == "L2":
        out["context"] = (
            f"[累犯上下文] 规则 {rule} 已违规 {cnt} 次（30 天窗口），"
            f"最近一次 {last}。本次拦截为第 {cnt + 1} 次。"
        )
    return out


def _state(session_id):
    """Contract: 返回本插件命名空间的会话状态 dict。"""
    return get_session_state(session_id, namespace=_STATE_PREFIX)

_BLOCK_IDENTICAL = (
    "[NO-GUESSING BLOCK] 这条命令与上一次失败的命令逐字相同——禁止原样重试。\n"
    "先读上次报错全文，定位到 文件:行号 或明确拒绝原因，改完命令再发。"
)

_BLOCK_TWICE = (
    "[NO-GUESSING BLOCK] 这条命令本会话已失败 2 次——禁止第三次。"
    "换策略：改思路、换工具、或先跑只读命令拿事实。"
)

_BLOCK_UNVERIFIED_NAME = (
    "[NO-GUESSING BLOCK] 命令携带服务名 {names}，但本会话没有执行过 "
    "build.sh/deploy.sh --list 拿到注册表输出，无法证明该名字存在。\n"
    "瞎猜服务名已被实锤多次（loom/loom-frontend 均为错名）。"
    "先跑: bash deploy/build.sh --list （或对应 --list），确认正名后再部署。"
)

_REMINDER_AFTER_FAIL = (
    "[瞎猜纪律提醒] 刚发生工具失败。三条铁律：\n"
    "1. 标识符（名字/路径/端点/账号/配置值）写入命令前，必须先有当轮只读工具输出它；\n"
    "2. 报错后禁原样重试——先读错误全文定位根因；同错两次必须换策略；\n"
    "3. 「应该是/大概是」= 未查证。要么查，要么明说未查证，禁止当命令参数。"
)

# 触发服务名核验的命令片段（按子串判断，集合小且稳定）
_NAME_CHECK_MARKERS = ("deploy.sh cloud", "cloud-svc-run", "build.sh")
# 永不拦截的只读标志
_READONLY_PASSES = ("--list", "-h", "--help")

# 规则4：长日志未预过滤——docker logs / docker compose logs 必须带 --tail / -f 配合 grep，
# 或本身含 grep/awk/head 管道。裸 docker logs 会把全量日志灌进上下文。
_BLOCK_RAW_LOGS = (
    "[NO-GUESSING BLOCK] docker logs 裸奔未预过滤——全量日志会撑爆上下文。\n"
    "先预过滤: docker logs {name} --tail 100 2>&1 | grep -E 'ERROR|WARN' | head -20\n"
    "或按关键词取根因行，禁止拉完整堆栈。"
)

# 规则5：纯 sleep 干等轮询（长任务铁律：>60s 或 network 一律 background+notify）
_BLOCK_SLEEP_LOOP = (
    "[NO-GUESSING BLOCK] 检测到 sleep 干等轮询。铁律：长任务(>60s/含网络)一律 "
    "background=true + notify_on_complete=true，用 process wait/poll 管理，禁止 sleep 干等。"
)

# 规则6：诊断命令吞错——2>/dev/null 会把报错证据扔掉，违反 log-first 诊断纪律
# （仅拦诊断类命令；编译/构建输出重定向属正常用法不拦）
_BLOCK_SWALLOW_ERR = (
    "[NO-GUESSING BLOCK] 诊断命令带 2>/dev/null——报错信息是定位根因的第一证据，禁止吞掉。\n"
    "诊断一律 2>&1 保留错误流；确要过滤输出用 grep 管道而非丢弃 stderr。"
)


def _is_diagnostic_command(command: str) -> bool:
    """Contract: 只判定诊断类命令（日志/健康/网络/DB 探测），编译构建不在此列。
    真机 0826 实锤误拦: grep 的 stderr 几乎无诊断价值（2>/dev/null 只防「文件不存在」噪声），
    grep/ls/cat 类常规检索不拦。"""
    diagnostic_markers = (
        "docker logs", "docker compose logs", "curl ", "psql", "redis-cli",
        "kubectl logs", "lsof ", "ps aux", "journalctl", "health",
    )
    return any(m in command for m in diagnostic_markers)


def _check_raw_logs(command: str):
    """规则4：docker logs 无 --tail 且无 grep/awk/head/tail 管道 → block。"""
    if "docker logs" not in command and "docker compose logs" not in command:
        return None
    if "--tail" in command or "--since" in command:
        return None
    if any(p in command for p in ("grep", "awk", "head", "tail -", "| tail")):
        return None
    return _BLOCK_RAW_LOGS


def _check_sleep_wait(command: str, sleep_limit: int = 10, is_background: bool = False):
    """规则5：纯 sleep 干等。L3 时 sleep_limit 收窄到 _L3_SLEEP_LIMIT(3s)。
    允许: 短暂等待页面渲染(≤limit 且命令含其他实质操作)；
          background=true 的 sleep（合法长任务姿势, 由 process wait/poll 管理）。
    拦: 前台大秒数 sleep 干等。
    """
    if is_background:
        return None  # background 长任务是正解姿势，永不拦
    import re as _re
    m = _re.search(r"sleep (\d+)", command)
    if not m:
        return None
    secs = int(m.group(1))
    if secs <= sleep_limit and _re.search(r"(curl|js\(|goto|new_tab|fill|click|fetch)", command):
        return None  # 短等待+实质操作，放行
    if secs <= sleep_limit and "&&" in command:
        return None  # 组合命令里的短间隔，放行
    if secs <= sleep_limit:
        return None  # 短 sleep 本身无害（页面渲染等待等场景），放行
    return _BLOCK_SLEEP_LOOP


def _check_swallowed_stderr(command: str):
    """规则6：诊断类命令重定向 stderr 到 /dev/null → block。"""
    if "2>/dev/null" not in command and "2>/dev/null" not in command.replace(" ", ""):
        return None
    if not _is_diagnostic_command(command):
        return None
    return _BLOCK_SWALLOW_ERR


def _normalize(command: str) -> str:
    """归一化：压缩空白。Contract: 幂等，不改变语义令牌。"""
    return " ".join(command.split())


def _extract_service_names(command: str):
    """从命令尾部提取非选项令牌（服务名位置）。
    Contract: 只返回不以 - 开头的非动词令牌；--list/help/服务名子命令除外。
    """
    try:
        tokens = shlex.split(command)
    except ValueError:
        return []
    known_subcommands = {"cloud", "cloud-run", "cloud-svc-run", "local", "run",
                         "build", "up", "down", "restart", "logs", "status",
                         "deploy.sh", "build.sh", "bash"}
    names = []
    for tok in tokens:
        if tok.startswith("-") or tok in known_subcommands or tok.endswith((".sh", ".yml", ".yaml")):
            continue
        if "/" in tok or tok.startswith("$"):
            continue
        names.append(tok)
    return names


def _is_terminal_with_command(tool_name: str, args: dict) -> bool:
    return tool_name == "terminal" and bool(args.get("command"))


def _needs_name_verification(command: str) -> bool:
    if any(p in command for p in _READONLY_PASSES):
        return False
    return any(m in command for m in _NAME_CHECK_MARKERS)


def _on_pre_tool_call(**kwargs):
    tool_name = kwargs.get("tool_name", "")
    args = kwargs.get("args", {})
    if not _is_terminal_with_command(tool_name, args):
        return {}
    command = _normalize(args["command"])
    state = _state(kwargs.get("session_id", ""))
    sid = kwargs.get("session_id", "")

    # 规则1：与上次失败命令逐字相同
    if command == state.get("last_failed_command"):
        return _block_with_escalation("R1", command, _BLOCK_IDENTICAL, sid)

    # 规则2：同命令累计失败 >= 2 次
    fail_counts = state.setdefault("fail_counts", {})
    if fail_counts.get(command, 0) >= 2:
        return _block_with_escalation("R2", command, _BLOCK_TWICE, sid)

    # 规则3：服务名必须出现在本会话 --list 注册表输出里
    if _needs_name_verification(command):
        registry_output = state.get("registry_output", "")
        if not registry_output:
            names = _extract_service_names(command)
            if names:
                return _block_with_escalation(
                    "R3", command,
                    _BLOCK_UNVERIFIED_NAME.format(names=",".join(names[:5])), sid)
        else:
            # 服务名=注册表每行首列令牌（--list 输出为表格：服务 仓库 Dockerfile ...）
            # 首列提取防止仓库名(如 loom)冒充服务名(loom-backend)通过
            service_names = set()
            for line in state["registry_output"].splitlines():
                toks = line.split()
                if toks and toks[0] not in ("服务", "----", "Service"):
                    service_names.add(toks[0])
            for name in _extract_service_names(command):
                if name not in service_names:
                    return _block_with_escalation(
                        "R3", command,
                        _BLOCK_UNVERIFIED_NAME.format(names=name), sid)

    # 规则4：docker logs 裸奔
    msg = _check_raw_logs(command)
    if msg:
        return _block_with_escalation("R4", command, msg, sid)

    # 规则5：纯 sleep 干等轮询（background 长任务放行; L3 收窄 sleep 上限）
    msg = _check_sleep_wait(
        command,
        sleep_limit=_L3_SLEEP_LIMIT if _current_level("R5") == "L3" else 10,
        is_background=bool(args.get("background")))
    if msg:
        return _block_with_escalation("R5", command, msg, sid)

    # 规则6：诊断命令 2>/dev/null 吞错
    msg = _check_swallowed_stderr(command)
    if msg:
        return _block_with_escalation("R6", command, msg, sid)

    return {}


def _on_post_tool_call(**kwargs):
    tool_name = kwargs.get("tool_name", "")
    args = kwargs.get("args", {})
    if not isinstance(args, dict):
        return {}
    result = kwargs.get("result", {}) or {}
    if isinstance(result, str):
        # terminal 等工具的 result 以 JSON 字符串传入钩子；解析成 dict 再取字段
        try:
            result = json.loads(result)
        except (json.JSONDecodeError, TypeError) as e:
            logging.warning("no-guessing: result JSON 解析失败按空dict: %s", e)
            result = {}
    if not isinstance(result, dict):
        result = {}
    if not _is_terminal_with_command(tool_name, args):
        return {}
    command = _normalize(args["command"])
    exit_code = result.get("exit_code", 0)
    output = str(result.get("output", ""))
    state = _state(kwargs.get("session_id", ""))

    if "--list" in command and exit_code == 0 and len(output) > 50:
        # 注册表输出采集（保留最新一份）
        state["registry_output"] = output
        state.pop("last_failed_command", None)
        return {}

    if exit_code != 0:
        fail_counts = state.setdefault("fail_counts", {})
        fail_counts[command] = fail_counts.get(command, 0) + 1
        state["last_failed_command"] = command
        state["need_fail_reminder"] = True
    else:
        # 成功即清除该命令的失败计数与上次失败标记
        state.setdefault("fail_counts", {}).pop(command, None)
        if state.get("last_failed_command") == command:
            state.pop("last_failed_command", None)
    return {}


def _on_pre_llm_call(**kwargs):
    state = _state(kwargs.get("session_id", ""))
    if state.pop("need_fail_reminder", None):
        logger.info("no-guessing: 注入失败后纪律提醒")
        return {"context": _REMINDER_AFTER_FAIL}
    return {}


def register(ctx):
    """插件入口。"""
    ctx.register_hook("pre_tool_call", _on_pre_tool_call)
    ctx.register_hook("post_tool_call", _on_post_tool_call)
    ctx.register_hook("pre_llm_call", _on_pre_llm_call)
    logger.info("no-guessing 插件已注册——瞎猜机械闸门就绪")
