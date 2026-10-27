"""guards.four_axis — pre_tool_call 四轴闸门副防线。

与 ReadThinkGate 协同：Gate 扫描 assistant_content 检测四轴证据，
写入 marker 文件。本插件读取 marker 文件，四轴未齐即阻断写工具。

双防线逻辑：
  - Gate（主）：每 turn 累积四轴证据 → 四轴齐备后写入 marker + 解锁
  - 插件（副）：每次 write_file/patch/execute_code 前读 marker → 不齐即阻断

marker 文件路径：~/.hermes/cache/four_axis_gate_{pid}_{session}.json（写读两端
经 gate 模块同一 slug 函数解析，见 _marker_file）
格式：{"verified": true, "timestamp": 1234567890.0, "axes": ["影响面","原意图","根因","风险"]}
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# marker 文件路径由 ReadThinkGate 经 get_hermes_home() 解析（profile-aware），
# 本插件必须用同一解析方式，否则 profile 隔离下写读两端路径不一致（PR #3575
# 同类 bug）。延迟解析：每次读取时现算，不冻结进程首次解析结果。
# 与主闸门 agent/read_think_gate.py::_FOUR_AXIS_REQUIRED_TOOLS 对齐——
# 主闸门明确裁定 execute_code 不强制四轴(可为纯只读分析,写路径已被
# write_file/patch 覆盖)。副防线曾额外拦 execute_code,与主防线裁定
# 冲突,实测产生 474 次误拦。以主防线为唯一裁定源。
_WRITE_TOOLS = frozenset({"write_file", "patch"})
_MARKER_MAX_AGE_SECONDS = 1800  # marker 有效期 30 分钟——600s 实测小于单任务写链时长
                              # （含验证跑测+多笔 patch），中途过期重拦 5.3k 次；仍保留
                              # 时效防跨任务沿用，且 gate 每 turn 重写刷新覆盖。


def _marker_file(session_id: str) -> Path:
    """解析 marker 文件路径（与 ReadThinkGate 写入端同一解析方式）。

    Args:
        session_id: 当前会话 ID——marker 按会话分桶。网关单进程服务多会话，
            写端（read_think_gate.gate）的 turn-start 只清本会话的桶，故读端
            必须以同一会话身份读，否则永远读不到写端刚写入的 marker。

    Returns:
        Path: hermes_home/cache/four_axis_gate_{pid}_{session}.json
    """
    from plugins.read_think_gate_host import _FOUR_AXIS_LABELS, _four_axis_marker_path

    # 路径解析与 slug 规则唯一实现在写端 gate 模块（经 read_think_gate_host
    # 唯一 import 面 re-export）——本插件直接复用，杜绝写读两端各写一份
    # 命名逻辑造成漂移。
    return _four_axis_marker_path(session_id)

# 2026-08-15 修复：agent 自维护产物（日报/指标/state/cron 输出/缓存/记忆）不是
# "核心代码编辑"，无调用方无编译影响，属闸门设计范围外（escape-valve-runbook
# 第79行设计意图是 core code edit）。此前无路径过滤导致三省 cron 写日报被拦，
# 与 ErrorDiscipline 级联形成互锁死锁（拦截计数>=2 -> terminal 封锁）。
# 前缀同样经 get_hermes_home() 延迟解析（profile-aware，与 marker 路径一致）。


def _agent_owned_prefixes() -> tuple[Path, ...]:
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    return (
        home / "retrospectives",
        home / "state",
        home / "cron" / "output",
        home / "cache",
        home / "memories",
        home / "skill_suggestions",  # 每日审计报告等 agent 自产文档（同 2026-08-15 设计意图）
        home / "skills",  # skill 复测记录/滚动表等 agent 自维护产物——非核心代码编辑，
        # 与 skill_suggestions 同类；skills 下含 agent 按 skill_view 纪律落盘的
        # run-XXX 复测文档，属四轴闸门设计范围外（escape-valve-runbook: core code edit）
    )


# hermes home 之外的 agent 自维护产物目录。aml-bid 是 AML 投标管线的巡检/
# 状态日志（PIPELINE-STATE.md、MONITOR-LOG.md），由夜间调度 cron 每 15 分钟
# 追加——与 retrospectives 同类（agent 自产文档，非核心代码编辑）。不豁免则
# 与 patch-first（改文件唯一合法通道=patch）互斥，巡检日志被迫绕行终端写入。
_EXTRA_AGENT_OWNED_PREFIXES: tuple[str, ...] = (
    "~/Documents/aml-bid",
)


def _extra_agent_owned_prefixes() -> tuple[Path, ...]:
    """解析 hermes home 之外的豁免前缀（expanduser，每次现算，不冻结）。"""
    return tuple(Path(s).expanduser() for s in _EXTRA_AGENT_OWNED_PREFIXES)


def _is_agent_owned_path(path_str: str) -> bool:
    """判断目标路径是否属于 agent 自维护产物（四轴豁免范围）。

    Contract:
      Preconditions: path_str 为字符串（可为空，空返回 False）
      Postconditions: 返回 True 当且仅当 resolve 后的路径等于某个
        agent-owned 前缀或位于其下；解析失败返回 False（fail-closed）
    """
    if not path_str:
        return False
    try:
        p = Path(path_str).expanduser().resolve()
    except (OSError, RuntimeError) as exc:
        logger.warning(
            "agent-owned path check failed (fail-closed, guard stays active): "
            "path=%r error=%s", path_str, exc, exc_info=True,
        )
        return False
    prefixes = _agent_owned_prefixes() + _extra_agent_owned_prefixes()
    result = any(p == prefix or prefix in p.parents for prefix in prefixes)
    if result and any(
        p == prefix or prefix in p.parents for prefix in _agent_owned_prefixes()
    ):
        from hermes_constants import get_hermes_home

        hermes_root = str(get_hermes_home())
        assert str(p).startswith(hermes_root), (
            f"exempted path escaped hermes home scope: {p}"
        )
    return result


def _read_marker(session_id: str) -> Optional[dict]:
    """读取本会话的 marker 文件，验证时效性。"""
    try:
        _marker = _marker_file(session_id)
        if not _marker.exists():
            return None
        data = json.loads(_marker.read_text(encoding="utf-8"))
        age = time.time() - data.get("timestamp", 0)
        if age > _MARKER_MAX_AGE_SECONDS:
            logger.debug("four-axis marker expired (age=%.0fs)", age)
            return None
        if data.get("verified") and len(data.get("axes", [])) == 4:
            return data
    except Exception:
        logger.warning("four-axis marker read failed (treated as no marker)", exc_info=True)
    return None


def _read_progress(session_id: str) -> Optional[dict]:
    """读取本会话 marker 的原始进度（不过滤 verified），供拦截消息诊断缺失轴。

    写端在每有新轴命中时落盘（verified=是否已齐 4 轴），本函数只做时效
    检查；放行判定仍走 _read_marker（verified=True + 4 轴）。
    """
    try:
        _marker = _marker_file(session_id)
        if not _marker.exists():
            return None
        data = json.loads(_marker.read_text(encoding="utf-8"))
        age = time.time() - data.get("timestamp", 0)
        if age > _MARKER_MAX_AGE_SECONDS:
            return None
        return data
    except Exception:
        logger.warning("four-axis progress read failed (treated as no progress)", exc_info=True)
        return None


def _block_message(tool_name: str, missing_axes: Optional[list[str]] = None) -> Optional[dict[str, Any]]:
    """生成阻断消息；missing_axes 非空时附缺失轴诊断，避免全列四项的盲目循环。"""
    hint = ""
    if missing_axes:
        hint = (
            f"\n\n诊断：本 turn 四轴证据已部分命中，还缺 {len(missing_axes)} 轴——"
            f"{'、'.join(missing_axes)}。在回复正文中补齐缺失轴的证据后重试，"
            "已命中的轴无需重复。"
        )
    return {
        "action": "block",
        "message": (
            f"[四轴闸门 · 副防线] 工具 '{tool_name}' 被阻断。\n\n"
            "四轴闸门未通过——在动手修改代码前，必须在回复中逐项输出：\n\n"
            "  1. 影响面清单：每个受影响调用方（绝对路径+函数名+行号）\n"
            "  2. 原意图溯源：git log -p 最初 commit + 不变量清单\n"
            "  3. 根因定位：症状位置 vs 根因位置（文件+行号）\n"
            "  4. 风险矩阵：最坏场景（触发条件+影响范围+可恢复性）\n\n"
            "四轴就是 commit message 的 body。不用另写。"
            f"{hint}"
        ),
    }


# ---------------------------------------------------------------------------
# 主防线在场探测（三态）——死锁防护。
#
# 实测事故（2026-10-26~27 hongtai-aml 网关）：网关进程启动时主防线插件
# read-think-gate 未注册（启用配置晚于进程启动），本副防线按 marker 缺失
# 拦截全部写操作约 19 小时，会话输出四轮四轴证据无效——marker 由主防线
# 写入，主防线缺席则拦截永远无法被满足。cron 豁免（on_pre_tool_call 内）
# 防的是主闸门按平台禁用，本探测防的是主防线插件整个缺席/被禁用：确认
# 缺席时放行（副防线是纵深防御第二层，主防线不在则四轴执法整体不在此
# 进程——与 cron 会话同等姿态），探测基础设施异常时保持拦截——
# unknown 态不得与确认缺席折叠：检测失效 ≠ 主防线缺席。
# 残留形态已收口（2026-10-27）：某表面向 pre_tool_batch 传空 session_id 时
# 主防线曾不扫描（_current_gate 返回 None）、marker 仍不写，探测报在场、
# 副防线照拦——现写端 pre_tool_batch 与本读端 marker 桶解析共用
# effective_session_id 回退（plugins/read_think_gate/__init__.py 单一实现），
# 空 session_id 两端落同一桶，形态消除（回归：tests/plugins/test_four_axis_empty_session.py）。
_GATE_PLUGIN_ID = "read-think-gate"
_gate_ctx: Any = None  # register() 捕获：主防线在场探测的唯一句柄（公共 API has_plugin）
_absence_warned: set[str] = set()  # 已打缺席告警的会话（每会话一次；恢复在场即重置）
_LAST_BLOCK: dict[str, tuple[float, str, str]] = {}  # 桶会话 -> (ts, tool, target) 最近一次拦截
_REPEAT_WINDOW_SECONDS = 60  # 同会话+工具+目标在此窗口内二次拦截 → 升级提示禁原样重试


def _probe_main_gate() -> str:
    """主防线在场探测。

    Returns:
        str: ``active``（read-think-gate 已加载启用且 config 段未显式禁用）/
          ``absent``（探测成功且确认主防线不会写 marker：未加载/被禁用/
          加载失败/config 段 enabled=false）/ ``unknown``（探测基础设施
          异常——ctx 未捕获或 has_plugin 抛错，须保持拦截）。

    Contract:
      Postconditions: 不抛异常出界；unknown 与 absent 永不混淆。
    """
    if _gate_ctx is None:
        logger.error("four-axis guard: main-gate probe has no ctx (guard not registered?) — unknown")
        return "unknown"
    try:
        present = bool(_gate_ctx.has_plugin(_GATE_PLUGIN_ID))
    except Exception:
        logger.error(
            "four-axis guard: main-gate probe failed — unknown, keep blocking", exc_info=True,
        )
        return "unknown"
    if not present:
        return "absent"
    # 插件在场但 config 段显式禁用：主防线 check_batch 恒 early-return、
    # marker 永不写——与缺席等价。读法镜像 read_think_gate/__init__.py
    # _config_section（异常按默认 enabled=True，在场侧容错）。
    try:
        from hermes_cli.config import load_config_readonly

        section = (load_config_readonly() or {}).get("read_think_gate")
        if isinstance(section, dict) and section.get("enabled") is False:
            return "absent"
    except Exception:
        logger.warning(
            "four-axis guard: read_think_gate config read failed — treated active", exc_info=True,
        )
    return "active"


def on_pre_tool_call(**kwargs) -> Optional[dict[str, Any]]:
    """pre_tool_call hook：写工具执行前检查四轴 marker。"""
    tool_name = kwargs.get("tool_name", "")
    if tool_name not in _WRITE_TOOLS:
        return None

    # cron 会话豁免：主闸门（agent_init.py ReadThinkGate 装配处）对 platform=="cron"
    # 显式 enabled=False——无人值守会话无法回答 gate 提示。主闸门 disabled 意味着
    # check_batch 恒 early-return，四轴 marker 永不写入；副防线若仍按 marker 缺失
    # 拦截，cron 会话的 write_file/patch 会结构性死锁（输出再多四轴证据也无效）。
    # 与主闸门同一设计意图：cron 会话 ID 恒为 cron_{job_id}_{timestamp} 前缀
    # （hermes_state_portability.list_cron_job_runs 的扫描契约）。
    session_id = str(kwargs.get("session_id") or "")
    if session_id.startswith("cron_"):
        logger.debug(
            "four-axis guard: skipping %s — cron session (main gate exempt) %s",
            tool_name, session_id,
        )
        return None

    args = kwargs.get("args") or {}
    target = str(args.get("path") or args.get("file_path") or "")
    if _is_agent_owned_path(target):
        logger.debug(
            "four-axis guard: skipping %s — agent-owned path %s", tool_name, target,
        )
        return None

    # marker 桶与主防线写端同源解析：空 session_id 时同样回退活跃会话
    # （effective_session_id 单一实现于 plugins/read_think_gate/__init__.py，
    # 写读两端共用；任一端单独回退即桶错配 → 永久拦截形态）。
    try:
        from plugins.read_think_gate import effective_session_id

        marker_sid = effective_session_id(session_id)
    except Exception:
        logger.warning(
            "four-axis guard: effective_session_id resolution failed — raw session bucket",
            exc_info=True,
        )
        marker_sid = session_id

    marker = _read_marker(marker_sid)
    if marker is None:
        # 会话 id 漂移回退：上下文压缩后 pre_tool_batch（写端）与 pre_tool_call
        # （读端）拿到的 session_id 可能不一致（写端登记新 id、读端仍收旧 id），
        # marker 落在活跃会话桶而读端按传入 id 找不到 → 回退活跃桶再读一次。
        # 语义不变：只认「写端刚验证过」的活跃桶，不做全桶扫描（防跨会话借用）。
        # 2026-10-27 实测形态：写端 17:09 写新桶 verified，读端读旧桶 16:17 过期，
        # 四轴证据三轮无效拦截。
        try:
            from plugins.read_think_gate import _ACTIVE_SESSION_ID

            active_sid = _ACTIVE_SESSION_ID or ""
            if active_sid and active_sid != marker_sid:
                active_marker = _read_marker(active_sid)
                if active_marker is not None:
                    marker = active_marker
                    marker_sid = active_sid
        except Exception:
            logger.warning(
                "four-axis guard: active-bucket fallback failed", exc_info=True,
            )
    if marker is None:
        verdict = _probe_main_gate()
        missing_axes = None
        progress = _read_progress(marker_sid)
        if progress:
            from plugins.read_think_gate_host import _FOUR_AXIS_LABELS  # 唯一 import 面，懒加载与 _marker_file 同风格
            hit = set(progress.get("axes") or [])
            missing_axes = [a for a in _FOUR_AXIS_LABELS if a not in hit]
        if verdict == "absent":
            # 确认缺席：放行（死锁防护）。每会话告警一次，恢复在场即重置。
            if marker_sid not in _absence_warned:
                _absence_warned.add(marker_sid)
                logger.warning(
                    "four-axis guard: main gate (read-think-gate) absent/disabled in this "
                    "process — secondary line fail-open for session %s (deadlock guard); "
                    "four-axis enforcement is NOT active here",
                    marker_sid,
                )
            else:
                logger.debug(
                    "four-axis guard: main gate absent — allowing %s (session %s already warned)",
                    tool_name, marker_sid,
                )
            return None
        if verdict == "unknown":
            logger.error(
                "four-axis guard: main-gate state unknown — keep blocking %s "
                "(fail-closed, session=%s)",
                tool_name, marker_sid,
            )
        else:
            _absence_warned.discard(marker_sid)
            # 日志必须带 session 与已见轴——2026-10-27 审计 77 次拦截无法归属，
            # 只能查 state.db 取证；此后一条 grep 即可归因。
            logger.info(
                "four-axis guard: blocking %s — marker missing or expired "
                "(session=%s, axes_seen=%s)",
                tool_name, marker_sid, sorted(set(progress.get("axes") or [])) if progress else None,
            )
        # 同会话+工具+目标短窗内重复拦截 = 模型在原样重试，重试不改变判定；
        # 升级提示直接指路「证据写进回复正文」，省一轮无效 API（2026-10-27
        # 实测 4 组重复拦截：16:17×2、16:22×2、00:41×2、01:32×2）。
        now = time.time()
        prev = _LAST_BLOCK.get(marker_sid)
        repeat = (
            prev is not None
            and now - prev[0] <= _REPEAT_WINDOW_SECONDS
            and prev[1] == tool_name
            and prev[2] == target
        )
        _LAST_BLOCK[marker_sid] = (now, tool_name, target)
        if len(_LAST_BLOCK) > 256:
            _LAST_BLOCK.clear()
        block = _block_message(tool_name, missing_axes)
        assert block is not None  # 期望: _block_message恒返回dict(实现无None返回路径)
        if repeat:
            block["message"] += (
                "\n\n注意：同一目标 60 秒内已被拦截两次——原样重试不会改变判定，"
                "四轴证据（影响面/原意图/根因/风险）必须写进回复正文后再重试本工具。"
            )
        return block

    logger.debug(
        "four-axis guard: allowing %s — marker valid (axes=%s)",
        tool_name, marker.get("axes"),
    )
    return None


def register(ctx: Any) -> None:
    global _gate_ctx
    _gate_ctx = ctx  # 主防线在场探测句柄：公共探针 has_plugin（force reload 复用同一 manager）
    ctx.register_hook("pre_tool_call", on_pre_tool_call)
    logger.info("guards.four_axis registered")
