"""ux-design-guard — UX 设计纪律机检插件（八层方法论执行层）。

双层同源（对齐 scientific-programming-guard 范式：事前注入+事后拦截）：

  L1 稳定段（register_system_prompt_section）
     动工前四判据+五问走查+六原则压缩段（≤400 字）注入每个新会话——
     不依赖 skill 是否被加载。理论全文唯一事实源
     =~/knowledge-base/ux-design/UX_INTERACTION_METHODOLOGY.md，段内不复制正文。

  L2 写前拦截（pre_tool_call）
     UI 源文件（.vue/.tsx/.jsx/.css/.scss/.less/.html）写入时机检两类理论违规：
     U1 装饰性无限动效——动效唯一合法用途是澄清状态（Cooper「用过渡解释变化」：
        过渡用于解释刚才发生了什么；常驻循环表演不承载状态信息）；
        合法白名单=loading/spinner/skeleton/shimmer/typing/pulse/progress/blink/
        error/warning/danger/alive/live/indicator/status 等状态语义；
     U2 队列入场动画——animationDelay 绑定索引乘法（逐项渐入延迟徒增感知延迟，
        排队是表演不是信息）。
     U3 动效时长超档——transition/animation 时长 >400ms 且无状态语义即拦
        （M3 现行动效=弹簧物理体系；CSS 固定时长按归档档位：小位移 150-200ms/
        大位移 300-400ms；超档感知为迟滞且挤占 Doherty 400ms 反馈预算。
        进度/骨架等状态动画走白名单）。

  设计约束：fail-open（插件异常不阻断会话）；dist/build/min 生成产物跳过；
  行内 ``ux-ok: <≥8字理由>`` 豁免；纯 str 方法逐行检测（零正则）。
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

logger = logging.getLogger(__name__)

_NAMESPACE = "ux_design_guard"

_EXCLUDED_PLATFORMS = {"subagent", "batch"}

_UI_SUFFIXES = (".vue", ".tsx", ".jsx", ".css", ".scss", ".less", ".html")

# 生成产物不检（库代码/构建输出，非手写设计面）
_ARTIFACT_MARKS = ("/dist/", "/build/", "/.output/", "/node_modules/", ".min.css", ".min.js")

# 状态语义白名单：命中即视为「澄清状态」的合法循环动画
_STATE_TOKENS = (
    "loading", "spinner", "skeleton", "shimmer", "typing",
    "pulse", "progress", "blink", "error", "warning", "danger",
    "alive", "live", "indicator", "status", "heartbeat", "scan",
)

_EXEMPT_MARK = "ux-ok:"

_UX_RULE = """\
[UX 设计纪律]
写界面文件(.vue/.tsx/.jsx/.css/.html)前对表八层方法论:①Kano投入顺序恒为
必备>期望>魅力——数据正确/操作不丢/可撤销/报错人话未清零,禁投入动效炫技
②五问走查:看(一眼懂)/动(几步/有反馈/可撤销)/错(能自恢复)/记(要记什么)/惯(一致吗)
③Cooper六原则:就地直接/留在页面/保持轻量/提供邀请/用过渡解释变化(动效只许澄清状态)/即时响应
④硬判据:反馈≤400ms;每屏主操作唯一;结果页/异常页投入最高;空态即引导;撤销优于确认;动效150-400ms超档即迟滞;操作按钮字号≥13px;正文行高比值∈[1.4,1.9]
理论=~/knowledge-base/ux-design/UX_INTERACTION_METHODOLOGY.md;判据=skill:ux-methodology"""


def _is_ui_target(path: str) -> bool:
    """路径是否属于机检范围的 UI 源文件。

    Contract:
      Postconditions: 后缀命中 UI 集合且不含生成产物标记 → True；其余 False
    """
    if not path.endswith(_UI_SUFFIXES):
        return False
    norm = path.replace("\\", "/").lower()
    return not any(mark in norm for mark in _ARTIFACT_MARKS)


def _has_exempt(line: str) -> bool:
    """行内豁免标记——ux-ok: 之后须留 ≥8 字符实质理由。

    Contract:
      Postconditions: 含合法豁免标记 → True；其余 False
    """
    idx = line.find(_EXEMPT_MARK)
    return idx >= 0 and len(line[idx + len(_EXEMPT_MARK):].strip()) >= 8


def _line_state_semantic(lines: list, idx: int) -> bool:
    """命中行及其前 4 行（选择器上下文窗口）是否含状态语义。

    Contract:
      Preconditions: 0 <= idx < len(lines)
      Postconditions: 窗口内任一行含状态令牌 → True
    """
    window = lines[max(0, idx - 4): idx + 1]
    blob = "\n".join(window).lower()
    return any(tok in blob for tok in _STATE_TOKENS)


def _check_decorative_infinite(lines: list) -> list:
    """U1 装饰性无限动效检测——无限循环且无状态语义即拦截。

    Contract:
      Postconditions: 返回违规消息列表（空=合规）
    """
    issues: list = []
    for lineno, line in enumerate(lines, start=1):
        low = line.lower()
        if "infinite" not in low or "animation" not in low:
            continue
        if _has_exempt(line):
            continue
        if _line_state_semantic(lines, lineno - 1):
            continue
        issues.append(
            f"U1 装饰性无限动效: line {lineno} '{line.strip()[:60]}' —"
            "动效唯一合法用途是澄清状态(Cooper/用过渡解释变化),常驻表演动画"
            "不承载状态信息。若确为状态指示(loading/skeleton/typing 等)请把状态"
            "语义写进类名或声明行;确有叙事必要则行内注释 ux-ok: <理由> 豁免。"
        )
    return issues


def _check_queue_in(lines: list) -> list:
    """U2 队列入场动画检测——animationDelay 随索引递增即拦截。

    Contract:
      Postconditions: 返回违规消息列表（空=合规）
    """
    issues: list = []
    for lineno, line in enumerate(lines, start=1):
        low = line.lower()
        if "animationdelay" not in low and "animation-delay" not in low:
            continue
        if _has_exempt(line):
            continue
        if any(sig in line for sig in ("*0.", "* 0.", "*$", "* $", "* i", "*i")):
            issues.append(
                f"U2 队列入场动画: line {lineno} '{line.strip()[:60]}' —"
                "animationDelay 随索引递增=排队弹入,是表演不是信息(逐项渐入"
                "徒增感知延迟)。整组同刻入场或按状态分批;确有叙事必要则行内"
                "注释 ux-ok: <理由> 豁免。"
            )
    return issues


_U3_MAX_MS = 400.0

# U4 操作按钮字号下限——多模型交叉审查实锤(12px 操作按钮三家一致判缺陷)
_U4_MIN_BTN_PX = 13.0
# token 分两类: 后缀类('-go'/'-act'/'-ops' 允许字母前缀相连,匹配 t-go/ops-btn 型);
# 完整词类(prev+next 非字母)。点前缀 '.act'/'.ops' 已删——'.active'/'.activity-feed'/'.ops-log'
# 字母/连字符相连即误拦(反方审查实测),其后缀场景由 '-act'/'-ops'/'-btn'(btn 完整词)覆盖。
_BTN_SELECTOR_TOKENS = ("button", "btn", ".go", ".cta", ".pager", "-go", "-act", "-ops")

# U5 正文行高窗口——中文排印正文(13-16px)行高比值须落 [1.4, 1.9]
_U5_BODY_FS_MIN = 13.0
_U5_BODY_FS_MAX = 16.0
_U5_LH_MIN = 1.4
_U5_LH_MAX = 1.9


def _font_size_px(line: str) -> float | None:
    """提取行内 font-size 的 px 字面量；非字面量（var/clamp/em/inherit）返回 None。

    Contract:
      Postconditions: 可解析的 ``font-size: <n>px``（含 !important 后缀）→ float；
        font 缩写 ``font: 500 12px/1.4`` 的字号段 → float；其余 → None（fail-open）
    """
    low = line.lower()
    idx = low.find("font-size")
    if idx < 0:
        idx = low.find("font:")
        if idx < 0:
            return None
    rest = line[idx:].split(":", 1)[-1]
    rest = rest.split(";")[0].split("}")[0].strip()
    # !important 是压过框架默认值的对抗写法,恰是高优先级拦截目标
    rest = rest.replace("!important", "").strip()
    # var/clamp/calc 是运行时合成值,静态不可判 fail-open
    if "var(" in rest or "clamp(" in rest or "calc(" in rest:
        return None
    px = rest.find("px")
    if px < 0:
        return None
    # font 缩写(font: 500 12px/1.4)取首个 px 前的数字段(=字号);
    # line-height 在 '/' 之后不干扰; weight 前置段无 px 不命中
    i = px - 1
    while i >= 0 and (rest[i].isdigit() or rest[i] == "."):
        i -= 1
    try:
        return float(rest[i + 1: px])
    except ValueError:
        return None


def _btn_token_boundary_hit(blob: str, token: str) -> bool:
    """token 在 blob 中以词边界命中。

    Contract:
      Postconditions: 后缀类 token('-go'/'-act'/'-ops')允许字母前缀相连
        (匹配 t-go/ops-btn 型),仅要求 next 非字母;其余 token 要求
        prev+next 均非字母——'.active' 对 'act' 族不命中('ive'字母相连)
    """
    start = 0
    suffix = token.startswith("-")
    while True:
        i = blob.find(token, start)
        if i < 0:
            return False
        after = i + len(token)
        next_ch = blob[after] if after < len(blob) else "\n"
        if not next_ch.isalpha():
            if suffix:
                return True
            prev_ch = blob[i - 1] if i > 0 else "\n"
            if not prev_ch.isalpha():
                return True
        start = i + 1


def _btn_selector_context(lines: list, idx: int) -> bool:
    """命中行向上至最近 '{' 的选择器窗口内是否含按钮语义 token（词界命中）。

    Contract:
      Preconditions: 0 <= idx < len(lines)
      Postconditions: 窗口含任一按钮 token(词界) → True（纯函数无副作用）
    """
    start = idx
    while start > 0 and "{" not in lines[start] and start > idx - 40:
        start -= 1
    blob = "\n".join(lines[start: idx + 1]).lower()
    return any(_btn_token_boundary_hit(blob, tok) for tok in _BTN_SELECTOR_TOKENS)


def _style_block_ranges(lines: list) -> list[tuple[int, int]]:
    """定位 .vue 文件 <style ...>...</style> 行号区间(1-based, 闭区间)。

    Contract:
      Postconditions: 返回区间列表; 无 style 块 → []
    """
    ranges = []
    start = None
    for i, line in enumerate(lines, start=1):
        if start is None and "<style" in line:
            start = i
        elif start is not None and "</style>" in line:
            ranges.append((start, i))
            start = None
    if start is not None:
        ranges.append((start, len(lines)))
    return ranges


def _check_btn_font_size(lines: list, style_only: bool = False) -> list:
    """U4 操作按钮字号下限——按钮语义选择器下 font-size <13px 即拦。

    Contract:
      Postconditions: 返回违规消息列表（空=合规）；豁免行(ux-ok:)跳过；
        style_only=True 时仅检 <style> 块内行(.vue 模板内联 style 非本规则范畴,
        模板 el-button 标签非 CSS 选择器上下文, 不得作为按钮语义依据)
    """
    issues: list = []
    in_scope = _style_block_ranges(lines) if style_only else [(1, len(lines))]
    for lineno, line in enumerate(lines, start=1):
        if not any(a <= lineno <= b for a, b in in_scope):
            continue
        low = line.lower()
        if "font-size" not in low and "font:" not in low:
            continue
        if _has_exempt(line):
            continue
        px = _font_size_px(line)
        if px is None or px >= _U4_MIN_BTN_PX:
            continue
        if not _btn_selector_context(lines, lineno - 1):
            continue
        issues.append(
            f"U4 操作按钮字号过小: line {lineno} '{line.strip()[:60]}' —"
            "有文本的操作按钮 <13px 不可读(多模型交叉审查三家一致判缺陷;"
            "Element Plus 默认按钮 14px 达标,small 变体 12px 须全局覆写 13px)。"
            "非按钮文本不受此限;纯图标按钮或确有空间约束则行内注释 ux-ok: <理由> 豁免。"
        )
    return issues


def _lh_value(line: str) -> tuple[float, bool] | None:
    """提取行内 line-height 字面量。

    Contract:
      Postconditions: ``line-height: <n>px`` → (px 值, True)；
        ``line-height: <n>``(无单位, 即比值) → (比值, False)；非字面量 → None
    """
    low = line.lower()
    idx = low.find("line-height")
    if idx < 0:
        return None
    rest = line[idx:].split(":", 1)[-1]
    rest = rest.split(";")[0].split("}")[0].strip()
    is_px = rest.endswith("px")
    body = rest[:-2].strip() if is_px else rest
    try:
        val = float(body)
    except ValueError:
        return None
    return (val, is_px)


def _block_font_size_px(lines: list, idx: int) -> float | None:
    """命中行所在规则块(向上至最近 '{')内的 font-size px 字面量。

    Contract:
      Preconditions: 0 <= idx < len(lines)
      Postconditions: 块内有 font-size:<n>px → 最近一处 float；无 → None
    """
    start = idx
    while start > 0 and "{" not in lines[start]:
        start -= 1
    found: float | None = None
    # 单行规则(.x{font-size:13px;line-height:1.1})声明与命中行同处一行:
    # 命中行 '{' 之后的片段也纳入探测,font-size 在 line-height 之前即命中
    probes = list(lines[start: idx])
    hit = lines[idx]
    if "{" in hit:
        probes.append(hit.split("{", 1)[-1] + ";}")
    for probe in probes:
        if "font-size" in probe.lower():
            found = _font_size_px(probe)
    return found


def _heading_selector(lines: list, idx: int) -> bool:
    """规则块选择器是否为标题元素(h1-h6)——标题压缩行高是排印惯例,豁免。

    Contract:
      Preconditions: 0 <= idx < len(lines)
      Postconditions: 块首选择器含 h1..h6 → True
    """
    start = idx
    while start > 0 and "{" not in lines[start]:
        start -= 1
    sel = lines[start].lower() if start < idx else lines[idx].lower()
    return any(f"h{i}" in sel for i in range(1, 7))


def _check_body_line_height(lines: list) -> list:
    """U5 正文行高窗口——正文字号(13-16px)下显式行高比值落 [1.4, 1.9]。

    Contract:
      Postconditions: 返回违规消息列表(空=合规); 无 font-size 上下文/
        非字面量/标题选择器均 fail-open; 豁免行(ux-ok:)跳过
    """
    issues: list = []
    for lineno, line in enumerate(lines, start=1):
        if "line-height" not in line.lower():
            continue
        if _has_exempt(line):
            continue
        lh = _lh_value(line)
        if lh is None:
            continue
        idx = lineno - 1
        if _heading_selector(lines, idx):
            continue
        fs = _block_font_size_px(lines, idx)
        if fs is None or not (_U5_BODY_FS_MIN <= fs <= _U5_BODY_FS_MAX):
            continue
        val, is_px = lh
        ratio = val / fs if is_px else val
        if _U5_LH_MIN <= ratio <= _U5_LH_MAX:
            continue
        issues.append(
            f"U5 正文行高越界: line {lineno} '{line.strip()[:60]}' —"
            f"正文 {fs:g}px 行高比值 {ratio:.2f} 须落 [{_U5_LH_MIN:g}, {_U5_LH_MAX:g}]"
            "(中文排印挤压<1.4 行间透不过气, >1.9 松散断行阅读节奏;"
            "巡检层判据 aesthetics_audit bodyLineHeights 同源)。标题 h1-h6 豁免;"
            "表格单元格等确有约束则行内注释 ux-ok: <理由> 豁免。"
        )
    return issues


def _durations_ms(line: str) -> list:
    """提取行内 CSS 时间字面量（.3s / 0.3s / 300ms），返回毫秒值有序列表。

    Contract:
      Postconditions: 每个可解析时间字面量按出现顺序入列；非时间 token 忽略
    """
    vals: list = []
    cleaned = line.replace(",", " ").replace(";", " ").replace("(", " ").replace(")", " ")
    for tok in cleaned.split():
        low = tok.lower()
        num = ""
        if low.endswith("ms") and len(low) > 2:
            num = low[:-2]
        elif low.endswith("s") and len(low) > 1:
            num = low[:-1]
        if not num:
            continue
        try:
            vals.append(float(num) * (1.0 if low.endswith("ms") else 1000.0))
        except ValueError:
            continue
    return vals


def _check_motion_duration(lines: list) -> list:
    """U3 动效时长超档检测——CSS 固定时长 >400ms 且无状态语义即拦。

    现行标准=M3 Expressive 动效物理体系（弹簧取代固定时长+缓动，2025-05 起）；
    CSS transition/animation 表达不了弹簧，按归档档位执行：小位移 150-200ms/
    大位移 300-400ms。CSS shorthand（transition/animation）首个时间值是时长、
    其后才是延迟，故只取首个时间值判定；纯延迟属性行（*-delay）属 U2 域不检。

    Contract:
      Postconditions: 返回违规消息列表（空=合规）
    """
    issues: list = []
    for lineno, line in enumerate(lines, start=1):
        low = line.lower()
        if "transition" not in low and "animation" not in low:
            continue
        if _has_exempt(line):
            continue
        head = low.split(":", 1)[0]
        if "delay" in head and "duration" not in head:
            continue
        durations = _durations_ms(line)
        if not durations or durations[0] <= _U3_MAX_MS:
            continue
        if _line_state_semantic(lines, lineno - 1):
            continue
        issues.append(
            f"U3 动效时长超档: line {lineno} '{line.strip()[:60]}' —"
            "M3 现行体系为弹簧动效(固定时长+缓动已被取代);CSS 固定时长按归档档位:"
            "小位移 150-200ms/大位移 300-400ms,>400ms 感知为迟滞且挤占 400ms 反馈"
            "预算。状态语义写进类名或声明行(loading/progress/skeleton 等)走白名单;"
            "大位移转场或弹簧实现则行内注释 ux-ok: <理由> 豁免。"
        )
    return issues


def _ux_section(session_info: Mapping) -> str:
    """渲染稳定段（subagent/batch 无界面写场景，返回空被 core 跳过）。

    Contract:
      Postconditions: 主会话返回 _UX_RULE 原文；排除平台返回 ""
    """
    if str(session_info.get("platform") or "") in _EXCLUDED_PLATFORMS:
        return ""
    return _UX_RULE


def on_pre_tool_call(**kwargs: Any) -> dict:
    """pre_tool_call 入口——UI 文件写前机检 UX 纪律。

    Contract:
      Preconditions: kwargs 含 tool_name 与 args（write_file/patch 为
        path+content / path+old_string+new_string）
      Postconditions: 违规返回 {"action":"block","message":...}；
        合规/非目标工具/非 UI 文件/解析失败 → {} 放行（fail-open）
    """
    try:
        tool_name = kwargs.get("tool_name", "")
        if tool_name not in ("write_file", "patch"):
            return {}
        args = kwargs.get("args", {}) or {}
        path = str(args.get("path", ""))
        if not _is_ui_target(path):
            return {}
        text = args.get("content") or args.get("new_string") or ""
        # patch 单行改小(new_string 无选择器上下文)是最高频写入形态——回读目标文件
        # 做一次替换模拟,让检测器看到改动后的完整文件而非孤立片段
        if tool_name == "patch" and args.get("old_string") and args.get("new_string"):
            try:
                with open(path, encoding="utf-8") as fh:
                    text = fh.read().replace(str(args["old_string"]), str(args["new_string"]), 1)
            except OSError:
                text = str(args["new_string"])
        lines = text.splitlines()
        issues = (
            _check_decorative_infinite(lines)
            + _check_queue_in(lines)
            + _check_motion_duration(lines)
            + _check_btn_font_size(lines, style_only=path.endswith(".vue"))
            + _check_body_line_height(lines)
        )
        if issues:
            return {
                "action": "block",
                "message": (
                    "[ux-design-guard] UX 设计纪律违规:\n"
                    + "\n".join(issues)
                    + "\n理论依据: ~/knowledge-base/ux-design/UX_INTERACTION_METHODOLOGY.md; "
                    "规则全文: skill:ux-methodology"
                ),
            }
        return {}
    except Exception:
        logger.warning("ux-design-guard 检查失败(放行)", exc_info=True)
        return {}


def register(ctx: Any) -> None:
    """插件入口：事前注入(system prompt 段) + 事后拦截(pre_tool_call) 双层同源执法。

    Contract:
      Postconditions: 两挂点已注册；异常不外泄
    """
    ctx.register_hook("pre_tool_call", on_pre_tool_call)
    ctx.register_system_prompt_section(_NAMESPACE, _ux_section)
    logger.info("ux-design-guard registered (hook + prompt section)")
