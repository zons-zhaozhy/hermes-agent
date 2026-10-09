"""devil-advocate-audit plugin — 反方审查检查点（防"我想错"：单视角决策拦截）

pre_llm_call 钩子：用辅助 LLM 语义判断当轮消息是否构成"重大方案定稿/决策承诺"
（不可穷举，禁关键词匹配），且本会话尚未见 delegate_task（反方审查）时，
注入红牌：重大判断必须先经反方视角（delegate 唱反调子代理或用户明示豁免）才可定稿。

设计依据：落实笔记-七心法七功法.md 心法一（反者道之动）——
"插件防的是忘了，防不了想错；防想错=决策与审查分脑"。

判定方式：LLM judge（语义，不可穷举场景唯一可行），非关键词。
成本控制：
- 消息 hash 去重（同一消息只判一次）
- judge 调用每会话硬上限 _MAX_JUDGE_CALLS（防高负荷会话烧 token）
- delegate_task 调用后本会话静默（reviewed=True）——仅当该委派经语义
  判定确属"反方审查/唱反调/挑漏洞"且成功完成；只读查询/跑腿类委派不免检
- fail-open：judge 失败/超时记日志透传，不阻塞主流程

缓存安全：注入 context 不改 system prompt/历史/toolset。

ACTIVATION: config.yaml plugins.enabled 添加 "devil-advocate-audit"。
Set DEVIL_ADVOCATE_AUDIT_DISABLE=1 to turn off.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
from typing import Any, Optional

from plugins._llm_judge import llm_judge_bool
from plugins._shared_state import get_session_state
# 用户侧合并判定：challenge/decision 一次调用（唯一出入口收拢）。
# 延迟导入：避免插件间循环依赖；yinyang 未启用时 judge_user_side 不可用，
# 此时本插件回退独立判定（llm_judge_bool 原路径）。
from importlib import import_module as _import_module


_YINYANG_LOOKUP_FAILED = False


def _judge_user_side(message: str) -> Optional[dict[str, Optional[bool]]]:
    # 运行时插件模块名 = hermes_plugins.<slug>（连字符转下划线，
    # 见 hermes_cli/plugins.py _directory_module_name）；依次尝试两个命名空间。
    # 查询失败缓存到进程生命期：yinyang 未启用时 import 必然失败且不会中途启用，
    # 每条消息重试两次 import 只产噪声（0924 实测单日 92 条重复 WARNING）。
    global _YINYANG_LOOKUP_FAILED
    if _YINYANG_LOOKUP_FAILED:
        return None
    for modname in ("hermes_plugins.yinyang_restate_guard",
                    "plugins.yinyang_restate_guard"):
        try:
            mod = sys.modules.get(modname) or _import_module(modname)
            return mod.judge_user_side(message)
        except Exception as e:
            logger.warning("user-side merge via %s unavailable: %s", modname, e)
    _YINYANG_LOOKUP_FAILED = True
    return None

logger = logging.getLogger(__name__)

_NAMESPACE = "devil_advocate_audit"

_MAX_JUDGE_CALLS = 30

_JUDGE_SYSTEM = (
    "你是决策审查哨兵。判断下面这条会话消息是否构成'重大方案定稿或决策承诺'——"
    "即：即将拍板采用某技术方案/架构选型/上生产部署/模型替换/大规模重构等"
    "影响面大且难回退的决策。只是讨论、提问、调研、汇报进度不算。"
    "只回答 JSON：{\\\"decision\\\": true} 或 {\\\"decision\\\": false}"
)

_DELEGATE_JUDGE_SYSTEM = (
    "你是委派任务分类器。判断下面这个 delegate_task 委派的 goal 是否属于"
    "'反方审查/唱反调/挑漏洞/独立审查某方案'类任务——即目的是质疑、批判、"
    "找缺陷、对抗性验证既有方案。普通的只读查询、数据收集、并行跑腿、"
    "只回答 JSON：{\\\"review\\\": true} 或 {\\\"review\\\": false}"
)

_REMINDER = (
    "[DevilAdvocateAudit] 反方审查检查点：检测到重大决策/方案定稿信号，"
    "但本会话尚未见反方视角审查（delegate_task 委派唱反调子代理）。\n"
    "  铁律：决策与审查必须分脑——单视角定稿=未审计。\n"
    "  现在做：委派一个只找漏洞的反方子代理审这个方案，或请用户明示豁免。\n"
    "若本次已在审查或用户已豁免，忽略本条。"
)

# 动作拦截:红牌不只是提醒——armed 状态下对非审查类工具调用发 block,
# 强制先走 delegate_task 反方审查正门。用户豁免词解除武装。
_WAIVE_JUDGE_SYSTEM = (
    "用户是否明示豁免反方审查？判断这条消息里用户是否明确说了"
    "类似'豁免/不用审/跳过审查/我拍板/我批准/别审了'的意思。"
    "只是提问、讨论、下指令干活不算豁免。"
    "只回答 JSON：{\"waive\": true} 或 {\"waive\": false}"
)

_BLOCK_MSG_TEMPLATE = (
    "[DevilAdvocateAudit 强制] 本会话存在未经反方审查的重大决策。"
    "在委派反方审查子代理之前，非审查类动作一律冻结。\n"
    "  正门：delegate_task 委派一个只找漏洞的唱反调子代理审查该决策"
    "（goal 须含'反方审查/挑漏洞/批判'语义）。\n"
    "  豁免：请用户明示豁免（说'豁免反方审查'即可）。\n"
    "  当前被拦工具：{tool_name}"
)


def _plugin_disabled() -> bool:
    return os.environ.get("DEVIL_ADVOCATE_AUDIT_DISABLE", "").lower() in {
        "1", "true", "yes", "on",
    }


def _count(sid: str, key: str = "count") -> int:
    return int(get_session_state(sid, _NAMESPACE).get(key, 0))


def _seen_hash(sid: str) -> set:
    """取会话已判消息 hash 集合（惰性建）。

    Contract:
      Postconditions: 返回可变 set 且已写回 session state（后续 add 持久生效）
    """
    st = get_session_state(sid, _NAMESPACE)
    s = st.get("seen")
    if not isinstance(s, set):
        s = set()
        st["seen"] = s
    return s


def _is_major_decision(text: str) -> Optional[bool]:
    """LLM judge 语义判定；失败返回 None（fail-open，视为非决策）。

    Contract:
      Preconditions: text is non-empty str
      Postconditions: 返回 True/False/None；绝不 raise
    """
    assert text, "text must be non-empty"
    return llm_judge_bool(
        task="devil_advocate_audit",
        system=_JUDGE_SYSTEM,
        text=text,
    )


def _extract_delegate_goals(args: Any) -> str:
    """从 delegate 载荷提取 goals 拼接文本（只读，无副作用）。

    两种委派形态都要认：tasks[].goal 批量形态 + 顶层 goal= 单任务形态
    （框架 schema 两者都允许，见 DELEGATE_TASK_SCHEMA）。
    """
    if not isinstance(args, dict):
        return ""
    texts = []
    single = args.get("goal")
    if isinstance(single, str) and single.strip():
        texts.append(single)
    tasks = args.get("tasks")
    if isinstance(tasks, list):
        texts.extend(
            str(t.get("goal") or "")
            for t in tasks
            if isinstance(t, dict) and t.get("goal")
        )
    return "\n".join(t for t in texts if t)


def _delegate_is_review(goals_text: str) -> Optional[bool]:
    """LLM 语义判定委派 goal 是否属反方审查类。

    Contract:
      Preconditions: goals_text is non-empty str
      Postconditions: 返回 True/False/None（None=判定通道故障/超时）；绝不 raise
    """
    assert goals_text, "goals_text must be non-empty"
    return llm_judge_bool(
        task="devil_advocate_delegate",
        system=_DELEGATE_JUDGE_SYSTEM,
        text=goals_text,
        true_key="review",
    )


# 门禁自身对外公布的语义标记（见 _BLOCK_MSG_TEMPLATE：「goal 须含'反方审查/
# 挑漏洞/批判'语义」）。这是门禁规定给委派方的契约用语，属机器可判文本；
# 人话语义判定仍走 LLM judge（_delegate_is_review）——两道并行，标记命中即
# 判定成立，使解锁不依赖可能超时的判定通道（2026-10-09 双出口死锁实录）。
_REVIEW_MARKERS = (
    "反方审查", "唱反调", "挑漏洞", "只找漏洞", "批判",
    "Devil's Advocate", "devil's advocate",
)

# 门禁公布给用户的豁免口令（见 _BLOCK_MSG_TEMPLATE：「说'豁免反方审查'即可」）。
# 字面命中即豁免，同样不依赖判定通道。
_WAIVE_PHRASES = ("豁免反方审查", "豁免审查", "跳过反方审查", "免除反方审查")


def _goal_has_review_marker(goals_text: str) -> bool:
    """goal 是否携带门禁公布的审查语义标记（零 LLM 依赖）。

    Contract:
      Preconditions: goals_text 为 str（可为空）
      Postconditions: 命中任一公布标记 → True；空串或未命中 → False
    """
    if not goals_text:
        return False
    return any(marker in goals_text for marker in _REVIEW_MARKERS)


def _has_waive_phrase(text: str) -> bool:
    """用户消息是否含门禁公布的口令（去空白后字面匹配，零 LLM 依赖）。

    Contract:
      Preconditions: text 为 str（可为空）
      Postconditions: 命中任一公布口令 → True；空串或未命中 → False
    """
    compact = "".join(text.split())
    if not compact:
        return False
    return any(phrase in compact for phrase in _WAIVE_PHRASES)


def _delegate_declared_review(goals_text: str) -> bool:
    """委派是否属反方审查：公布标记优先（零延迟零依赖），否则 LLM judge。

    Contract:
      Preconditions: goals_text 为 str
      Postconditions: 返回 True/False；judge 返回 None（通道故障）记 warning
        并视为 False——但标记路径已先行覆盖门禁公布的委派形态，故通道
        故障不再导致锁死。
    """
    if _goal_has_review_marker(goals_text):
        return True
    verdict = _delegate_is_review(goals_text)
    if verdict is None:
        logger.warning(
            "devil-advocate-audit: 审查委派判定通道故障（fail-open），"
            "仅有公布标记的委派可解锁；本次 goal 未含公布标记"
        )
    return verdict is True


def _user_declared_waive(text: str) -> bool:
    """用户是否明示豁免：公布口令优先（零依赖），否则 LLM judge。

    Contract:
      Preconditions: text 为 str
      Postconditions: 返回 True/False；judge 通道故障（None）记 warning 并
        视为 False，但公布口令路径已先行覆盖门禁指引用户使用的措辞。
    """
    if _has_waive_phrase(text):
        return True
    verdict = _user_waived(text)
    if verdict is None:
        logger.warning(
            "devil-advocate-audit: 豁免判定通道故障（fail-open）；"
            "用户可用门禁公布口令「豁免反方审查」字面解锁"
        )
    return verdict is True


def on_post_tool_call(**kwargs) -> None:
    """delegate_task 委派确属反方审查且成功 → 本会话静默。

    Contract:
      Postconditions: 仅当 tool_name 属于 delegate 集合、status=ok（框架
      observer 词表的唯一成功态）、goal 命中门禁公布标记或经 LLM 判定为
      反方审查类时写 reviewed 标记；判定通道故障且未含标记时不写（但门禁
      已在 block 文案中明示标记用语，故仍有可达出口）。
    """
    if str(kwargs.get("tool_name", "")) not in {"delegate_task", "delegate"}:
        return
    # 框架词表（model_tools._tool_result_observer_fields）成功态唯一真名是
    # "ok"；写 "success" 等一个永不出现的词 = armed 永不解除的死锁。
    if str(kwargs.get("status") or "") != "ok":
        return
    sid = kwargs.get("session_id", "") or kwargs.get("task_id", "")
    if not sid:
        return
    goals = _extract_delegate_goals(kwargs.get("args"))
    if goals and _delegate_declared_review(goals):
        get_session_state(sid, _NAMESPACE)["reviewed"] = True
        logger.info("devil-advocate-audit: 反方审查委派已识别，本会话解除武装")


def on_pre_llm_call(**kwargs) -> Optional[dict[str, Any]]:
    """语义判定重大决策 + 未见反方审查 → 注入红牌。fail-open。"""
    try:
        if _plugin_disabled():
            return None
        sid = kwargs.get("session_id", "") or kwargs.get("task_id", "")
        if not sid:
            return None
        st = get_session_state(sid, _NAMESPACE)
        if st.get("reviewed"):
            return None
        text = str(kwargs.get("user_message", "") or "")
        if not text.strip():
            return None
        # 豁免出口必须先于一切判定/去重/cap：用户明示豁免若排在「消息须先被
        # 判为重大决策」之后，一条纯豁免指令本身不是决策陈述，判定 false 即
        # 提前 return → waived 永不写入，armed 冻结无解（2026-10-09 实录：
        # 用户连发两次门禁公布口令仍被拦）。公布口令零依赖先行；已 armed 的
        # 会话再补一次 LLM 判定，使非字面措辞也不被 hash 去重与决策门挡住。
        if _has_waive_phrase(text) or (st.get("armed") and _user_declared_waive(text)):
            st["waived"] = True
            return None
        # cron 等无人值守平台禁 armed：正门（delegate_task）可能不在工具集、
        # 豁免出口（用户明示）无人在场——armed 即无解死锁（2026-09-24 日学习
        # job 4162e5ea 全工具冻结整轮失败实录）。降级为仅注入提醒：有
        # delegate 工具的 cron 会话仍会被提醒引导自行反方审查。
        platform = str(kwargs.get("platform", "") or "")
        unattended = platform in {"cron", "subagent"}
        if _count(sid, "judge_calls") >= _MAX_JUDGE_CALLS:
            # cap 满：决策判定停摆，但 armed 会话仍须保留豁免出口——
            # 否则冻结无解（用户说豁免词也到不了判定）。仅 armed 态探测，
            # waived/reviewed 置位后本分支不再触发，成本有界。
            if st.get("armed") and _user_declared_waive(text):
                st["waived"] = True
            return None
        h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
        if h in _seen_hash(sid):
            return None
        _seen_hash(sid).add(h)
        st["judge_calls"] = _count(sid, "judge_calls") + 1
        # 合并判定：优先走 yinyang 的多键一次调用；两插件各自 hash 去重仍生效，
        # 但合并结果同时供 yinyang 消费（见 yinyang on_pre_llm_call 侧缓存）。
        merged = _judge_user_side(text)
        if merged is None:
            # 合并通道不可用（yinyang 未启用等）→ 回退原独立判定
            if _is_major_decision(text) is not True:
                return None
        elif merged.get("decision") is not True:
            return None
        st["count"] = _count(sid) + 1
        # 用户明示豁免 → 记 waived 并解除武装。豁免判定不设 cap——
        # cap 只限"决策判定"；豁免是用户主动出口，被 cap 挡=armed 死锁无解。
        if _user_declared_waive(text):
            st["waived"] = True
            return None
        if unattended:
            # 无人值守：不武装，只提醒（防止上面注释所述死锁）
            return {"context": _REMINDER}
        st["armed"] = True
        return {"context": _REMINDER}
    except Exception as e:
        logger.warning("devil-advocate-audit hook failed: %s", e,
                       exc_info=True)
        return None


def _user_waived(text: str) -> Optional[bool]:
    """LLM 语义判定用户是否明示豁免反方审查。

    Contract:
      Preconditions: text is non-empty str
      Postconditions: 返回 True/False/None（None=判定通道故障/超时）；绝不 raise
    """
    assert text, "text must be non-empty"
    return llm_judge_bool(
        task="devil_advocate_waive",
        system=_WAIVE_JUDGE_SYSTEM,
        text=text,
        true_key="waive",
    )


def on_pre_tool_call(**kwargs) -> Optional[dict[str, Any]]:
    """armed（重大决策未审）时对非 delegate 工具发 block 强制先过反方审查。

    Contract:
      Postconditions: 仅当 armed=True 且未 reviewed/waived 且工具不属于
      {delegate_task, delegate} 时返回 block 指令；其余一律返回 None
      (放行)。judge 不在此路径——armed 状态由 pre_llm_call 预先语义判定
      写入，此处纯状态读取，零 LLM 延迟。
    """
    try:
        if _plugin_disabled():
            return None
        sid = kwargs.get("session_id", "") or kwargs.get("task_id", "")
        if not sid:
            return None
        st = get_session_state(sid, _NAMESPACE)
        if not st.get("armed") or st.get("reviewed") or st.get("waived"):
            return None
        tool_name = str(kwargs.get("tool_name", ""))
        if tool_name in {"delegate_task", "delegate"}:
            # 派发时刻即解除武装：正门已走，不再依赖 post_tool_call 的回灌——
            # 后台委派完成于本回合之后，status/session 任一环节失配都会让
            # reviewed 永不写入 = armed 无解（2026-10-09 实录）。
            if _goal_has_review_marker(_extract_delegate_goals(kwargs.get("args"))):
                st["reviewed"] = True
            return None  # 正门：反方审查委派本身放行
        return {
            "action": "block",
            "message": _BLOCK_MSG_TEMPLATE.format(tool_name=tool_name),
        }
    except Exception as e:
        logger.warning("devil-advocate-audit pre_tool_call failed: %s", e,
                       exc_info=True)
        return None


def register(ctx) -> None:
    ctx.register_hook("pre_tool_call", on_pre_tool_call)
    ctx.register_hook("post_tool_call", on_post_tool_call)
    ctx.register_hook("pre_llm_call", on_pre_llm_call)
    logger.info("devil-advocate-audit registered")
