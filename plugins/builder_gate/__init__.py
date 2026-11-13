"""builder_gate — 构建者定位运行态执法（2026-11-13 用户令）。

双层同源：
  事前注入——prompt_sections 段 ``builder_gate``（session 级稳定段）；
  事后拦截——pre_verify 钩子（零写入回合 opt-in，声明式
  ``pre_verify_zero_edit = True``，机制同 finish_guard）。

执法面收窄到 finish_guard 的结构性缺口：零文件写入回合以提案式问句收尾
（「要不要我把这份复盘沉淀成 skill？」——分析者签名动作）。写入回合的
请示/宣言/推迟式收尾已由 finish_guard 覆盖，本插件对 changed_paths
非空的回合一律放行，不重复执法。

判定方式（2026-11-13 二次修正：禁关键词穷举，同 0915 语义分类裁定）：
  人话语式不可穷举（首版正则实测漏网「如需我…即可」变体），意图判定
  交辅助 LLM 按语义规则判——复用 plugins/_llm_judge.llm_judge_bool
  （yinyang-restate-guard 同款：缓存 + single-flight + fail-open）。
  机械事实检测（commit 哈希）不属语义枚举，保留为廉价快速通道。

Contract:
    Preconditions: pre_verify 载荷含 final_response / changed_paths / attempt
      （零写入回合经 pre_verify_zero_edit 声明同样被咨询）。
    Postconditions: 零写入 + 末段非完成证据 + judge 判定提案式收尾为真 +
      attempt < 1 时返回 {"action":"continue","message":...} 强制续跑；
      judge 失败/超时（None）fail-open 放行；其余一律 {}。单回合最多续跑 1 次。
"""
from __future__ import annotations

import logging
import re  # noqa: 检测器本质需要——机械事实检测（哈希/完成动词）无 str 等价
from typing import Any, Mapping

from plugins._llm_judge import llm_judge_bool

logger = logging.getLogger(__name__)

_MAX_NUDGES = 1  # 单回合最多强制续跑 1 次（防续跑死循环）
_TAIL_CHARS = 2000  # 送判末段（judge 建议输入 <=4000 字符，留裕量）
_MIN_RESPONSE_CHARS = 20  # 过短回复（寒暄级）不执法
_JUDGE_TIMEOUT = 20.0  # judge 超时即 fail-open，不拖慢回合收尾

# 完成证据豁免（机械事实检测，非语义枚举）：本回合虽零写入但带真凭实据的
# 报告（commit 哈希 + 完成动词），其尾部后续方向询问属正常汇报，直接放行
# 不送判，省 judge 预算。
# re-ok: 哈希为 7-40 位十六进制变长匹配，str 方法无法表达
_COMMIT_HASH_RE = re.compile(r"\b[0-9a-f]{7,40}\b")
_DONE_VERB_RE = re.compile(r"(已完成|已修复|已验证|已提交|已推送|已发布)")  # re-ok: 同上

_JUDGE_SYSTEM = (
    "你是回复收尾形态判定器。判定给定回复末段是否为「提案式收尾」：把本可由"
    "助手自行执行、可控可回滚的建设动作，以问句或邀请句形式退回用户（如询问"
    "是否需要自己做某事、暗示用户开口自己才继续，含陈述式变体「如需我…即可」）。"
    "以下不算提案式收尾：①真须用户拍板的事项（花钱/对外发布/不可逆操作/凭据/"
    "方向抉择）以确认句式呈报；②正常陈述结论、剩余边界；③带完成证据的报告尾部"
    "询问后续方向。只回答 JSON：{\"decision\": true/false}，true=是提案式收尾。"
)


def _has_completion_evidence(text: str) -> bool:
    """Contract: Postconditions: 文本同时含 commit 哈希与完成动词时 True。"""
    return bool(_COMMIT_HASH_RE.search(text)) and bool(_DONE_VERB_RE.search(text))


def _is_analyst_signature(final_response: str) -> bool:
    """语义判定末段是否提案式收尾（judge）；fail-open（None/异常）按非提案放行。

    Contract: Preconditions: final_response 为 str（可空）；Postconditions:
    过短或带完成证据 → False（不送判）；否则 judge 判定，None → False。
    """
    if not final_response or len(final_response) < _MIN_RESPONSE_CHARS:
        return False
    if _has_completion_evidence(final_response):
        return False
    tail = final_response[-_TAIL_CHARS:]
    try:
        verdict = llm_judge_bool(
            task="判定助手回复末段是否提案式收尾",
            system=_JUDGE_SYSTEM,
            text=tail,
            timeout=_JUDGE_TIMEOUT,
        )
    except Exception:  # noqa: judge 任何异常 fail-open，不阻塞回合（logger 必达）
        logger.warning("builder_gate: judge 异常，fail-open 放行", exc_info=True)
        return False
    if verdict is None:
        logger.warning("builder_gate: judge 无有效判定（超时/解析失败），fail-open 放行")
        return False
    return verdict


def _on_pre_verify(**kwargs: Any) -> dict:
    """pre_verify 钩子：零写入回合的提案式收尾强制续跑一次。

    Contract:
        Preconditions: kwargs 含 attempt / final_response / changed_paths。
        Postconditions: 命中执法面（零写入 + judge 判提案式 + attempt <
        _MAX_NUDGES）返回 continue 续跑；写入回合/豁免/judge fail-open/超限
        一律返回 {} 放行。
    """
    attempt = int(kwargs.get("attempt") or 0)
    if attempt >= _MAX_NUDGES:
        return {}
    changed_paths = list(kwargs.get("changed_paths") or [])
    if changed_paths:
        return {}  # 写入回合的收尾形态归 finish_guard
    final_response = str(kwargs.get("final_response") or "")
    if not _is_analyst_signature(final_response):
        return {}
    logger.warning(
        "builder_gate: 零写入回合以提案式收尾停轮（attempt=%d），强制续跑", attempt
    )
    return {
        "action": "continue",
        "message": (
            "[builder_gate] 构建者定位：收尾出现把建设动作退回用户的提案式句"
            "（要不要我/是否需要我/如需我…即可等语义形态），而本回合没有任何"
            "文件写入——提案不是交付。二选一后重发：①动作属自我决策域（可控可"
            "回滚）→ 当轮直接执行，办完在回复里报告结果与验证；②真须拍板（花"
            "钱/对外/不可逆/凭据）→ 用「必须先确认：」句式列出事项、默认动作和"
            "不回复时的默认行为。剩余边界用陈述句写明，不用问句。"
        ),
    }


# 声明式 opt-in：零写入回合也咨询本回调（核心默认契约=零写入不咨询 pre_verify；
# 机制见 hermes_cli.plugins.pre_verify_zero_edit_requested）
_on_pre_verify.pre_verify_zero_edit = True  # type: ignore[attr-defined] # pyright: ignore[reportFunctionMemberAccess]


def _builder_section(session_info: Mapping[str, Any]) -> str:
    """渲染构建者纪律稳定段（读段库取最新）；段缺失返回空串（空段被 core 跳过）。

    Contract:
        Preconditions: session_info 为 core 冻结的只读映射；段库存在键
        builder_gate。
        Postconditions: 返回段文本；段库不可读返回 ""（load_section 内部告警一次）。
    """
    from plugins.prompt_sections import load_section  # 延迟导入: 仅段渲染路径需要

    return load_section("builder_gate")


def register(ctx: Any) -> None:
    """插件入口：事前注入(system prompt 段) + 事后拦截(pre_verify) 双层同源执法。"""
    ctx.register_hook("pre_verify", _on_pre_verify)
    ctx.register_system_prompt_section("builder_gate", _builder_section)
    logger.info("builder_gate 已注册——构建者纪律稳定段 + 零写入回合提案式收尾语义拦截")
