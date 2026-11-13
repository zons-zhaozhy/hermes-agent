"""builder_gate — 构建者定位运行态执法（2026-11-13 用户令）。

双层同源：
  事前注入——prompt_sections 段 ``builder_gate``（session 级稳定段）；
  事后拦截——pre_verify 钩子（零写入回合 opt-in，声明式
  ``pre_verify_zero_edit = True``，机制同 finish_guard）。

执法面收窄到 finish_guard 的结构性缺口：零文件写入回合以提案式问句收尾
（「要不要我把这份复盘沉淀成 skill？」——分析者签名动作）。写入回合的
请示/宣言/推迟式收尾已由 finish_guard 覆盖，本插件对 changed_paths
非空的回合一律放行，不重复执法。

Contract:
    Preconditions: pre_verify 载荷含 final_response / changed_paths / attempt
      （零写入回合经 pre_verify_zero_edit 声明同样被咨询）。
    Postconditions: 零写入 + 回复末段命中提案式签名 + 未命中合法拍板豁免 +
      无完成证据豁免 + attempt < 1 时返回 {"action":"continue","message":...}
      强制续跑；其余一律返回 {} 放行。单回合最多续跑 1 次。
"""
from __future__ import annotations

import logging
import re
from typing import Any, Mapping

logger = logging.getLogger(__name__)

_MAX_NUDGES = 1  # 单回合最多强制续跑 1 次（防续跑死循环）
_TAIL_CHARS = 400  # 只看回复末段——签名动作定义在收尾
_MIN_RESPONSE_CHARS = 20  # 过短回复（寒暄级）不执法；真实分析收尾通常远超此长度

# 提案式签名句式（末段命中即执法）
# re-ok: 多模式中文句式或集，str 方法（split/find/endswith）无法表达
_SIGNATURE_RE = re.compile(
    r"(要不要我|是否需要我|需不需要我|需要我[^。，!？?\n]{0,12}吗[？?]"
    r"|如果你想[^。\n]{0,20}(我)?可以|需要的话我(可以|就|来)"
    r"|我可以帮[^。，\n]{0,10}(做|建|写|整理|沉淀|生成|打包|提交))"
)

# 合法拍板豁免：真须用户决策/凭据/不可逆的请示不拦（对齐 finish_guard _LEGIT_ASK_RE）
# re-ok: 多模式豁免词或集，str 方法无法表达
_LEGIT_DECISION_RE = re.compile(
    r"(拍板|授权|密码|凭据|付费|购买|公开发布|上传对外|删除|清空|覆盖"
    r"|不可逆|生产库|生产环境|选哪个|方案[AB12]?还是|你决定|请你定"
    r"|歧义|必须先确认|需要你确认|待确认|请确认)"
)

# 完成证据豁免：本回合虽零写入但带真凭实据的报告（含 commit 哈希 + 完成动词），
# 其尾部的后续方向询问属正常汇报，不按提案式签名执法。
# re-ok: 哈希为 7-40 位十六进制变长匹配，str 方法无法表达
_COMMIT_HASH_RE = re.compile(r"\b[0-9a-f]{7,40}\b")
_DONE_VERB_RE = re.compile(r"(已完成|已修复|已验证|已提交|已推送|已发布)")  # re-ok: 完成动词或集


def _has_completion_evidence(text: str) -> bool:
    """Contract: Postconditions: 文本同时含 commit 哈希与完成动词时 True。"""
    return bool(_COMMIT_HASH_RE.search(text)) and bool(_DONE_VERB_RE.search(text))


def _is_analyst_signature(final_response: str) -> bool:
    """Contract: Preconditions: final_response 为 str（可空）；Postconditions:
    末段命中提案式签名且不命中合法豁免/完成证据豁免时 True，否则 False。"""
    if not final_response or len(final_response) < _MIN_RESPONSE_CHARS:
        return False
    if _has_completion_evidence(final_response):
        return False
    tail = final_response[-_TAIL_CHARS:]
    if _LEGIT_DECISION_RE.search(tail):
        return False
    return bool(_SIGNATURE_RE.search(tail))  # re-ok: 见 _SIGNATURE_RE 声明处理由


def _on_pre_verify(**kwargs: Any) -> dict:
    """pre_verify 钩子：零写入回合的提案式收尾强制续跑一次。

    Contract:
        Preconditions: kwargs 含 attempt / final_response / changed_paths。
        Postconditions: 命中执法面（零写入 + 提案式签名 + attempt < _MAX_NUDGES）
        返回 continue 续跑；写入回合/豁免/超限一律返回 {} 放行。
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
            "[builder_gate] 构建者定位：收尾出现「要不要我/是否需要我/如果想"
            "想…我可以」类提案式问句，而本回合没有任何文件写入——提案不是交付。"
            "二选一后重发：①动作属自我决策域（可控可回滚）→ 当轮直接执行，办完"
            "在回复里报告结果与验证；②真须拍板（花钱/对外/不可逆/凭据）→ 用"
            "「必须先确认：」句式列出事项、默认动作和不回复时的默认行为。"
            "剩余边界用陈述句写明，不用问句。"
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
    logger.info("builder_gate 已注册——构建者纪律稳定段 + 零写入回合提案式收尾拦截")
