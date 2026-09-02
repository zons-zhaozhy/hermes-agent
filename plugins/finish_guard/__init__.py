"""
收尾守卫插件（finish_guard v2）
==============================

用户 0830 拍板「不做完不停手」；0831 根因调查发现 v1 无效且双重死亡：
1) transform_llm_output 只能替换返回给用户的文本（契约见 hermes_cli/plugins.py
   VALID_HOOKS 注释），不存在「触发重做」通道；
2) v1 回调签名 (output, **kwargs) 与核心实际 payload 键 response_text 不符
   （agent/turn_finalizer.py:615），TypeError 被核心 try/except 吞——v1 从未生效。

v2 改用真实有效通道，三层：
  L1 pre_tool_call: 拦截 clarify 工具调用——问题命中「要不要继续/做吗」类请示
     且不属合法豁免（不可逆/方向性决策/凭据）时返回 {"action":"block"}，
     模型收到 block 消息被迫自行决断继续。核心对 pre_tool_call 超时也 fail-closed。
  L2 pre_verify: 本回合改过文件且最终回复尾部为请示式收尾时返回
     {"action":"continue"}——核心注入 synthetic nudge 强制续跑
     （conversation_loop.py pre_verify 段，get_pre_verify_continue_message 消费）。
  L3 transform_llm_output(response_text=...): 给用户可见回复打标记（前两层
     未覆盖场景的留痕），签名与核心 payload 键对齐。

合法豁免（不拦）：拍板/授权/凭据/付费/不可逆删除清空覆盖/生产环境/
方向性选型/需求歧义澄清（requirement_interview 的「必须先确认：」前缀在此豁免）。
"""

import json
import logging
import re

logger = logging.getLogger(__name__)

# 请示式句式（出现即命中）
_ASK_RE = re.compile(
    r"(要做吗|现在做吗|需要我继续吗|要不要我|是否继续|继续吗[？?]|要继续吗"
    r"|还是等|再说[？?]|现在建吗|做的话我|要不要|需不需要|是否需要"
    r"|请确认[一下]?|需要你[^。，\n]{0,20}做)"
)

# 合法请示豁免：不可逆操作 + 方向性决策 + 需求歧义澄清（不拦）
_LEGIT_ASK_RE = re.compile(
    r"(拍板|授权|密码|凭据|付费|购买|公开发布|上传对外"
    r"|删除|删掉|清空|覆盖|格式化|不可逆|生产库|生产环境"
    r"|选哪个|方案[AB12]?还是|还是方案|你决定|请你定"
    r"|歧义|必须先确认|需要你确认|待确认)"
)

# 宣言式假继续（0902 实测穿透）：陈述句宣布继续/下一批但随即停轮。
# 判定只看「尾部命中计划句式 + 回合结束」，不看本回合 tool 调用数——
# 做了部分工作后宣言继续再停轮，同样是失效（用户 0902 拍板修正）。
_DECLARE_RE = re.compile(
    r"(继续|接下来|下一批|开始.{0,8}批|下一步|随后就|稍后|待续|进行中)"
)
# 完成态豁免：任务真完成后的收尾陈述（「下一步如需…」类边界话术）不拦
_DONE_RE = re.compile(
    r"(全部完成|已全部完成|已完成|完成完毕|收官|清零|收工|告一段落|全部推送"
    r"|待拍板|需要你拍板|必须先确认|需要你确认|等你确认|如需|若需|除非另有|另行指示)"
)


def _is_false_continue(text: str) -> bool:
    """Contract: Preconditions: text 为 str（可空）；Postconditions: 命中
    宣言式继续句式、且不命中完成态/合法请示豁免时返回 True。

    完成态豁免规则：完成词与继续宣言并存时（批间汇报「X已清零，继续Y」），
    只有完成词出现在宣言**之后**（真终态收尾）才放行；宣言在前=还有
    未做之事=拦。"""
    if not text:
        return False
    if _LEGIT_ASK_RE.search(text):  # re-ok: 中文句式模糊匹配无str等价
        return False
    m = _DECLARE_RE.search(text)  # re-ok: 同上
    if not m:
        return False
    done = _DONE_RE.search(text)  # re-ok: 同上
    if done and done.start() > m.start():
        return False  # 完成词在宣言后=终态收尾，放行
    return True  # 无完成词或完成词在宣言前=批间汇报，拦


# pre_verify 单回合最多强制续跑次数（防请示/续跑死循环，量级对齐核心 max nudges）
_MAX_PRE_VERIFY_NUDGES = 3


def _is_pushback(text: str) -> bool:
    """Contract: Preconditions: text 为 str（可空）；Postconditions: 命中请示句式
    且不命中合法豁免时返回 True，否则 False。"""
    if not text:
        return False
    return bool(_ASK_RE.search(text)) and not _LEGIT_ASK_RE.search(text)


def _tail_text(text: str, chars: int = 200) -> str:
    """取回复末尾段。Contract: Postconditions: 返回末尾 chars 字符（不足全量）。"""
    return text[-chars:] if text and len(text) > chars else (text or "")


# ---------------------------------------------------------------- L1: clarify 拦截

def _on_pre_tool_call(**kwargs) -> dict:
    """Contract: Preconditions: kwargs 含 tool_name/args（clarify 为 questions 列表）；
    Postconditions: 命中推责式提问时返回 {"action":"block","message":...}，
    否则返回 {} 放行。pre_tool_call 超时由核心 fail-closed 兜底。"""
    tool_name = kwargs.get("tool_name", "")
    if tool_name != "clarify":
        return {}
    args = kwargs.get("args") or {}
    try:
        blob = json.dumps(args, ensure_ascii=False)
    except (TypeError, ValueError) as e:
        logger.warning("finish_guard: clarify args 序列化失败，降级 str: %s", e)
        blob = str(args)
    if not _is_pushback(blob):
        return {}
    logger.warning(
        "finish_guard: 拦截请示式 clarify 调用——要求自行决断继续（摘要: %s）",
        blob,
    )
    return {
        "action": "block",
        "message": (
            "[finish_guard] 该提问属于请示式收尾，已拦截。除不可逆操作"
            "（删除/覆盖/发布/生产库）、凭据、付费、方向性选型外，一律自行"
            "判断并直接执行剩余步骤；可逆决定标注「假设：xxx」后继续。"
            "重新组织你的下一步行动，不要再问。"
        ),
    }


# ---------------------------------------------------------------- L2: pre_verify

def _on_pre_verify(**kwargs) -> dict:
    """Contract: Preconditions: 本回合改过文件（核心侧 changed_paths 非空才调）；
    Postconditions: 最终回复尾部为请示式收尾且 attempt 未超上限时返回
    {"action":"continue","message":...} 强制续跑，否则返回 {} 放行。"""
    attempt = int(kwargs.get("attempt") or 0)
    if attempt >= _MAX_PRE_VERIFY_NUDGES:
        return {}
    final_response = kwargs.get("final_response") or ""
    tail = _tail_text(final_response)
    if not (_is_pushback(tail) or _is_false_continue(tail)):
        return {}
    logger.warning(
        "finish_guard: pre_verify 拦截请示式/宣言式收尾（attempt=%d），强制续跑", attempt
    )
    return {
        "action": "continue",
        "message": (
            "[finish_guard] 你的收尾是请示式提问或宣言式假继续（宣布「继续/下一批」"
            "却停轮）。撤回收尾：把可逆决策标注为假设后直接连做剩余步骤，跑完拿"
            "真实结果收尾。只有不可逆操作、凭据、付费、方向性选型才允许停下请示。"
        ),
    }


# ------------------------------------------------- L3: 用户可见标记（签名对齐核心）

def _transform_llm_output(response_text: str = "", **kwargs) -> str:
    """Contract: Preconditions: response_text 为本回合 LLM 输出（核心键名）；
    Postconditions: L1/L2 未兜住且尾部命中请示式时返回打标文本（用户可见留痕），
    否则原样返回。"""
    output = response_text or ""
    if not output.strip():
        return output
    tail = _tail_text(output)
    if not (_is_pushback(tail) or _is_false_continue(tail)):
        return output
    logger.warning("finish_guard: 请示式/宣言式收尾漏网（L1/L2 未覆盖），打用户可见标记")
    return (
        output
        + "\n\n[finish_guard] 上面以请示式提问或宣言式假继续收尾。按既定纪律："
        "可逆决策标注假设直接继续，不必等待确认；下轮回复我将继续执行。"
    )


def register(ctx):
    """插件入口：三层钩子注册。"""
    ctx.register_hook("pre_tool_call", _on_pre_tool_call)
    ctx.register_hook("pre_verify", _on_pre_verify)
    ctx.register_hook("transform_llm_output", _transform_llm_output)
    logger.info("finish_guard v2 已注册——clarify 拦截 + pre_verify 续跑 + 可见标记")
