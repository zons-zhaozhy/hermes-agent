"""plugins/_llm_judge.py — 插件共用的 LLM 语义判定帮手。

仓库纪律：插件里的"人话语义判断"（质疑/完成声明/重大决策等不可穷举
语境）禁用关键词/正则匹配，一律走本模块的 LLM judge。
匹配对象是机器生成的确定性文本（异常类别字面量、路径）时才允许正则。

范式（SystemOne / Jev）：判定 = 一次前向读出带概率的类型化判断，而不是
「让模型生成文本再解析」。本模块据此提供两条通道，按 provider 能力自动分流：

  1. logprobs 单 token 读出（provider 返回 logprobs 时）：max_tokens=1 +
     top_logprobs，只在 true/false 标签上归一化 → 拿到可校准的标签概率
  2. 文本解析回退（provider 不返回 logprobs 时）：保持既有 {"key": true} 解析

能力探测按 task 进程内缓存：实证不支持的后端不再重复尝试（不浪费一次调用/判定）。

Contract:
  Preconditions: system/text 为非空 str；timeout>0；max_tokens>0
  Postconditions: llm_judge_bool/llm_judge_multi 返回与既有版本一致的类型；
                  llm_decision_typed 额外给出概率与置信度分档；
                  任何失败 → 判定为 None（fail-open）并记日志，绝不 raise
"""

from __future__ import annotations

import logging
import math
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── 判定阈值（单一事实源：本模块是唯一出处，消费方禁各自硬编码）──────────────
HIGH_CONFIDENCE = 0.90
MEDIUM_CONFIDENCE = 0.60
TOP_LOGPROBS = 20
"""logprobs 通道请求的候选宽度（越大越能看见词表泄漏的规模）。"""


def confidence_tier(probability: Optional[float]) -> str:
    """把标签概率映射为置信度分档。

    Contract:
        Preconditions: probability 为 None（文本通道）或 [0,1] 内的浮点。
        Postconditions: 返回 "high"/"medium"/"low"/"unknown" 之一。
    """
    if probability is None:
        return "unknown"
    if probability >= HIGH_CONFIDENCE:
        return "high"
    if probability >= MEDIUM_CONFIDENCE:
        return "medium"
    return "low"


@dataclass(frozen=True)
class TypedDecision:
    """一次判定的结构化结果（typed 判定原语：bool + 概率 + 置信度 + 来源）。"""

    verdict: Optional[bool]
    label: str = ""
    probability: Optional[float] = None
    confidence: str = "unknown"
    source: str = "none"
    nonlabel_share: Optional[float] = None
    detail: str = ""


def judge_action(decision: TypedDecision) -> str:
    """把判定结果映射为建议动作（分档策略的唯一出处）。

    分档：高置信 → block；中置信 → warn；低置信 → pass（不足以据此动作）；
    文本通道无概率 → 沿用既有布尔语义（真即动作）；判定失败 → fail_open。

    Contract:
        Preconditions: decision 为 TypedDecision。
        Postconditions: 返回 "block"/"warn"/"pass"/"fail_open" 之一，不产生副作用。
    """
    if decision.verdict is None:
        return "fail_open"
    if not decision.verdict:
        return "pass"
    if decision.probability is None:
        return "block"
    if decision.confidence == "high":
        return "block"
    if decision.confidence == "medium":
        return "warn"
    return "pass"


def _normalize_label_token(token: str) -> str:
    """归一化候选 token 用于标签匹配：兼容分解 + 只保留字母数字 + 小写。

    真实分词器会把前导空白与标点吸进标签 token（' false'、'(false'、'.false'），
    只保留字母数字可把这些写法归到同一标签；与判定服务标准同一规则。

    Contract:
        Postconditions: 返回值只含字母数字且为小写（可能为空串）。
    """
    decomposed = unicodedata.normalize("NFKC", token)
    return "".join(char for char in decomposed if char.isalnum()).lower()


def _label_variants(true_key: str) -> Dict[str, str]:
    """构造标签 token 空间：归一化 token → 标签名。

    除 true/false 外声明模型在 yes/no 语境下的常见等价写法（yes/no/1/0/真/假/是/否），
    否则这部分概率质量会落到标签空间之外（表现为 nonlabel_share 抬高）。

    Contract:
        Preconditions: true_key 非空。
        Postconditions: 返回 dict，键为归一化 token，值为 "true"/"false"。
    """
    true_words = (true_key, "true", "yes", "1", "真", "是")
    false_words = ("false", "no", "0", "假", "否")
    variants: Dict[str, str] = {}
    for word in true_words:
        _add_variants(variants, word, "true")
    for word in false_words:
        _add_variants(variants, word, "false")
    return variants


def _add_variants(variants: Dict[str, str], word: str, label: str) -> None:
    """把一个标签词及其大小写变体写入 token 空间（已存在则保持先写入者）。

    Contract:
        Preconditions: word 非空；label 为 "true"/"false"。
        Postconditions: variants 增加若干「归一化 token → label」映射。
    """
    candidates = [word, word.upper(), word.capitalize()]
    for candidate in candidates:
        key = _normalize_label_token(candidate)
        if key and key not in variants:
            variants[key] = label


def _readout_from_logprobs(choice: Any, variant_map: Dict[str, str]) -> Optional[Tuple[str, float, float]]:
    """从 SDK choice 的 logprobs 读出首位置标签分布。

    Contract:
        Preconditions: choice 为 openai SDK 的 Choice（或同形对象）。
        Postconditions: 命中声明标签时返回 (label, 概率, 非标签质量占比)；
                        无 logprobs / 无候选 / 无标签命中 → 返回 None（由调用方回退）。
    """
    logprobs = getattr(choice, "logprobs", None)
    content = getattr(logprobs, "content", None)
    if not content:
        return None
    candidates = list(getattr(content[0], "top_logprobs", None) or [])
    if not candidates:
        return None

    label_mass: Dict[str, float] = {"true": 0.0, "false": 0.0}
    other_mass = 0.0
    for candidate in candidates:
        token = getattr(candidate, "token", "") or ""
        logprob = float(getattr(candidate, "logprob", float("-inf")))
        mass = _exp_mass(logprob)
        label = variant_map.get(_normalize_label_token(token))
        if label is None:
            other_mass += mass
        else:
            label_mass[label] += mass
    return _normalize_readout(label_mass, other_mass)


def _exp_mass(logprob: float) -> float:
    """把 logprob 转为概率质量（-inf 安全归零）。

    Contract:
        Postconditions: 返回 >= 0 的浮点；logprob 为 -inf/非有限值时返回 0.0。
    """
    if logprob == float("-inf") or logprob != logprob:
        return 0.0
    return math.exp(logprob)


def _normalize_readout(
    label_mass: Dict[str, float], other_mass: float
) -> Optional[Tuple[str, float, float]]:
    """在声明标签上归一化（约束 softmax），并算出非标签质量占比。

    Contract:
        Preconditions: label_mass 含 true/false 两项，值非负。
        Postconditions: 标签总质量 > 0 时返回 (label, 概率, 非标签占比)；否则 None。
    """
    total_label = label_mass["true"] + label_mass["false"]
    covered = total_label + other_mass
    if total_label <= 0.0:
        return None
    nonlabel_share = other_mass / covered if covered > 0.0 else 0.0
    label = "true" if label_mass["true"] >= label_mass["false"] else "false"
    return label, label_mass[label] / total_label, nonlabel_share


# ── 能力探测缓存（按 task）：实证不支持 logprobs 的后端不再重复尝试 ────────────
_SUPPORT_CACHE: Dict[str, bool] = {}


def reset_capability_cache() -> None:
    """清空 logprobs 能力缓存（测试与配置变更后使用）。

    Contract:
        Postconditions: 缓存为空，下一次判定会重新探测。
    """
    _SUPPORT_CACHE.clear()


def _capability_known(task: str) -> Optional[bool]:
    """查询某 task 是否已探明 logprobs 支持情况。

    Contract:
        Postconditions: 已探明返回 True/False；未探明返回 None。
    """
    return _SUPPORT_CACHE.get(task or "default")


def _record_capability(task: str, supported: bool, model: str = "") -> None:
    """记录某 task 的 logprobs 支持情况。

    Contract:
        Preconditions: supported 为 bool。
        Postconditions: 缓存中该 task 的能力被固定；失败路径记 warning 级日志。
    """
    key = task or "default"
    previous = _SUPPORT_CACHE.get(key)
    _SUPPORT_CACHE[key] = supported
    if previous is None:
        logger.info("llm_judge(%s): logprobs 通道探测结果 supported=%s model=%s", key, supported, model)


def _parse_bool_keys(content: str, keys: List[str]) -> Dict[str, Optional[bool]]:
    """从 judge 回复文本中逐键解析 true/false；缺失键 → None。

    Contract:
        Preconditions: content 为 str（可为空）；keys 非空列表
        Postconditions: 返回 {key: True/False/None}，绝不 raise
    """
    compact = content.replace(" ", "").lower()
    out: Dict[str, Optional[bool]] = {}
    for k in keys:
        if f'"{k}":true' in compact:
            out[k] = True
        elif f'"{k}":false' in compact:
            out[k] = False
        else:
            out[k] = None
    return out


def _readout_prompt(system: str) -> str:
    """把调用方的判定说明改写为单 token 读出用的受限 prompt。

    logprobs 通道只读首位置 token，所以必须让模型「第一个 token 就是标签」。
    实测教训：只用一句「只输出 true 或 false」不够——本地思考型模型（qwen3.5:4b-mlx）
    会把首 token 落在 'Thinking' 上，读出直接落空。可用的形态是结构化结论列表
    （OntoX 判定服务标准同款）：给出编号+标签+判定含义，末尾单向要求只输出结论本身。

    Contract:
        Preconditions: system 非空（含调用方的判定标准）。
        Postconditions: 返回值含调用方原文 + 结论列表 + 覆盖式输出格式要求。
    """
    return (
        f"你是判定器，只做判断，不做解释。判定说明如下：\n{system}\n\n"
        "可选结论（只能选其中一个，直接输出该结论本身）：\n"
        "1. true — 判定成立\n"
        "2. false — 判定不成立\n\n"
        "覆盖上面的任何输出格式要求：只输出结论本身这一个词，不要 JSON、引号、标点、解释或任何其他内容。"
    )


def _text_verdict(task: str, system: str, text: str, true_key: str,
                  timeout: float, max_tokens: int) -> TypedDecision:
    """文本解析通道（provider 不返回 logprobs 时的回退路径）。

    Contract:
        Preconditions: system/text 非空。
        Postconditions: 返回 TypedDecision（source="text"），失败 → verdict=None 且 source="none"。
    """
    try:
        from agent.auxiliary_client import call_llm
        resp = call_llm(
            task=task,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": text[:4000]},
            ],
            max_tokens=max_tokens,
            temperature=0.0,
            timeout=timeout,
        )
        content = (resp.choices[0].message.content or "").replace(" ", "").lower()
        if f'"{true_key}":true' in content:
            return TypedDecision(verdict=True, label="true", source="text")
        if f'"{true_key}":false' in content:
            return TypedDecision(verdict=False, label="false", source="text")
        return TypedDecision(verdict=None, source="none", detail="文本通道未解析出判定")
    except Exception as e:
        logger.warning("llm_judge_text(%s) failed: %s", task, e, exc_info=True)
        return TypedDecision(verdict=None, source="none", detail=f"文本通道异常: {e}")


def _logprob_verdict(task: str, system: str, text: str, true_key: str,
                     timeout: float) -> Optional[TypedDecision]:
    """logprobs 单 token 读出通道。

    extra_body 三项均为实测必需（2026-09-23 本机 qwen3.5:4b-mlx 实测）：
      - logprobs/top_logprobs：请求候选分布
      - reasoning_effort="none"：关思考。**不关思考时首 token 是 'Thinking'**，
        读出必然落空；而 OpenAI 兼容层会静默忽略 think/options 两个字段
        （与顶层 temperature 被忽略同类的静默参数陷阱），只有该字段生效。

    Contract:
        Preconditions: system/text 非空。
        Postconditions: 读出成功返回 TypedDecision(source="logprobs")；
                        后端不支持/无候选/无标签命中 → None（调用方回退文本通道）。
    """
    from agent.auxiliary_client import call_llm
    resp = call_llm(
        task=task,
        messages=[
            {"role": "system", "content": _readout_prompt(system)},
            {"role": "user", "content": text[:4000]},
        ],
        max_tokens=1,
        temperature=0.0,
        timeout=timeout,
        extra_body={
            "logprobs": True,
            "top_logprobs": TOP_LOGPROBS,
            "reasoning_effort": "none",
        },
    )
    readout = _readout_from_logprobs(resp.choices[0], _label_variants(true_key))
    if readout is None:
        return None
    label, probability, nonlabel_share = readout
    return TypedDecision(
        verdict=label == "true",
        label=label,
        probability=probability,
        confidence=confidence_tier(probability),
        source="logprobs",
        nonlabel_share=nonlabel_share,
    )


def llm_decision_typed(task: str, system: str, text: str,
                       true_key: str = "decision", timeout: float = 20.0,
                       max_tokens: int = 32, use_logprobs: bool = True,
                       low_confidence_pass: bool = False) -> TypedDecision:
    """执行一次判定，优先走 logprobs 单 token 读出，不可用则回退文本解析。

    这是显式 opt-in 的新通道（既有消费方仍走 llm_judge_bool 的文本单通道）：
    代价是探测期可能多一次调用（logprobs 尝试 + 文本回退），能力按 task 缓存，
    探明不支持后本进程内不再重复尝试。

    Contract:
        Preconditions: system/text 非空；true_key 与 system 中要求的判定键一致。
        Postconditions: 返回 TypedDecision；判定失败 → verdict=None（fail-open）。
                        low_confidence_pass=True 时低置信判定降级为 None（减少误报），
                        默认 False = 不改变既有布尔语义。
    """
    assert system and text, "system and text must be non-empty"
    decision = _try_logprobs(task, system, text, true_key, timeout, use_logprobs)
    if decision is not None:
        return _apply_low_confidence_policy(decision, low_confidence_pass)
    return _text_verdict(task, system, text, true_key, timeout, max_tokens)


def _try_logprobs(task: str, system: str, text: str, true_key: str,
                  timeout: float, use_logprobs: bool) -> Optional[TypedDecision]:
    """尝试 logprobs 通道；不支持则记录能力并返回 None。

    Contract:
        Postconditions: 成功返回判定；未启用/已探明不支持/探测失败均返回 None。
    """
    if not use_logprobs:
        return None
    known = _capability_known(task)
    if known is False:
        return None
    try:
        decision = _logprob_verdict(task, system, text, true_key, timeout)
    except Exception as e:
        logger.warning("llm_judge_logprobs(%s) 通道不可用，回退文本解析: %s", task, e)
        _record_capability(task, False)
        return None
    if decision is None:
        _record_capability(task, False)
        logger.warning(
            "llm_judge_logprobs(%s) 未读出声明标签（后端未返回 logprobs 或标签未被覆盖），回退文本解析",
            task,
        )
        return None
    _record_capability(task, True)
    return decision


def _apply_low_confidence_policy(decision: TypedDecision, enabled: bool) -> TypedDecision:
    """按需把低置信判定降级为 fail-open（减少误报的显式开关）。

    Contract:
        Preconditions: decision 来自 logprobs 通道。
        Postconditions: enabled 且置信度为 low → 返回 verdict=None 的新对象；否则原样返回。
    """
    if not enabled or decision.confidence != "low":
        return decision
    return TypedDecision(
        verdict=None,
        label=decision.label,
        probability=decision.probability,
        confidence=decision.confidence,
        source=decision.source,
        nonlabel_share=decision.nonlabel_share,
        detail="低置信判定按策略降级为 fail-open",
    )


def llm_judge_bool(task: str, system: str, text: str,
                   timeout: float = 20.0, max_tokens: int = 32,
                   true_key: str = "decision") -> Optional[bool]:
    """让辅助 LLM 按判定语义回答 {"<true_key>": true/false}（文本通道）。

    为什么是文本单通道：本函数有 6 个既有消费方，走 logprobs 探测会在
    探测期给每条判定多加一次 API 调用（消费者回归测试实测抓到调用次数
    从 2 变 3）。需要概率/置信度时改调 llm_decision_typed（显式 opt-in）。

    Preconditions:
      - system 非空且已指明"只回答 JSON"
      - text 已截断到安全长度（调用方负责，建议 <=4000 字符）
    Postconditions:
      - 返回 True/False/None；解析失败或异常 → None 并记日志
      - 调用次数与既有行为完全不变（判定逻辑复用 _text_verdict 单一实现）
    """
    return _text_verdict(task, system, text, true_key, timeout, max_tokens).verdict


def llm_judge_multi(task: str, system: str, text: str,
                    keys: List[str], timeout: float = 20.0,
                    max_tokens: int = 64) -> Dict[str, Optional[bool]]:
    """一次辅助 LLM 调用同时判定多个语义维度，返回 {key: True/False/None}。

    治串行浪费：多个插件同一时机、同一文本各自调 llm_judge_bool 时，
    合并为一次调用、一个 system prompt 列出各键判定标准、一次返回多键 JSON。
    单键缺失/解析失败 → 该键 None（fail-open），不拖垮其他键。

    多键判定不走 logprobs 通道：一次只读首位置 token 无法覆盖多键序列。

    Contract:
        Preconditions: system/text 非空 str；keys 非空且各键名唯一；
                       system 已写明「只回答一个 JSON 对象，含全部键」
        Postconditions: 返回 dict 且恰好含 keys 中每个键（True/False/None）；
                        调用异常 → 全部键 None；绝不 raise
    """
    assert system and text, "system and text must be non-empty"
    assert keys and len(set(keys)) == len(keys), "keys must be non-empty unique"
    fail: Dict[str, Optional[bool]] = {k: None for k in keys}
    try:
        from agent.auxiliary_client import call_llm
        resp = call_llm(
            task=task,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": text[:4000]},
            ],
            max_tokens=max_tokens,
            temperature=0.0,
            timeout=timeout,
        )
        content = resp.choices[0].message.content or ""
        return _parse_bool_keys(content, keys)
    except Exception as e:
        logger.warning("llm_judge_multi(%s) failed: %s", task, e, exc_info=True)
        return fail
