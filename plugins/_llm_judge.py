"""plugins/_llm_judge.py — 插件共用的 LLM 语义判定帮手。

仓库纪律：插件里的"人话语义判断"（质疑/完成声明/重大决策等不可穷举
语境）禁用关键词/正则匹配，一律走本模块的 LLM judge。
匹配对象是机器生成的确定性文本（异常类别字面量、路径）时才允许正则。

Contract:
  Preconditions: system/text 为非空 str；timeout>0；max_tokens>0
  Postconditions: 返回 True/False/None（None=fail-open，judge 失败/超时）；
                  绝不 raise 到调用方
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


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



def llm_judge_bool(task: str, system: str, text: str,
                   timeout: float = 20.0, max_tokens: int = 32,
                   true_key: str = "decision") -> Optional[bool]:
    """让辅助 LLM 按判定语义回答 {"<true_key>": true/false}。

    Preconditions:
      - system 非空且已指明"只回答 JSON"
      - text 已截断到安全长度（调用方负责，建议 <=4000 字符）
    Postconditions:
      - 返回 True/False/None；解析失败或异常 → None 并记日志
    """
    assert system and text, "system and text must be non-empty"
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
        needle_t = f'"{true_key}":true'
        needle_f = f'"{true_key}":false'
        if needle_t in content:
            return True
        if needle_f in content:
            return False
        return None
    except Exception as e:
        logger.warning("llm_judge(%s) failed: %s", task, e, exc_info=True)
        return None


def llm_judge_multi(task: str, system: str, text: str,
                    keys: List[str], timeout: float = 20.0,
                    max_tokens: int = 64) -> Dict[str, Optional[bool]]:
    """一次辅助 LLM 调用同时判定多个语义维度，返回 {key: True/False/None}。

    治串行浪费：多个插件同一时机、同一文本各自调 llm_judge_bool 时，
    合并为一次调用、一个 system prompt 列出各键判定标准、一次返回多键 JSON。
    单键缺失/解析失败 → 该键 None（fail-open），不拖垮其他键。

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

