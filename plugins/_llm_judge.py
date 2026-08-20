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
from typing import Optional

logger = logging.getLogger(__name__)


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
