"""plugins/_judge_warm.py — 把同一时隙内互不依赖的判定提前点燃。

为什么需要：`pre_llm_call` 每轮只触发一次，而且是**唯一**能把文本注入本轮的缝
（`pre_tool_call` 只能拦或改参数，不能注入上下文）。所以这一轮要用的判定必须在本轮
第一 token 之前就绪——没有"稍后补送"这个选项。能拿回来的只有时间：同一轮里
`user_side_guards`（~2.5s）与 `read_think_intent`（~4s）判的是同一条用户消息、互不
依赖，串行执行等于白等其中一次。

做法：先跑的那个把另一个扔进后台线程，随后自己同步跑；后跑的那个走
`plugins._llm_judge` 的内容寻址缓存 / 并发单飞，直接拿到（或等）同一份结果，
不会重复付费。

契约：
  Postconditions: `warm()` 立刻返回；被点燃的判定在守护线程里执行，**异常全部吞掉**
                  并记 debug（预热失败只影响等待时长，绝不能影响本轮）。
                  同一 (callable, 参数) 在飞行中不重复点燃。
"""

from __future__ import annotations

import hashlib
import logging
import threading
from typing import Any, Callable

from agent.memory_provider import spawn_context_thread  # 话题化线程：必须带上调用方的 profile 作用域

logger = logging.getLogger(__name__)

_INFLIGHT: set[str] = set()
_LOCK = threading.Lock()


def _key(fn: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    digest = hashlib.sha1()
    digest.update(f"{getattr(fn, '__module__', '')}.{getattr(fn, '__name__', fn)!r}".encode())
    for arg in args:
        digest.update(b"\x00")
        digest.update(repr(arg).encode("utf-8", "replace"))
    for name in sorted(kwargs):
        digest.update(b"\x00")
        digest.update(f"{name}={kwargs[name]!r}".encode("utf-8", "replace"))
    return digest.hexdigest()


def warm(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> bool:
    """在后台守护线程里执行一次判定，供同一时隙内的其他消费者复用。

    Contract:
        Preconditions: fn 为判定调用（通常走 plugins._llm_judge，自带缓存与并发单飞）
        Postconditions: 立刻返回 True；同一调用在飞行中时返回 False 且不重复点燃；
                        fn 的任何异常都被吞掉，只记 debug
    """
    key = _key(fn, args, kwargs)
    with _LOCK:
        if key in _INFLIGHT:
            return False
        _INFLIGHT.add(key)

    def _runner() -> None:
        try:
            fn(*args, **kwargs)
        except Exception:
            logger.debug("judge warm failed (ignored)", exc_info=True)
        finally:
            with _LOCK:
                _INFLIGHT.discard(key)

    # 必须走 spawn_context_thread：裸 Thread 带不来调用方的 ContextVars（profile 作用域 +
    # 密钥作用域），预热会悄悄落到默认 profile 上判定——一个进程服务多 profile 时这是泄漏。
    spawn_context_thread(_runner, name="judge-warm").start()
    return True


def reset() -> None:
    """清空飞行表（测试用）。"""
    with _LOCK:
        _INFLIGHT.clear()
