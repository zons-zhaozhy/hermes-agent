"""Shared engineering-intent classification for plugin injection gates."""

from __future__ import annotations

import re

ENGINEERING_HINT_RE = re.compile(
    r"代码|文件|报错|错误|bug|测试|接口|函数|数据库|表结构|字段|配置|部署|构建|编译|"
    r"服务|页面|样式|日志|性能|分支|提交|仓库|git|api|http|python|typescript|"
    r"javascript|java|vue|react|pytest|ci|pr",
    re.IGNORECASE,
)
ENGINEERING_ACTION_RE = re.compile(
    r"继续|接着|重构|实现|开发|迁移|修复|编写|创建|删除|清理|部署|提交"
)


def has_engineering_intent(message: str, history_text: str = "") -> bool:
    """Return whether this turn starts or continues engineering work."""
    combined = f"{message}\n{history_text}" if history_text else message
    return bool(
        ENGINEERING_HINT_RE.search(combined)
        or ENGINEERING_ACTION_RE.search(message)
    )
