"""negative-conclusion-guard 误拦截测试。

按 discipline-plugin-architecture 测试方法论：
1. 正常场景不被拦（probe 未达阈值 / 已走权威入口）
2. 违规场景被拦（probe 达阈值且无权威验证 → pre_llm_call 注入提醒）
3. 会话隔离（不同 session_id 状态不串）
4. hook 返回键必须是 "context"（框架读取键，turn_context.py:1184）
"""
import os
import sys
import importlib.util

os.environ.setdefault("HERMES_HOME", os.path.expanduser("~/.hermes"))

# 加载插件（目录名含连字符，用 spec_from_file_location）
PLUGIN_DIR = os.path.expanduser("~/.hermes/plugins/negative-conclusion-guard")
spec = importlib.util.spec_from_file_location(
    "negative_conclusion_guard",
    os.path.join(PLUGIN_DIR, "__init__.py"),
)
mod = importlib.util.module_from_spec(spec)
sys.modules["negative_conclusion_guard"] = mod
spec.loader.exec_module(mod)

# 清理 session 状态（防串测）
from plugins._shared_state import clear_session

SID = "test-session-001"
clear_session(SID)
clear_session("test-session-002")

passed = 0
failed = 0


def check(name: str, cond: bool, detail: str = ""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}  {detail}")


print("=== 1. 正常场景：probe 未达阈值 → 不注入 ===")
# 1 次 probe，阈值 2 → pre_llm_call 返回 None
mod.on_post_tool_call(
    session_id=SID, task_id="", tool_name="terminal",
    args={"command": "ls /tmp/nonexistent"}, result={}, status="",
)
r = mod.on_pre_llm_call(session_id=SID, task_id="", user_message="test")
check("1 次 probe 不注入", r is None, f"got {r}")

print("=== 2. 正常场景：已走权威入口 → 不注入 ===")
# 关键场景：先 hermes plugins list 再 probe，不应注入（权威已见）
mod.on_post_tool_call(
    session_id=SID, task_id="", tool_name="terminal",
    args={"command": "hermes plugins list | grep coding"}, result={}, status="",
)
mod.on_post_tool_call(
    session_id=SID, task_id="", tool_name="terminal",
    args={"command": "find ~/.hermes -name xyz"}, result={}, status="",
)
mod.on_post_tool_call(
    session_id=SID, task_id="", tool_name="terminal",
    args={"command": "find /opt -name abc"}, result={}, status="",
)
r = mod.on_pre_llm_call(session_id=SID, task_id="", user_message="test")
check("权威入口已见 → 不注入", r is None, f"got {r}")

print("=== 3. 违规场景：2 次 probe 且无权威 → 注入 ===")
clear_session("test-session-002")
SID2 = "test-session-002"
# 关键场景复现：只 ls user 插件目录 + find，不下任何权威命令
mod.on_post_tool_call(
    session_id=SID2, task_id="", tool_name="terminal",
    args={"command": "ls -la ~/.hermes/plugins/coding-standards-guard/ 2>/dev/null"},
    result={}, status="",
)
mod.on_post_tool_call(
    session_id=SID2, task_id="", tool_name="terminal",
    args={"command": "find /Users/stan/.hermes -type d -name coding-standards-guard 2>/dev/null"},
    result={}, status="",
)
r = mod.on_pre_llm_call(session_id=SID2, task_id="", user_message="test")
check("2 次 probe 无权威 → 注入 dict", isinstance(r, dict), f"got {type(r)}")
check("注入键是 context", isinstance(r, dict) and "context" in r, f"keys={list(r.keys()) if r else None}")
check("注入内容含权威入口指引", isinstance(r, dict) and "hermes plugins list" in r.get("context", ""), f"{r}")

print("=== 4. 违规场景：search_files 零命中累积 → 注入 ===")
clear_session("test-session-002")
mod.on_post_tool_call(
    session_id=SID2, task_id="", tool_name="search_files",
    args={"pattern": "xyz", "path": "/tmp"}, result={"total_count": 0}, status="",
)
mod.on_post_tool_call(
    session_id=SID2, task_id="", tool_name="search_files",
    args={"pattern": "abc", "path": "/tmp"}, result={"total_count": 0}, status="",
)
r = mod.on_pre_llm_call(session_id=SID2, task_id="", user_message="test")
check("search_files 2 次零命中 → 注入", isinstance(r, dict), f"got {type(r)}")

print("=== 5. 会话隔离：SID 已见权威，SID2 未见 → 状态独立 ===")
r1 = mod.on_pre_llm_call(session_id=SID, task_id="", user_message="test")
check("SID(权威已见)不注入", r1 is None, f"got {r1}")
r2 = mod.on_pre_llm_call(session_id=SID2, task_id="", user_message="test")
check("SID2(无权威)注入", isinstance(r2, dict), f"got {type(r2)}")

print("=== 6. 非 probe 命令（正常 ls 用法）不计数 ===")
clear_session("test-session-002")
# 普通 ls 列出目录（不带探测语义）——如 `ls -la` 单独跑在非验证场景
mod.on_post_tool_call(
    session_id=SID2, task_id="", tool_name="terminal",
    args={"command": "ls -la"}, result={}, status="",
)
r = mod.on_pre_llm_call(session_id=SID2, task_id="", user_message="test")
check("ls -la 不计数不注入", r is None, f"got {r}")

print(f"\n结果: {passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
