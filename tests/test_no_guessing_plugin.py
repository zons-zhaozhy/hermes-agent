"""no-guessing 插件单测——瞎猜根治闸门六规则 + 升级机制 + 回归。
运行: cd ~/code/ai/github/fork/hermes-agent && python3 -m pytest tests/test_no_guessing_plugin.py -q
"""
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import importlib

m = importlib.import_module("plugins.no-guessing")

REGISTRY = (
    "OntoX 全家族服务注册表\n\n服务 仓库 Dockerfile 版本 端口\n---- ----\n"
    "auth-backend ontox-auth Dockerfile 2.23.0 12342\n"
    "loom-backend loom Dockerfile 4.70.2 12350\n"
    "gateway deploy Dockerfile.gateway 1.0.0 80"
)


def _pre(cmd, sid):
    return m._on_pre_tool_call(tool_name="terminal", args={"command": cmd}, session_id=sid)


def _post(cmd, sid, exit_code=0, output="ok"):
    return m._on_post_tool_call(
        tool_name="terminal", args={"command": cmd},
        result={"exit_code": exit_code, "output": output}, session_id=sid)


def test_r1_identical_retry_blocked():
    _post("curl -s http://x", "s1", exit_code=1, output="timeout")
    r = _pre("curl -s  http://x", "s1")  # 归一化后逐字相同
    assert r.get("action") == "block"
    assert "重试" in r["message"]


def test_r2_third_attempt_blocked():
    for cmd in ("ping h", "ping h2"):
        _post(cmd, "s2", exit_code=1, output="t")
    _post("ping h", "s2", exit_code=1, output="t")   # h 累计2次
    _post("ping h2", "s2", exit_code=1, output="t")  # h2 累计2次, last=h2
    r = _pre("ping h", "s2")  # h 非 last → 走 R2
    assert r.get("action") == "block" and "2 次" in r["message"]


def test_r2_success_resets_counter():
    _post("ping z", "s3", exit_code=1, output="t")
    _post("ping z", "s3", exit_code=1, output="t")
    _post("ping z", "s3", exit_code=0, output="ok")
    assert not _pre("ping z", "s3")


def test_r3_no_registry_blocked():
    r = _pre("bash deploy/deploy.sh cloud loom-backend", "s4")
    assert r.get("action") == "block" and "--list" in r["message"]


def test_r3_registry_collected_then_pass():
    _post("bash deploy/build.sh --list", "s5", output=REGISTRY)
    assert not _pre("bash deploy/deploy.sh cloud loom-backend", "s5")


def test_r3_repo_name_masquerade_blocked():
    _post("bash deploy/build.sh --list", "s6", output=REGISTRY)
    r = _pre("bash deploy/deploy.sh cloud loom", "s6")  # 仓库名非服务名
    assert r.get("action") == "block"


def test_r3_nonexistent_service_blocked():
    _post("bash deploy/build.sh --list", "s7", output=REGISTRY)
    assert _pre("bash deploy/deploy.sh cloud loom-frontend", "s7").get("action") == "block"


def test_r4_raw_docker_logs_blocked():
    assert _pre("docker logs ontox-loom-backend", "s8").get("action") == "block"
    assert not _pre("docker logs x --tail 100 2>&1 | grep ERROR | head -20", "s8")
    assert not _pre("docker logs x --since 5m", "s8")


def test_r5_sleep_wait_blocked():
    assert _pre("sleep 30", "s9").get("action") == "block"
    assert not _pre("sleep 3 && curl -s localhost/health", "s9")
    # 真机 0826 暴露的误拦修复: 纯短 sleep 放行
    assert not _pre("sleep 3", "s9")
    assert not _pre("sleep 5", "s9")


def test_r5_background_sleep_allowed():
    """真机暴露: background=true 的 sleep 是合法长任务姿势, 永不拦。"""
    r = m._on_pre_tool_call(
        tool_name="terminal",
        args={"command": "sleep 30", "background": True},
        session_id="s9b")
    assert not r
    r = m._on_pre_tool_call(
        tool_name="terminal",
        args={"command": "sleep 120", "background": True, "notify_on_complete": True},
        session_id="s9b")
    assert not r


def test_r6_diagnostic_stderr_swallowed_blocked():
    assert _pre("curl -s localhost:9222/json 2>/dev/null", "s10").get("action") == "block"
    assert not _pre("npm run build 2>/dev/null", "s10")


def test_reminder_after_failure():
    _post("ping r", "s11", exit_code=1, output="t")
    r = m._on_pre_llm_call(session_id="s11")
    assert "瞎猜纪律提醒" in r.get("context", "")
    assert not m._on_pre_llm_call(session_id="s11")  # 一次性


def test_readonly_never_blocked():
    assert not _pre("grep -rn foo src/", "s12")
    assert not _pre("bash deploy/build.sh --list", "s12")


# ============ 升级机制：三级惩罚阶梯 ============

def test_escalation_l1_to_l2_to_l3():
    """累犯3次→L2警告;8次→L3;记档连续可查。用临时HERMES_HOME隔离。"""
    with tempfile.TemporaryDirectory() as td:
        with mock.patch.dict(os.environ, {"HERMES_HOME": td}):
            sid = "esc1"
            # 第1-2次: L1 纯拦截
            r = _pre("sleep 30", sid)
            assert r["action"] == "block" and "累犯" not in r["message"]
            _pre("sleep 30", sid)
            _pre("sleep 30", sid)  # 第3次触发时按已记2次算 → L2
            r = _pre("sleep 30", sid)
            assert "L2" in r["message"] and "累犯" in r["message"], r["message"][:80]
            # 补到 8 次 → L3
            for _ in range(5):
                _pre("sleep 30", sid)
            r = _pre("sleep 30", sid)
            assert "L3" in r["message"], r["message"][:80]
            # 记档数=触发次数, 每条带 level
            cnt, _last, _d = m._violation_stats("R5")
            assert cnt == 10  # 3(L1)+1(L2过渡)+... 逐次记档累计


def test_escalation_l3_narrows_sleep_limit():
    """R5 达到 L3 后, sleep 5(原放行区间)也被拦。"""
    with tempfile.TemporaryDirectory() as td:
        with mock.patch.dict(os.environ, {"HERMES_HOME": td}):
            sid = "esc2"
            with mock.patch.object(m, "_violation_stats",
                                   return_value=(8, "2026-08-26T00:00:00", 0)):
                assert m._current_level("R5") == "L3"
            # 直接播种前科再验证收窄: 清档案后写8条
            for i in range(8):
                m._record_violation("R5", f"sleep 30 #{i}", "L1", sid)
            assert m._current_level("R5") == "L3"
            # sleep 5 && 实质操作 原本放行(limit=10), L3 下被拦(limit=3)
            assert _pre("sleep 5 && curl -s localhost/health", sid)["action"] == "block"


def test_escalation_rules_independent():
    """R5 前科不影响 R4 判级。"""
    with tempfile.TemporaryDirectory() as td:
        with mock.patch.dict(os.environ, {"HERMES_HOME": td}):
            for i in range(5):
                m._record_violation("R5", f"sleep 30 #{i}", "L1", "esc3")
            assert m._current_level("R5") == "L2"
            assert m._current_level("R4") == "L1"


def test_escalation_demotes_after_clean_window():
    """最近一次违规>14天 → 不再L3（降级条件可验证）。"""
    with tempfile.TemporaryDirectory() as td:
        with mock.patch.dict(os.environ, {"HERMES_HOME": td}):
            with mock.patch.object(m, "_violation_stats",
                                   return_value=(10, "2026-07-01T00:00:00", 56)):
                assert m._current_level("R5") == "L2"  # 计数够但太旧, 不进L3


def test_escalation_records_even_when_db_missing():
    """库不可用时拦截不崩、照常返回 block。"""
    with mock.patch.dict(os.environ, {"HERMES_HOME": "/nonexistent-xyz"}):
        r = _pre("sleep 30", "esc4")
        assert r["action"] == "block"
