"""no-guessing 插件单测——瞎猜根治闸门六规则 + 回归。
运行: cd ~/code/ai/github/fork/hermes-agent && python3 -m pytest tests/test_no_guessing_plugin.py -q
"""
import sys
from pathlib import Path

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
