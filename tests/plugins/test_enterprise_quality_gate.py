"""enterprise-quality-gate 回归测试。

覆盖行为契约（期望值独立推导自 core 契约源，非实现反推）：

1. L2 门禁在官方 pre_verify 载荷（final_response/changed_paths）下真实触发
   ——契约源 hermes_cli/hooks.py:121 与 plugins.py:2111（载荷无 response_text/cwd 键）
2. 返回体形状是 core 消费端可用的续跑指令
   ——契约源 plugins.py::get_pre_verify_continue_message 只认 action|decision + message|reason
3. 无声明关键词 / 已有计分卡 / 非应用工程 → 静默
4. 同会话一次（状态落在 (session_id, namespace) 桶内，参数序正确）
5. 稳定段：主会话注入、subagent/batch 排除

Contract:
  Preconditions: 无真实 LLM、无网络；全部路径落在 pytest tmp_path
  Postconditions: 全绿 = 五条契约成立
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_plugin() -> Any:
    """按绝对导入路径加载插件模块（插件内 `from plugins import _shared_state`）。"""
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    spec = importlib.util.spec_from_file_location(
        "eqg_under_test",
        _REPO_ROOT / "plugins" / "enterprise-quality-gate" / "__init__.py",
    )
    assert spec is not None and spec.loader is not None  # 期望: 插件文件存在，否则测试自身配置错
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeCtx:
    """收集 register_hook / register_system_prompt_section 的注册物。"""

    def __init__(self) -> None:
        self.hooks: dict[str, Any] = {}
        self.sections: dict[str, Any] = {}

    def register_hook(self, name: str, fn: Any) -> None:
        self.hooks[name] = fn

    def register_system_prompt_section(self, section_id: str, fn: Any) -> None:
        self.sections[section_id] = fn


def _wired() -> tuple[Any, Any]:
    mod = _load_plugin()
    ctx = _FakeCtx()
    mod.register(ctx)
    return mod, ctx


def _app_project(tmp_path: Path) -> Path:
    """造一个「纲领应用工程」目录：含 apps/ 子目录、无 QUALITY_SCORECARD.md。"""
    proj = tmp_path / "proj"
    (proj / "apps").mkdir(parents=True)
    return proj


def _payload(paths: list[str], reply: str = "本项目已达成企业级可交付标准",
             sid: str = "sess-main") -> dict[str, Any]:
    """官方 pre_verify 载荷字段（hermes_cli/hooks.py:121）。

    会话号按测试唯一化——_shared_state 是进程级全局，同文件内跨测试共享。
    """
    return dict(
        session_id=sid,
        platform="cli",
        model="gpt",
        coding=True,
        attempt=0,
        final_response=reply,
        changed_paths=paths,
    )


# ── 1+2. 官方载荷触发 + 返回体可被 core 消费 ─────────────────────────────────

def test_gate_fires_on_official_payload_and_returns_consumable_shape(tmp_path: Path) -> None:
    _mod, ctx = _wired()
    proj = _app_project(tmp_path)
    res = ctx.hooks["pre_verify"](**_payload([str(proj / "apps" / "a.py")]))
    # 期望: 关键词命中 + 应用工程 + 无计分卡 → 真实触发（原实现读 response_text 恒不触发）
    assert isinstance(res, dict)  # 期望: 触发返回 dict 而非 None
    # 期望: core 消费端只认 action|decision + message|reason（{"context":...} 会被丢弃）
    action = str(res.get("action") or res.get("decision") or "").lower()
    message = res.get("message") or res.get("reason")
    assert action in ("continue", "block")  # 期望: action 落在 core 认可的两值内
    assert isinstance(message, str) and message.strip()  # 期望: message 为非空字符串
    assert "QUALITY_SCORECARD" in message  # 期望: 提示文本点名缺失的计分卡文件


def test_gate_ignores_non_official_keys(tmp_path: Path) -> None:
    """载荷若只给 response_text/cwd（非官方键）→ 不应触发，证明契约键名已按官方口径。"""
    _mod, ctx = _wired()
    proj = _app_project(tmp_path)
    res = ctx.hooks["pre_verify"](
        session_id="sess-legacy", platform="cli",
        response_text="企业级可交付", cwd=str(proj),
    )
    # 期望: 非官方载荷不含 final_response → 关键词判据取不到声明 → 静默
    assert res is None  # 期望: 非官方键不被采纳


# ── 3. 静默条件 ────────────────────────────────────────────────────────────

def test_silent_without_claim_keyword(tmp_path: Path) -> None:
    _mod, ctx = _wired()
    proj = _app_project(tmp_path)
    res = ctx.hooks["pre_verify"](**_payload([str(proj / "apps" / "a.py")], reply="改完了，谢谢"))
    assert res is None  # 期望: 无声明关键词 → 不拦


def test_silent_when_scorecard_exists(tmp_path: Path) -> None:
    _mod, ctx = _wired()
    proj = _app_project(tmp_path)
    (proj / "apps" / "a" / "docs").mkdir(parents=True)
    (proj / "apps" / "a" / "docs" / "QUALITY_SCORECARD.md").write_text("# 计分卡\n", encoding="utf-8")
    res = ctx.hooks["pre_verify"](**_payload([str(proj / "apps" / "a" / "x.py")], reply="企业级验收完成"))
    assert res is None  # 期望: 计分卡在场 → 不拦


def test_silent_outside_app_project(tmp_path: Path) -> None:
    _mod, ctx = _wired()
    plain = tmp_path / "plain"
    plain.mkdir()
    res = ctx.hooks["pre_verify"](**_payload([str(plain / "a.py")]))
    # 期望: 无 apps/ 与 docs/standards/ 信号 → 视为非应用仓库，不拦（防误报）
    assert res is None  # 期望: 非应用工程目录静默


# ── 4. 同会话一次 + 状态桶参数序 ───────────────────────────────────────────

def test_once_per_session_and_state_namespace(tmp_path: Path) -> None:
    plugin, ctx = _wired()
    proj = _app_project(tmp_path)
    first = ctx.hooks["pre_verify"](**_payload([str(proj / "apps" / "a.py")], reply="企业级达标", sid="sess-once"))
    second = ctx.hooks["pre_verify"](**_payload([str(proj / "apps" / "a.py")], reply="企业级达标", sid="sess-once"))
    assert isinstance(first, dict)  # 期望: 首轮触发
    assert second is None  # 期望: 同会话第二次静默
    # 期望: 状态落在 (session_id, namespace) 桶内——参数序必须是 (sid, namespace)
    bucket = plugin._shared_state.get_session_state("sess-once", "enterprise_quality_gate")
    assert bucket.get("reminded") is True  # 期望: 标记写入正确命名空间桶


# ── 5. 稳定段 ──────────────────────────────────────────────────────────────

def test_section_injected_for_main_and_excluded_for_subagent() -> None:
    _mod, ctx = _wired()
    render = ctx.sections["enterprise_quality_gate"]
    main = render({"platform": "cli"})
    sub = render({"platform": "subagent"})
    assert main and "企业级" in main  # 期望: 主会话注入纪律框架
    assert sub == ""  # 期望: subagent 平台排除
