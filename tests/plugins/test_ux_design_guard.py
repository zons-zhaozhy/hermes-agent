"""ux-design-guard 行为契约测试——U1/U2 三态（拦/放/豁免）与范围门。

理论依据: ~/knowledge-base/ux-design/UX_INTERACTION_METHODOLOGY.md
(Cooper「用过渡解释变化」: 动效唯一合法用途是澄清状态)。
期望值从理论独立推导，非从实现反推。
"""

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_MOD_PATH = REPO / "plugins" / "ux-design-guard" / "__init__.py"
_spec = importlib.util.spec_from_file_location("ux_design_guard_under_test", _MOD_PATH)
guard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(guard)


def _call(path: str, content: str) -> dict:
    return guard.on_pre_tool_call(tool_name="write_file", args={"path": path, "content": content})


def test_u1_blocks_decorative_infinite():
    css = ".celebrate-badge {\n  animation: spin 2s linear infinite;\n}\n"
    verdict = _call("/tmp/app/src/badge.css", css)
    assert verdict.get("action") == "block" and "U1" in verdict["message"]  # 期望: 无状态语义的无限动画被拦


def test_u1_allows_state_semantic_animation():
    css = ".loading-spinner {\n  animation: rot 1s linear infinite;\n}\n"
    assert _call("/tmp/app/src/spin.css", css) == {}  # 期望: loading 状态语义白名单放行


def test_u1_exemption_channel():
    css = ".orbit-ring {\n  animation: spin 8s linear infinite; /* ux-ok: 轨道环表现行星运转,叙事必需 */\n}\n"
    assert _call("/tmp/app/src/orbit.css", css) == {}  # 期望: 行内豁免(≥8字理由)放行


def test_u2_blocks_queue_in_animation_delay():
    vue = "<div v-for=\"(it, i) in items\" :style=\"{ animationDelay: (i * 0.08) + 's' }\">{{ it }}</div>\n"
    verdict = _call("/tmp/app/src/List.vue", vue)
    assert verdict.get("action") == "block" and "U2" in verdict["message"]  # 期望: 索引乘法队列入场被拦


def test_u2_allows_static_animation_delay():
    css = ".stagger-fixed {\n  animation-delay: 0.2s;\n}\n"
    assert _call("/tmp/app/src/fixed.css", css) == {}  # 期望: 固定值不属队列模式,放行


def test_non_ui_file_passes():
    assert _call("/tmp/app/src/logic.py", "x = 1\n") == {}  # 期望: 非 UI 后缀不在机检范围


def test_build_artifact_skipped():
    css = ".x { animation: spin 1s infinite; }\n"
    assert _call("/tmp/app/dist/assets/app.css", css) == {}  # 期望: dist 构建产物跳过


def test_patch_new_string_channel():
    verdict = guard.on_pre_tool_call(
        tool_name="patch",
        args={"path": "/tmp/app/src/B.vue", "old_string": "a", "new_string": "<i :style=\"{animationDelay: i*0.1+'s'}\"/>"},
    )
    assert verdict.get("action") == "block"  # 期望: patch 通道同样受检
