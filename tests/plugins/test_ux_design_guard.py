"""ux-design-guard 行为契约测试——U1/U2/U3 三态（拦/放/豁免）与范围门。

理论依据: ~/knowledge-base/ux-design/UX_INTERACTION_METHODOLOGY.md
(Cooper「用过渡解释变化」: 动效唯一合法用途是澄清状态;
 Material/Fluent 档位: 小位移 150-200ms/大位移 300-400ms,超 400ms=迟滞)。
期望值从理论独立推导，非从实现反推。
"""

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_MOD_PATH = REPO / "plugins" / "ux-design-guard" / "__init__.py"
_spec = importlib.util.spec_from_file_location("ux_design_guard_under_test", _MOD_PATH)
assert _spec is not None and _spec.loader is not None  # 期望: 测试文件加载失败立即显式报错
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


def test_u3_blocks_over_tier_duration():
    css = ".panel-slide {\n  transition: transform .8s ease;\n}\n"
    verdict = _call("/tmp/app/src/panel.css", css)
    assert verdict.get("action") == "block" and "U3" in verdict["message"]  # 期望: 800ms 超 400ms 档且无状态语义,被拦


def test_u3_allows_in_tier_duration():
    css = ".fade {\n  transition: opacity .25s ease;\n}\n"
    assert _call("/tmp/app/src/fade.css", css) == {}  # 期望: 250ms 档内放行


def test_u3_allows_long_state_semantic_duration():
    css = ".progress-fill {\n  transition: width 1.2s ease-out;\n}\n"
    assert _call("/tmp/app/src/progress.css", css) == {}  # 期望: 进度条填充 1200ms 属状态语义,白名单放行


def test_u3_ignores_delay_property_line():
    css = ".seq {\n  transition-delay: 0.6s;\n}\n"
    assert _call("/tmp/app/src/seq.css", css) == {}  # 期望: 纯延迟属性行是 U2 域,U3 不检


def test_u3_shorthand_first_time_value_is_duration():
    css = ".card {\n  transition: all .5s .2s ease;\n}\n"
    verdict = _call("/tmp/app/src/card.css", css)
    assert verdict.get("action") == "block" and "U3" in verdict["message"]  # 期望: shorthand 首时间值 500ms 为时长,超档被拦


def test_u3_shorthand_ok_duration_large_delay_passes():
    css = ".y {\n  transition: all .3s .8s ease;\n}\n"
    assert _call("/tmp/app/src/y.css", css) == {}  # 期望: 首时间值 300ms 档内,其后 800ms 是延迟,不属 U3


def test_u3_blocks_ms_unit_duration():
    css = ".drawer {\n  animation-duration: 600ms;\n}\n"
    verdict = _call("/tmp/app/src/drawer.css", css)
    assert verdict.get("action") == "block" and "U3" in verdict["message"]  # 期望: 600ms 毫秒单位同样超档被拦


def test_u4_blocks_small_button_font():
    css = ".task .t-go {\n  font-size: 12px;\n}\n"
    verdict = _call("/tmp/app/src/go.css", css)
    assert verdict.get("action") == "block" and "U4" in verdict["message"]  # 期望: 按钮语义选择器下 12px 被拦


def test_u4_blocks_inline_small_button_font():
    css = "button.mini { font-size: 12px; }\n"
    verdict = _call("/tmp/app/src/mini.css", css)
    assert verdict.get("action") == "block" and "U4" in verdict["message"]  # 期望: 单行 CSS 同样命中


def test_u4_allows_non_button_small_font():
    css = ".sub-label {\n  font-size: 12px;\n}\n"
    assert _call("/tmp/app/src/sub.css", css) == {}  # 期望: 无按钮语义的辅助文本不受限


def test_u4_allows_qualified_button_font():
    css = ".pager .pgbtn {\n  font-size: 13px;\n}\n"
    assert _call("/tmp/app/src/pg.css", css) == {}  # 期望: 达 13px 下限放行


def test_u4_allows_non_literal_font_size():
    css = ".btn {\n  font-size: var(--el-font-size-base);\n}\n"
    assert _call("/tmp/app/src/varbtn.css", css) == {}  # 期望: 非字面量(var)静态不可判,fail-open


def test_u4_exemption_channel():
    css = ".icon-only-btn {\n  font-size: 12px; /* ux-ok: 纯图标按钮无文本,尺寸由点击目标24px判据约束 */\n}\n"
    assert _call("/tmp/app/src/icon.css", css) == {}  # 期望: 行内豁免(≥8字理由)放行


def test_u5_blocks_squeezed_body_line_height():
    css = ".article-body {\n  font-size: 14px;\n  line-height: 15.4px;\n}\n"
    verdict = _call("/tmp/app/src/body.css", css)
    assert verdict.get("action") == "block" and "U5" in verdict["message"]  # 期望: 正文14px行高1.1被拦


def test_u5_blocks_unitless_ratio_out_of_window():
    css = ".desc { font-size: 13px; line-height: 1.1; }\n"
    verdict = _call("/tmp/app/src/desc.css", css)
    assert verdict.get("action") == "block" and "U5" in verdict["message"]  # 期望: 无单位比值1.1越下限被拦


def test_u5_allows_healthy_ratio():
    css = ".article-body {\n  font-size: 14px;\n  line-height: 22px;\n}\n"
    assert _call("/tmp/app/src/ok.css", css) == {}  # 期望: 比值1.57在[1.4,1.9]内放行


def test_u5_allows_heading_squeeze():
    css = "h2.title {\n  font-size: 16px;\n  line-height: 18px;\n}\n"
    assert _call("/tmp/app/src/h2.css", css) == {}  # 期望: 标题压缩行高是排印惯例,豁免


def test_u5_allows_non_body_font_size():
    css = ".hero-big {\n  font-size: 24px;\n  line-height: 26px;\n}\n"
    assert _call("/tmp/app/src/hero.css", css) == {}  # 期望: 24px 非正文字号窗口,fail-open


def test_u5_allows_no_font_size_context():
    css = ".dense {\n  line-height: 15px;\n}\n"
    assert _call("/tmp/app/src/dense.css", css) == {}  # 期望: 无 font-size 上下文静态不可判,fail-open


def _call_patch(path: str, old: str, new: str) -> dict:
    return guard.on_pre_tool_call(tool_name="patch", args={"path": path, "old_string": old, "new_string": new})


def test_u4_patch_single_line_shrink_blocked(tmp_path):
    """patch 单行改小(new_string 无选择器)是最高频写入形态——回读文件拼上下文后必须拦。"""
    target = tmp_path / "btn.css"
    target.write_text(".task .t-go {\n  font-size: 14px;\n}\n")
    verdict = _call_patch(str(target), "  font-size: 14px;", "  font-size: 12px;")
    assert verdict.get("action") == "block" and "U4" in verdict["message"]  # 期望: 回读上下文后单行改小被拦


def test_u4_important_suffix_blocked():
    css = ".btn-primary {\n  font-size: 12px !important;\n}\n"
    verdict = _call("/tmp/app/src/imp.css", css)
    assert verdict.get("action") == "block" and "U4" in verdict["message"]  # 期望: !important 对抗写法被拦


def test_u4_font_shorthand_blocked():
    css = ".btn-cta {\n  font: 500 12px/1.4 system-ui;\n}\n"
    verdict = _call("/tmp/app/src/shorthand.css", css)
    assert verdict.get("action") == "block" and "U4" in verdict["message"]  # 期望: font 缩写 12px 被拦


def test_u4_active_state_class_not_misjudged():
    css = ".tab.active {\n  font-size: 12px;\n}\n"
    assert _call("/tmp/app/src/tab.css", css) == {}  # 期望: .active 状态类是字母相连,词界排除非按钮


def test_u4_hyphen_prefix_button_still_blocked():
    css = ".row .ops-btn {\n  font-size: 12px;\n}\n"
    verdict = _call("/tmp/app/src/ops.css", css)
    assert verdict.get("action") == "block" and "U4" in verdict["message"]  # 期望: ops-btn 连字符前缀是按钮语义,仍拦


def test_u4_declared_boundary_tailwind_fail_open_inline_now_blocked():
    """边界声明: tailwind 类映射静态不可判(放行); 内联 style 与 button 标签同行可判(被拦)。"""
    assert _call("/tmp/app/src/tw.html", '<button class="text-xs">go</button>') == {}  # 期望: 类系统映射静态不可判,放行
    verdict = _call("/tmp/app/src/inline.html", '<button style="font-size:12px">go</button>')
    assert verdict.get("action") == "block" and "U4" in verdict["message"]  # 期望: button 标签+内联 font-size 同行可判,拦
