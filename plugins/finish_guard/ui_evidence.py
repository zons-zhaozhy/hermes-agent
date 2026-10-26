"""finish_guard L4 — UI 交付用户旅程证据判定。

含界面/文案源文件的交付, 收口前必须存在走查证据文件
<仓库根>/.hermes/ui_walkthrough.json 且证据时间戳晚于全部 UI 改动文件——
技术证据(pytest/vue-tsc/curl/构建成功)不构成 UI 交付的完成证据,
「陌生用户能否无帮助走通」由本判定降级为硬闸。

证据文件结构:
  {"walkthroughs": [
      {"url": "http://…", "checked_at": 1730000000,
       "assertions": [{"claim": "顶栏显示 身份 · admin", "result": "pass"}]}
  ]}
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("hermes.plugins.finish_guard.ui_evidence")

_TIME_TOLERANCE_S = 5  # 文件系统 mtime 精度容差
_FUTURE_TOLERANCE_S = 120  # checked_at 上界容差——未来时间戳=永真伪证, 必拒
_SCREENSHOT_WINDOW_S = 120  # 截图文件 mtime 与 checked_at 的允许偏差
_UI_SUFFIXES = (".vue", ".tsx", ".jsx", ".html", ".css", ".scss")
# 前端 .ts（router/i18n/store 等）同属用户可感知界面源——skb /compare 断链
# 根因即 router/index.ts 路由修改, 不识别则路由类改动整体漏拦。
# 范围限定前端目录特征, 避免把任意工具脚本 .ts 算进 UI 源。
_FRONTEND_TS_SEGMENTS = ("/src/", "/frontend/")
_NON_SOURCE_SEGMENTS = ("/dist/", "/node_modules/", "/build/", "/.hermes/cache/")
_EVIDENCE_NAME = "ui_walkthrough.json"


def _is_ui_source(path: str) -> bool:
    """Contract: Preconditions: path 为 str;
    Postconditions: 源码形态的界面/样式/前端路由文件返回 True,
    构建产物与依赖目录内返回 False。
    """
    if any(seg in path for seg in _NON_SOURCE_SEGMENTS):
        return False
    if path.endswith(_UI_SUFFIXES):
        return True
    return path.endswith(".ts") and any(seg in path for seg in _FRONTEND_TS_SEGMENTS)


def _repo_root(path: str) -> Path | None:
    """Contract: Preconditions: path 为 str;
    Postconditions: 祖先链上含 .git 的目录返回其 Path, 无则 None。
    """
    p = Path(path)
    for cand in [p, *p.parents]:
        if (cand / ".git").exists():
            return cand
    return None


def _validate_assertions(idx: int, w: dict) -> str | None:
    """Contract: Preconditions: w 为单条 walkthrough 字典, idx 为其序号;
    Postconditions: assertions 非空且每条含 claim 返回 None, 否则返回缺陷说明;
    可达性走查断言（claim 含「可达」锚）至少 1 条, 缺则返回缺陷说明。
    """
    assertions = w.get("assertions")
    if not isinstance(assertions, list) or not assertions:
        return f"walkthroughs[{idx}] 缺 assertions（至少 1 条, 每条 {{claim, result}}）"
    for j, a in enumerate(assertions):
        if not isinstance(a, dict) or not a.get("claim"):
            return f"walkthroughs[{idx}].assertions[{j}] 缺 claim"
    if not any("可达" in str(a.get("claim")) for a in assertions if isinstance(a, dict)):
        return (f"walkthroughs[{idx}] 缺全站可达性断言——至少 1 条 claim 含「可达」"
                "（枚举页面全部链接/导航目标逐一实访, 断言非 404/非静默回退）")
    return None


def _validate_walkthrough(idx: int, w: Any, min_checked_at: float,
                          root: Path | None = None,
                          ui_file: str = "") -> str | None:
    """Contract: Preconditions: w 为 walkthroughs[idx], min_checked_at 为 epoch 秒,
    root 为仓库根(可空——空则跳过 dom_assets 锚), ui_file 为首个 UI 源文件路径;
    Postconditions: 结构/时间上下界/截图锚/dom 锚全过返回 None, 否则返回缺陷说明。
    """
    if not isinstance(w, dict) or not w.get("url"):
        return f"walkthroughs[{idx}] 缺 url"
    checked_at = w.get("checked_at")
    if not isinstance(checked_at, (int, float)):
        return f"walkthroughs[{idx}] 缺 checked_at 时间戳"
    if checked_at < min_checked_at - _TIME_TOLERANCE_S:
        return (f"walkthroughs[{idx}] 的 checked_at({checked_at}) 早于 UI 改动文件"
                f" mtime({min_checked_at:.0f})——证据须在最后一次界面改动之后取得")
    if checked_at > time.time() + _FUTURE_TOLERANCE_S:
        return (f"walkthroughs[{idx}] 的 checked_at({checked_at}) 是未来时间"
                f"(now={time.time():.0f})——自报时间戳只查结构不防伪, 上界必拒")
    shot_defect = _validate_screenshots(idx, w, checked_at)
    if shot_defect:
        return shot_defect
    if root is not None:
        dom_defect = _validate_dom_assets(idx, w, Path(ui_file), min_checked_at)
        if dom_defect:
            return dom_defect
    return _validate_assertions(idx, w)


def _validate_role_coverage(data: dict, walkthroughs: list) -> str | None:
    """Contract: Preconditions: data 为证据顶层 dict, walkthroughs 为已过结构校验的列表;
    Postconditions: 顶层 "roles" 声明项目全部用户角色时, 每个声明角色被
    至少一条 walkthrough 的 role 覆盖则返回 None, 否则返回缺陷说明;
    未声明 roles 的单角色项目返回 None（不误拦）。
    """
    declared = data.get("roles")
    if not isinstance(declared, list) or not declared:
        return None
    covered = {str(w.get("role") or "") for w in walkthroughs if isinstance(w, dict)}
    missing = [r for r in declared if str(r) not in covered]
    if missing:
        return (f"多角色走查未覆盖: 声明 roles={declared} 但缺角色 {missing}"
                "——每个角色须以该身份各走一遍并各留至少 1 条 walkthrough(role=…)")
    return None


def _validate_screenshots(idx: int, w: dict, checked_at: float) -> str | None:
    """Contract: Preconditions: w 为单条 walkthrough, checked_at 为其时间戳;
    Postconditions: screenshots 非空且每个为存在的绝对路径文件、mtime 与
    checked_at 偏差在窗口内返回 None, 否则返回缺陷说明。
    """
    shots = w.get("screenshots")
    if not isinstance(shots, list) or not shots:
        return (f"walkthroughs[{idx}] 缺 screenshots（至少 1 张走查截图绝对路径, "
                "dpr≥3——截图与走查同时刻产生）")
    for s in shots:
        p = Path(str(s))
        if not p.is_absolute():
            return f"walkthroughs[{idx}].screenshots 须绝对路径: {s}"
        if not p.exists():
            return f"截图不存在: {s}"
        if abs(p.stat().st_mtime - checked_at) > _SCREENSHOT_WINDOW_S:
            return (f"截图 {s} 的 mtime 与 checked_at 偏差超 {_SCREENSHOT_WINDOW_S}s"
                    "——截图须与走查同时刻产生, 禁挪用旧图")
    return None


def _load_asset_refs(ui_file: Path) -> tuple[set[str], float] | None:
    """Contract: Preconditions: ui_file 为 UI 源文件路径(其祖先链含仓库根);
    Postconditions: 由近及远在每级祖先探测 dist*/build 下的 index.html,
    首个含 JS 引用者返回 (引用文件名集合, 其 mtime); 未探测到返回 None(降级)。
    monorepo 应用产物形如 apps/<app>/dist/lab-ui/index.html, 由祖先链遍历覆盖。
    """
    for base in [ui_file.parent, *ui_file.parents]:
        candidates = [
            base / "dist" / "index.html",
            base / "build" / "index.html",
            *base.glob("dist*/index.html"),
            *base.glob("dist*/*/index.html"),
        ]
        for idx_html in candidates:
            if not idx_html.exists():
                continue
            try:
                text = idx_html.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            refs = set()
            for seg in text.split('src="')[1:]:
                name = seg.split('"', 1)[0]
                if name.endswith((".js", ".mjs")):
                    refs.add(name.rsplit("/", 1)[-1])
            if refs:
                return refs, idx_html.stat().st_mtime
        if (base / ".git").exists():
            break  # 仓库根层已探测, 不越仓库边界
    logger.info("ui_journey: 未探测到构建产物 index.html, dom_assets 锚降级")
    return None


def _validate_dom_assets(idx: int, w: dict, ui_file: Path, min_checked_at: float) -> str | None:
    """Contract: Preconditions: ui_file 为首个 UI 源文件路径, min_checked_at 为
    UI 源文件最大 mtime;
    Postconditions: 构建产物新鲜且 dom_assets 与产物引用有交集返回 None,
    否则返回缺陷说明; 无构建产物时返回 None(降级)。
    """
    refs = _load_asset_refs(ui_file)
    if refs is None:
        return None
    asset_names, bundle_mtime = refs
    if bundle_mtime < min_checked_at - _TIME_TOLERANCE_S:
        return (f"构建产物 index.html mtime({bundle_mtime:.0f}) 早于 UI 源文件"
                f"({min_checked_at:.0f})——改码后未重建, 页面跑的是旧产物")
    declared = w.get("dom_assets")
    if not isinstance(declared, list) or not declared:
        return (f"walkthroughs[{idx}] 缺 dom_assets（走查时页面实际加载的 JS 文件名"
                "列表, 浏览器执行 [...document.scripts].map(s=>s.src.split('/').pop()) 取值）")
    if not asset_names & {str(d) for d in declared}:
        return (f"walkthroughs[{idx}].dom_assets 与当前构建产物引用零交集"
                f"（产物引用 {sorted(asset_names)[:5]}）——须从真实加载页面读取, "
                "证明走查发生在当前构建版本上")
    return None


def _validate_evidence(data: Any, min_checked_at: float,
                       root: Path | None = None,
                       ui_file: str = "") -> str | None:
    """Contract: Preconditions: data 为 json.loads 产物;
    Postconditions: 存在至少一条结构校验全过的 walkthrough 且角色覆盖达标返回 None
    (账本累积语义: 历史段早于本次改动 mtime 属正常, 不得因旧段缺陷否定新证据),
    否则返回首条缺陷说明。
    """
    if not isinstance(data, dict):
        return "证据文件顶层不是 JSON 对象"
    walkthroughs = data.get("walkthroughs")
    if not isinstance(walkthroughs, list) or not walkthroughs:
        return "缺 walkthroughs 数组（至少 1 条走查记录）"
    first_defect = None
    valid: list = []
    for idx, w in enumerate(walkthroughs):
        defect = _validate_walkthrough(idx, w, min_checked_at, root, ui_file)
        if defect:
            first_defect = first_defect or defect
        else:
            valid.append(w)
    if not valid:
        return first_defect or "全部 walkthrough 校验失败"
    return _validate_role_coverage(data, valid)


def _check_evidence(ui_files: list[str]) -> str | None:
    """Contract: Preconditions: ui_files 为非空 UI 源文件路径列表;
    Postconditions: 证据存在且合法返回 None, 否则返回「缺失」或缺陷说明。
    """
    root = _repo_root(ui_files[0])
    if root is None:
        return None  # 无仓库根上下文: 不拦, 交由其他门禁
    evidence = root / ".hermes" / _EVIDENCE_NAME
    if not evidence.exists():
        return "缺失"
    min_checked_at = max(
        (Path(f).stat().st_mtime for f in ui_files if Path(f).exists()), default=0
    )
    try:
        data = json.loads(evidence.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("ui_journey 证据文件读取失败: %s", e)
        return f"证据文件不可读/非 JSON: {e}"
    return _validate_evidence(data, min_checked_at, root, ui_files[0])


def nudge_message(ui_files: list[str], defect: str) -> str:
    """Contract: Preconditions: ui_files 非空, defect 为缺陷说明;
    Postconditions: 返回含证据格式与断言要求的强制续跑提示文本。
    """
    return (
        f"[finish_guard/L4-ui] 本回合改动了 {len(ui_files)} 个界面/文案/前端路由源文件"
        f"（如 {ui_files[0]}），但没有有效的用户走查证据（{defect}）。"
        "技术证据（pytest/vue-tsc/curl/构建成功）不构成 UI 交付的完成证据——"
        "应用面向最终用户, 验收单位=陌生用户完整旅程。请完成两类走查后把证据写入仓库根"
        f" .hermes/{_EVIDENCE_NAME} 再收口:\n"
        "1) 全站可达性: 枚举页面全部链接/导航/正文/页脚目标, 逐一实访, 断言非 404 且"
        "非静默回退——断链必须当轮暴露;\n"
        "2) 陌生用户旅程: 以冷缓存/清 sessionStorage 状态进入界面, 以目标用户身份真实点击"
        "走一遍（逐导航真点→全页滚动→多语言/多角色各走一遍→手机宽度复走）。\n"
        '证据格式: {"roles": ["public", "staff"], "walkthroughs": '
        '[{"url": "<实际访问地址>", "role": "<角色>", "checked_at": '
        f'{int(time.time())}, "screenshots": ["<走查截图绝对路径(dpr≥3)>"], '
        '"dom_assets": ["<页面实际加载的 JS 文件名, 取 '
        "[...document.scripts].map(s=>s.src.split('/').pop())>\"], "
        '"assertions": [{"claim": "<界面某处显示什么/全部导航可达>", '
        '"result": "pass"}]}]}\n'
        "防伪三锚(缺一即拦): checked_at 须在 [最后改动 mtime, now+120s] 区间(未来时间"
        "=伪证); screenshots 截图须存在且 mtime 与 checked_at 偏差≤120s(禁挪旧图); "
        "dom_assets 须与当前构建产物引用有交集且产物 mtime 晚于源码(改码必须重建+从"
        "真实页面取值)。\n"
        "断言须覆盖: 首屏第一眼能看懂这是什么、不知道内部术语的用户能否找到下一步动作、"
        "须身份的操作有友好获取指引、本次改动点在真实页面上生效、全站链接可达。"
        "多角色项目（公开访客/登录用户/管理端等）须在顶层 roles 声明并逐角色覆盖。"
        "证据须在最后一次界面改动之后取得（checked_at 晚于改动文件 mtime）。"
    )


def check(changed_paths: list[str], attempt: int, max_nudges: int) -> dict:
    """Contract: Preconditions: changed_paths 为本回合改动路径, attempt 为续跑轮次;
    Postconditions: UI 源文件存在、证据缺失/不合法且 attempt 未达上限时返回
    {"action":"continue","message":...}; 其余情形返回 {} 放行。
    """
    if attempt >= max_nudges:
        return {}
    ui_files = [p for p in changed_paths if _is_ui_source(p)]
    if not ui_files:
        return {}
    defect = _check_evidence(ui_files)
    if defect is None:
        return {}
    logger.warning(
        "finish_guard L4-ui: UI 改动(%d 文件)无有效走查证据(%s), attempt=%d, 强制续跑",
        len(ui_files), defect, attempt,
    )
    return {"action": "continue", "message": nudge_message(ui_files, defect)}
