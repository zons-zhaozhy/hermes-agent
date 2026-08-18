# 护栏速度影响审计 — 2026-08-18 09:15

触发：用户「护栏是为了安全和规范、不是为了刹车卡速度！审查所有护栏行为是否有严重影响速度的」。

## 范围

config.yaml plugins.enabled 22 个插件 + read_think_gate（核心闸门）+ agent.post_response_hooks。

## 结论分级

### 🔴 P0 已修：read_think_gate LLM 分类器配置失效（真·速度杀手）

- 症状：config `read_think_gate.use_llm_classifier: false`，但 agent.log 两天 110 次
  `LLM classified complexity=...`（08-17 44 次 / 08-18 66 次，含本会话 09:15:36）。
- 根因：read_think_gate.py reset_for_turn else 分支调 `detect_complexity()`，
  其内部无条件优先 `_classify_via_llm()`——同步 auxiliary LLM 调用（timeout=10s，
  glm-5.3/5.2），阻塞在每 turn 启动路径上（complexity_adaptive=true 时每 turn 一次）。
  配置开关是装饰品。
- 修复：else 分支直调 `_fallback_detect()`（关键词路径）。
  agent/read_think_gate.py:704-716；回归测试
  tests/agent/test_read_think_gate_classifier_switch.py（5 项，全绿 61/61）。
- 收益：每 turn 省一次同步 LLM round-trip（实测单次 2-10s 量级，上限 10s timeout）。

### 🟡 P1 设计代价（用户自设规则，非 bug，但可感知）

1. **四轴 marker TTL 10 分钟**：`_MARKER_MAX_AGE_SECONDS=600`（用户 SOUL.md 自设）。
   长 turn（>10min）中途写文件会被再次拦截要求重出四轴。实测 08-18 09:00 API call
   #695 深处仍见 `four-axis guard: blocking patch`。每轮拦截 = 模型重写证据 + 再一轮
   10-30s API 调用。日志共 77 次 four-axis 拦截 + 55 次 gate 拦截。
2. **quality_audit.jsonl 无界增长 + 每 turn 全量读**：pre_llm_call 每 turn
   `f.readlines()` 读整文件（现 11MB / 7392 行）倒序找 session 匹配 ≈ 数十 ms/turn。
   文件只增不减。建议：tail 读取或按天轮转。audit 本体在 daemon 线程（60s timeout），
   不在回合路径上——OK。

### 🟢 P2 已核实非慢源（排除项）

| 护栏 | 机制 | 实测 |
|---|---|---|
| coding-standards-guard | ast.parse 写入内容 | ms 级，无 subprocess |
| duplicate-check | git ls-files(10s,cached)+rg(8s)，仅新建 .py 触发 | 首 cwd 一次 ~1s，后缓存 |
| indexer-sync | find 发现(~1s)+后台 debounce 线程同步 | 0 次同步实际触发；启动一次性 ~1s |
| pre-response-review | sqlite 按 session_id 查 messages（51.7 万行） | idx_messages_session 索引命中，ms 级 |
| blame-shield / sandwich-inject / negative-conclusion-guard | 纯字符串/内存 | ns-ms |
| outcome-collector / adaptive-reasoning / post_response_hooks 三件套 | 内存 dict / regex-only | 零 LLM 零 subprocess |
| quality-auditor 审计本体 | daemon 线程异步 | 不在回合路径 |

### 附带发现（非性能）

- pre-response-review 硬编码 `os.path.expanduser("~/.hermes")`（该插件 __init__.py:24）
  ——profile 隔离 bug（PR #3575 同类），应改 get_hermes_home()。
- errors.log 会话 aea6aa 每 API 调用重复 `Pre-call sanitizer: healed 8-10 empty
  messages`——transcript 早前中毒未愈，每次调用自愈内存修补（CPU 可忽略，但值得根治）。

## 验证

scripts/run_tests.sh 三文件 61/61 绿（含新回归 5 项）。
