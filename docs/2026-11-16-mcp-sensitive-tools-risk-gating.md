# MCP 工具敏感读分级（sensitive_tools）——设计四件套

日期：2026-11-16 ｜ 状态：已批准实施（用户拍板「按 MAF 实测验证过的四个机制对齐」）

## 背景：Microsoft Agent Framework 1.26 对齐盘点

MAF 1.26（2026-11-07 发布）核心变化：MCP 发现审批支持 callable 策略函数，
`Callable[[Tool|Prompt], Literal["always_require","never_require"]]`，逐工具评估、
目录刷新后结论重算。本机实测（Python 3.12.14 venv，agent-framework 1.26.0 + mcp 2.3.0）：
- 三工具场景（get_time/delete_file/read_salary）按策略分别打上
  never_require/always_require/always_require 标记，属真。
- 再次 load_tools() 策略函数对全部工具重评——刷新重算属真。
- 构造器默认 request_timeout=30 / max_discovery_pages=100 / max_discovery_items=1000，与发布说明一致。
- 坑：AF 自建 stdio 子进程通道在 macOS 本环境死锁；须注入自建 mcp Client 才通（根因未查证）。

四机制 × Hermes 现状（2026-11-16 源码调查）：

| MAF 机制 | Hermes 现状 | 结论 |
|---|---|---|
| 1 工具风险分级 | server 级 trust 两档 + 工具级 readOnlyHint 布尔（发现期记录，mcp_tool_registration.py:89） | 缺「只读但敏感」维度 |
| 2 三态审批闸 | 已有：pre_tool_call approve/block（plugins.py:1181）+ `_trust_gate_check` fail-closed + session/always grant（mcp_tool_handlers.py:190） | 已达标，零改动 |
| 3 刷新重算 | 已有：发现期重录 hints；重连后 hint 翻转即拒并重问（_refuse_unconsented_write） | 已达标，零改动 |
| 4 Override 留痕 | 部分有：always 落 config command_allowlist；tripwire 只记拒；决策时刻无统一审计行 | 补一条 _persist_choice 审计日志 |

## 敞口（根因）

`tools/mcp_tool_handlers.py` `_trust_gate_check`：`gated = trust==untrusted and not
read_only`——untrusted 服务器上 readOnlyHint=True 的工具直放。MAF 生态的核心教训
「只读≠无害」（read_salary 只读但涉敏感数据）在 Hermes 当前无覆盖。

## goal

untrusted MCP 服务器上「只读但敏感」的工具不再无审批直放：新增
`mcp_servers.<name>.sensitive_tools` 声明键（fork 专属，对称于既有 read_only_tools，
先例 commit 741bad0e6b），敏感工具即使 readOnlyHint=True 也走既有审批门。
成功判据：新增单测（敏感读被拦→批准放行；普通读直放；注册期校验与未知名告警）全绿，
存量 trust gating 测试全绿。

## architecture

发现期 `_record_tool_trust_metadata` 记录 sensitive 集合到新账本
`_tool_sensitive_sets`（与 read_only hints 同键、同刷新语义，重连重录即刷新重算）→
调用期门条件扩为 `not read_only or sensitive` → 复用既有 `_ask_for_call`/grant/
allowlist/fail-closed 全链路，无新审批机制。
`_persist_choice`（session/always 持久化时刻）补 INFO 审计行：choice/keys/session。

## out_of_scope

- per-tool 多档风险枚举（read_only × sensitive 两布尔覆盖当前需求；镀金不做）
- 独立审计账本表（先日志行；always 决策本就持久化在 config 可查）
- 三态闸与刷新重算（已存在）
- connectors 托管工具（connectors__*）分级——另行立项

## 验证

`cd /Users/stan/code/ai/github/fork/hermes-agent && python3 -m pytest tests/tools/test_mcp_trust_gating.py -q`
（仓库无 scripts/run_tests.sh 包装器时裸 pytest 合法）
