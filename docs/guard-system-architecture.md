# 护栏体系分层架构（2026-09-05 梳理）

## 数据基础 [实测]
- outcomes.db: patch 被四轴拦 1630 次、execute_code 被副防线误拦 474 次、
  1175 个 turn 被拦≥3 次（打地鼠签名）。
- 护栏类插件约 30 个,各自独立串联,first-wins 投票。

## 分层模型（收拢后）

```
L4 审计层  OutcomeAnalysis / outcomes.db        —— 事后度量,不改行为
L3 执行纪律 no-guessing / db-safety / error-discipline / curl-safety
            —— 管"怎么跑命令"(不瞎猜/不裸奔/不吞错)
L2 写入质量 coding-standards-guard / four-axis / PreWriteGuard /
            duplicate-check / patch-first / source-code-write-guard
            —— 管"写什么"(规范/证据/查重/唯一通道)
L1 调查准入 ReadThinkGate(主)                    —— 管"动手前查没查"
L0 机制黑名单 tool-blacklist                     —— 管"必败工具根本不可调"
```

## 裁定规则（消除冲突）
1. **每层只有一个裁定源**：L1 主闸门=ReadThinkGate,副防线插件
   (four-axis-guard)必须与其 `_FOUR_AXIS_REQUIRED_TOOLS` 逐字对齐,
   不得私自扩面（已修:execute_code 对齐,消除 474 次误拦）。
2. **一次暴露全部违规**：pre_tool_call 聚合总线（hermes_cli/plugins.py
   `_get_pre_tool_call_directive_details`）收集所有 block verdict 合并成
   一条消息——禁 first-wins 打地鼠（已上线,含护栏聚合标头）。
3. **写前预检优先于事后拦截**：
   `python3 plugins/guards/preflight.py <file>` 提交前
   一次性全量自查；拦截消息指路该命令。
4. **必败工具进 L0 而非 memory 提醒**：vision_analyze→glm_vision、
   codegraph 系非 hermes 仓→gitnexus、memory 批量 operations→逐条单 op。
5. **新增护栏前先查层内重复**：同层已有等价裁定源的,扩展既有插件,
   禁新建平行护栏（防 30→50 个插件继续膨胀）。

## 已知存量债务（未修,挂账）
- hermes_cli/plugins.py 历史违规约 40 处(R001/R008/R013/R021/R022),
  属 god-file 存量,修它们=独立任务,按层归 L2 规范层处理。
- 四轴闸门 marker 10 分钟时效 vs 长 turn 中途失效的时序问题待观察。

## 收拢执行记录（0905,全部实测）
- patch-first 已禁用并表: source_code_write_guard 为严格超集
  (sed 4 变体全拦/只读 sed -n 放行,等价性验证过);
  config.yaml plugins.enabled 43 项、disabled=[patch-first]。
- PreWriteGuard vs ReadThinkGate write-target 检查: 语义重复但状态源
  不同(session_state vs gate._files_read)——暂共存(PreWriteGuard 覆盖
  gate 未激活的 turn),待观察误拦率后再并。
- 聚合总线端到端实测: 真实插件环境 write_file 违规一次同时输出
  CodingStandardsGuard+四轴闸门 两条 verdict。
