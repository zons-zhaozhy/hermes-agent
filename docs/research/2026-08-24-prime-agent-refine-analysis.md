# prime-agent Continual Harness / refine 机制调研与 Hermes 借鉴评估

日期：2026-08-24
调研对象：https://github.com/PrimeIntellect-ai/prime-agent （MIT，~17.5k star）
触发来源：公众号文 https://mp.weixin.qq.com/s/ri36KwRf9xbLpOxxE-Y9nA
证据方式：全部结论基于 GitHub raw 源码与官方 docs 实测抓取（zread 镜像 + raw.githubusercontent.com），非公众号转述。

## 1. prime-agent 是什么（实测源码）

monorepo（TypeScript），核心包 packages/coding-agent。两大抽象：

1. **RLM（递归语言模型）**：一等工具是常驻 IPython kernel——文件操作/shell/子代理/压缩/目标全部是 kernel 里的 Python 函数，而非 20 个并列 schema 工具。
2. **Continual Harness（持续挽具）**：把 supplemental prompt notes / memories / skill 描述 / subagent 规格做成持久可编辑状态，通过 `/refine` 做"小步、有证据"的自我改进；基础 system prompt 不可变；每次 refine 记录快照，支持按 ID 回滚。

## 2. refine 机制源码级拆解（src/core/refinement/refinement.ts，952 行）

### 数据结构
- `HarnessState`：`{schema:1, entries:{prompt|memory|skill|subagent: {id→HarnessEntry}}, refinements:[RefinementEvent]}`
- `HarnessEntry`：id/kind/title/content/path/scope/created_at/updated_at/version；skill 条目强制带 `reference`（Python import+callable）与 `arguments` 契约。
- 存储：`<agentDir>/harness/harness_state.json`（global）+ 会话产物目录同名文件（local）；原子写（tmp+rename，保留原 mode）；历史 `refinements.jsonl` append-only，坏行跳过不炸回滚。

### 双作用域（设计精髓）
- **local（默认）**：只写当前会话 store，存"本任务进度/临时阻塞/本轮协作注记"；refine 时 global 条目是只读上下文，禁止改删。
- **global（显式请求）**：只准写跨会话稳定教训/持久偏好/可复用 skill/subagent，且项目级教训必须 title/path 显式含项目名才准入 global。
- system prompt 里合并渲染：`mergeHarnessStates`，冲突时 local 条目以 `local:<id>` 前缀并列而非覆盖。

### refine 流程
1. `planRefinement`：独立 LLM 调用（非 reasoning 模式，max 32k 输出），输入=当前 harness 概览（每类最多 6 条、每条 content 截 180 字符的**路由摘要**，明确声明"是提示不是全文"）+ refine 历史（截 5 条）+ 轨迹文本（截 40k）+ 作用域指令，产出 JSON edits（create/update/delete，逐条带 reason）。
2. `applyRefinementProposal`：逐条应用，记录 before/after 快照，产出 `RefinementResult`（含 summary/rationale/expectedOutcome/appliedEdits）。
3. 回滚：`/refine rollback <refinement-id>`，按历史记录恢复 before 状态。
4. auto-refine 门控：turn 间隔/compaction 触发时先过 `reviewAutoRefine`（4k 输出小模型判定 shouldRefine），拒绝一次性噪声/无证据假设——防 refine 污染。

### prompt 渲染预算
`formatHarnessStateForPrompt`：每类条目 6 条 × 180 字符 + 历史 5 条的紧凑概览注入 system prompt；超过显示 "+N more"。**prompt 注入的是路由提示，全文按需读**——控 token 成本。

## 3. 与 Hermes 现有能力对照

| prime-agent 机制 | Hermes 现状 | 差距 |
|---|---|---|
| 后台会话/daemon + attach | process/cronjob/session 持久化 | 已覆盖 |
| 子代理编排 | delegate_task + steer | 已覆盖 |
| skill 体系 | skill_manage + 243 skills，比它强（有 category/absorbed_into 治理） | 已覆盖 |
| memory | MEMORY.md 手工维护 | 已覆盖但**无快照/回滚** |
| refine 小步改进+回滚 | 无 | **真实差距** |
| 双作用域（session/global）改进准入 | 无显式机制（靠自律） | 真实差距 |
| auto-refine 证据门控 | outcome-feedback-flywheel 近似但无门控判定 | 部分差距 |
| 常驻 IPython kernel | execute_code 每次全新解释器 | 差距，但 Hermes AGENTS.md 明确"core 是窄腰"，按 footprint ladder 不宜进 core |
| agent 间直接消息 | 子代理只能经主代理汇总 | 差距，成本高 |

## 4. 值得落地的建议（按性价比排序）

### 建议 1（强烈推荐）：memory/skill 写操作快照+回滚
现状痛点：MEMORY.md 和 skill 编辑是原地覆盖，写坏了只能靠 git（memory 不在 git 里则无法回滚）。
方案：给 memory/skill 写路径加 append-only 历史 `~/.hermes/refinements.jsonl`（记录 before/after/id/理由），配一个 skill 或 CLI 子命令 `rollback`。prime-agent 证明这套 jsonl+跳坏行的实现只约 300 行。
约束：写操作走已有工具（skill_manage/memory 工具内部），不新增 core 工具，符合 footprint ladder。

### 建议 2（推荐）：改进准入双作用域纪律
把 prime-agent 的准入规则写进 memory/skill 写入工具的提示层：默认会话级草稿，只有"跨会话稳定教训/持久偏好/显式项目限定"才准入全局。Hermes 的 MEMORY.md 已经接近满（7961/8000），正好需要准入闸门控制增长。

### 建议 3（可选）：prompt 注入用"路由摘要"而非全文
formatHarnessStateForPrompt 的做法（每条截 180 字符+明确声明是提示不是全文）可直接用于 skills 列表膨胀后的 system prompt 瘦身——当前 skill 索引已在系统提示里占很大篇幅。

### 不建议跟进
- 常驻 IPython kernel：与 Hermes 窄腰原则冲突，且 execute_code+terminal 组合已覆盖大部分场景。
- agent 间直接消息：delegate/steer 已够用，改造大。
- ARC-AGI-3 成绩：特定 harness×特定模型的调参结果，工程不可迁移。

## 5. 结论

prime-agent 的核心创新不在"自我改进"概念（Hermes 已有 memory+skill+flywheel），而在**工程化纪律**：小步 edits、before/after 快照、按 ID 回滚、双作用域准入、auto-refine 证据门控、prompt 注入路由摘要。这五条里前三条（快照/回滚/准入）是 Hermes 当前真实缺口且实现成本低（≈一个 skill + jsonl 文件），建议落地。

---
参考源码：
- packages/coding-agent/src/core/refinement/refinement.ts（952 行，核心）
- packages/coding-agent/docs/long-running-agents.md（refine.run 契约）
- packages/coding-agent/docs/skills.md（Python-backed skills）
