# 三篇外部文章 vs Hermes 本体对照分析

日期：2026-09-12
来源：微信公众号三篇（智能体折腾日记《我给 Hermes Agent 装了3个器官》；钰见增长《三零五带七抓》；硅星人Pro《RabbitOS 3 体验》）
性质：外部思想对照本体代码的差距审计。所有本体结论均标注代码位置，[实测]=当轮工具输出核对。

## 一、器官一 WORKFLOW.md（行为定义收进仓库 + 每轮注入）vs Hermes 上下文文件链

本体现状 [实测]：
- `agent/coding_context.py:37` `_CONTEXT_FILES = ("AGENTS.md", "CLAUDE.md", ".cursorrules")`，`.hermes.md` 优先级更高（`hermes_cli/config_defaults.py:464` 对 SOUL.md/AGENTS.md/CLAUDE.md/.hermes.md 自动加载设硬上限）
- 上下文文件在会话启动时一次性注入 system prompt，加载前经提示注入安全扫描（`hermes_cli/tips.py:193`）
- 插件可注册 `pre_llm_call` hook，其返回内容由 `agent/turn_context.py:578 _collect_pre_llm_call_context` 追加到**当前用户消息**（临时上下文，不动缓存前缀）（`CONTRIBUTING.md:97`、`docs/observability/README.md:57`）

对照结论：
- "行为定义进仓库随 git 管理" —— Hermes 已具备（.hermes.md/AGENTS.md 即此物），无差距
- "每次调用前由 hook 自动重读注入" —— Hermes **有机制无用法**：pre_llm_call 通道存在但没有任何插件用它做"纪律规则每轮重注入"。价值证据：`docs/investigations/2026-08-28-patch-first-root-cause.md` [实测] 记录了长会话+上下文压缩后 memory 条目「patch优先」约束力衰减为可选项，patch-first 插件正是靠 pre_llm_call 60 秒窗口注入才挽回约束力
- 差距定性：规则衰减是真实发生过的缺陷，重注入是已验证有效的修法，且通道现成、缓存安全（追加到当前用户消息，不改历史前缀）

## 二、器官二 Per-Task Workspace vs Hermes delegate 隔离

本体现状 [实测]：
- `tools/delegate_tool.py` 已有 `_get_worktree_isolation` / `_resolve_workspace_hint`；`tools/delegate_tool_child_run.py:590 seed_workspace` 为子代理播种独立 cwd，可选 git worktree 隔离，失败时静默降级共享 workspace（`delegate_tool_child_run.py:315`）
- `delegation.max_concurrent_children` 默认 3——与文章的"最多并行 3 个子任务"同一数字
- 多会话层面另有 SESSION_BOARD.md 认领约定（多会话并发协作纪律）

对照结论：Hermes 领先。文章的目录式隔离（.hermes/workspaces/task-xxx）是 worktree 隔离的弱化版。无差距，不参考。

## 三、器官三 Harness Self-Check（项目基建就绪度门禁）vs Hermes

本体现状 [实测]：
- `agent/read_think_gate.py` 拦"未调查就动手"（写前调查门禁）——管的是 agent 行为，不管项目基建
- `scripts/run_tests.sh` 探测 .venv/venv 归属——基建检查的最小片段
- 无任何"长自动任务开跑前先给项目 CI/测试/linter/依赖锁定/密钥保护打就绪分"的门禁

对照结论：真实差距。Hermes 的门禁全部对着 agent（ReadThink/finish_guard/编码守卫），没有一条对着"这个仓库配不配被全自动改"。文章的 10 维度加权评分（<60% NOT_READY）是可借鉴形态，落点应为 skill（`hermes harness-check` 类）或插件 pre_llm_call 软提示，零核心改动。

## 四、RabbitOS 3 对照

| RabbitOS 设计 | Hermes 对应 | 定性 |
|---|---|---|
| 统一输入流，取消"新建对话" | 会话制 + memory + session_search(FTS5) | 不参考。Rabbit 方案与提示缓存经济学正面冲突（本仓最高政策：per-conversation prompt caching is sacred）；Hermes 用跨会话检索解决同一问题 |
| 设备即 Node，一行命令接入 | tools/environments/（local/ssh/docker/modal/daytona/singularity）+ gateway ~20 平台 + acp_adapter | 能力已覆盖，Rabbit 赢在接入 UX（一行命令）。低优先级参考 |
| Skill 贴 GitHub 链接即装 + 装完给审计总结 | skillhub-install skill + curator 管理 | 半差距：装已有，**装完无安全审计总结**（上下文文件有注入扫描，skill 安装链无对应报告）。可补 |
| Auto Teach 任务成功即泛化为 Skill | 任务后主动提议存 skill + behavioral-regression-verification | 已覆盖 |
| 花钱/删除动作停下确认 | gateway /approve /deny 审批链 | 已覆盖 |
| Generative UI | desktop 插件面板 | 方向不同，不参考 |

RabbitOS 核心论点"让用户少理解十个概念"与 Hermes 脚手架梯（Footprint Ladder）同源：能力加在边缘不加核心。

## 五、三零五带七抓对照

管理纪律非代码资产。"凡检查必带结果"与本仓 delivery-evidence-discipline（称[实测]必须当轮贴原始输出）是同一纪律。已内化于用户纪律体系，无工程动作。

## 结论：有价值的参考按优先级

1. **规则每轮重注入（WORKFLOW.md 模式）**——最高价值。通道现成（pre_llm_call 插件），解决已被实测证实的"长会话纪律衰减"缺陷，缓存安全，零核心改动。形态：插件读仓库内规则文件摘要，每轮追加注入当前用户消息。注意注入体量要小（每轮都付 token 成本）
2. **Skill 安装后审计总结**——中价值。对齐 Rabbit"装完给审计结果"，复用上下文文件的注入扫描思路扩到 skill 安装链
3. **项目就绪度自检（harness self-check）**——中价值。长自动任务前的基建评分门禁，落点 skill
4. 不参考：取消会话边界（毁缓存）、目录式 workspace（弱于现有 worktree 隔离）、Generative UI
