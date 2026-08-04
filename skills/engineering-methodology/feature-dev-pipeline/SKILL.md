---
name: feature-dev-pipeline
description: "需求开发全流程——7阶段流水线 skill，强制阶段顺序+退出标准。触发：开发新需求/修 bug/重构。"
tags: [engineering, pipeline, development-workflow]
metadata:
  hermes:
    category: engineering-methodology
---

# 需求开发流水线

> **这不是参考文档，是执行指令。** 加载此 skill 后，必须按阶段顺序执行。
> 每个阶段有明确的退出标准——不满足退出标准，不进入下一阶段。
> 每次在回复开头声明当前阶段：「📋 阶段 N/7: <阶段名>」
>
> **阶段解锁**：进入新阶段时必须更新 phase 状态文件（pipeline-guard plugin 代码级阻断，仅在流水线激活时生效）：
> ```python
> # 阶段 1 开始时激活流水线（创建 phase 文件 = 激活）
> python3 -c "import json,pathlib; p=pathlib.Path.home()/'.hermes'/'cache'/'pipeline_phase_<session>.json'; p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps({'phase':1,'active':True}))"
> # 后续阶段转换只需改 phase 数字
> python3 -c "import json,pathlib; p=pathlib.Path.home()/'.hermes'/'cache'/'pipeline_phase_<session>.json'; p.write_text(json.dumps({'phase':4,'active':True}))"
> # 流水线结束（阶段 7）后关闭
> python3 -c "import json,pathlib; p=pathlib.Path.home()/'.hermes'/'cache'/'pipeline_phase_<session>.json'; d=json.loads(p.read_text()); d['active']=False; p.write_text(json.dumps(d))"
> ```
> 没有 phase 文件 = 流水线未激活 = 正常工作模式（不拦截任何操作）。

## 触发条件

- 用户说"开发"/"实现"/"修 bug"/"重构"/"加功能"
- 用户给出一个需求描述让你实现
- 你自己判断这是一个需要多步骤的开发任务

## 七阶段流水线

### 阶段 1: 需求理解

**做什么**：搞清楚要做什么、不做什么、入口在哪。

**执行步骤**：
1. 用自己的话复述需求（一句话）
2. 明确边界：做什么 + 不做什么
3. 如果是多子需求，用 subtask_ledger 建台账：
   `python scripts/subtask_ledger.py init "<feature-name>"` + `add M1 "<描述>"`

**退出标准**：
- [ ] 能一句话说清做什么
- [ ] 能说清不做什么
- [ ] 知道改哪些模块（文件级别）

**禁止**：此阶段不许写代码。

### 阶段 2: 定位

**做什么**：找到改动点，建立调用链。

**执行步骤**：
1. 先看目录结构（`search_files target='files'`），不读文件内容
2. 关键词搜索定位候选文件（`search_files target='content'`）
3. 读候选文件确认改动点（`read_file`）
4. 追踪调用链：谁调用它、它调用谁

**退出标准**：
- [ ] 知道改哪个文件的哪个函数
- [ ] 知道上下游调用方
- [ ] 能列出影响范围

**禁止**：此阶段不许写代码。

### 阶段 3: 方案设计

**做什么**：确定怎么改，考虑替代方案。

**执行步骤**：
1. 列出 2-3 种可行方案
2. 每种方案说出优劣
3. 选择一种并说明理由
4. 确认不变式——哪些东西不能动

**退出标准**：
- [ ] 有方案选择和理由
- [ ] 有不变式清单
- [ ] 有验证计划

**禁止**：此阶段不许写代码。

### 阶段 4: 实现

**做什么**：写代码。

**执行步骤**：
1. 先读完整方法再改——先看后写
2. 模仿已有模式——搜索同类实现做参照
3. 逐文件改，改完一个验证一个
4. 多子需求每个完成后更新台账：
   `python scripts/subtask_ledger.py done M1 <commit-hash>`

**退出标准**：
- [ ] 代码改动完成
- [ ] 编译/语法检查通过

### 阶段 5: 验证

**做什么**：证明代码真的能用。

**执行步骤**：
1. 跑相关测试（`scripts/run_tests.sh`）
2. 跑 linter（`write_file`/`patch` 的自动 lint）
3. 如果改了 skill 文件，跑红线审计：
   `python scripts/skill_redline_scanner.py --skill <skill-name>`
4. A/B/C 分类处理失败：
   - A 真问题（代码 bug）→ 回阶段 4 修
   - B 环境问题（路径/配置不对）→ 修环境
   - C 时序问题（重试 ≤2 次）

**退出标准**：
- [ ] 测试通过
- [ ] 无新增 lint 错误

### 阶段 6: 沉淀

**做什么**：记录改了什么、为什么这么改。

**执行步骤**：
1. 更新或创建 TECH_SPEC（按 tech-spec skill 模板）
2. 如果改了 skill 引用的源文件，跑 SHA 漂移检查：
   `python scripts/skill_sha_drift.py --update`
3. 检查文档一致性

**退出标准**：
- [ ] TECH_SPEC 的 §2（涉及文件）、§7（演进事件）、§9（产物清单）已更新
- [ ] 受影响文档已同步

### 阶段 7: 提交

**做什么**：commit + push。

**执行步骤**：
1. 写 commit message（类型 + 影响范围 + 关键改动）
2. git add 只加本次改动的文件
3. git commit（守卫会自动检查）
4. 守卫阻断 → 修根因，不跳守卫
5. git push 到所有 remote

**退出标准**：
- [ ] commit 成功
- [ ] push 成功

## 跨会话接力

新会话恢复时：
1. 读 TECH_SPEC §0→§1→§3→§5→§7
2. 如果有台账，跑 `python scripts/subtask_ledger.py resume`
3. 确认当前进度后从对应阶段继续

## 红线

- RL-1: 禁止跳阶段。后一阶段的输入=前一阶段的产出。
- RL-2: 阶段 1-3 禁止写代码。
- RL-3: 阶段 5 验证失败超过 3 轮→停下报告。
- RL-4: 不可逆操作（commit/push/delete）前确认。