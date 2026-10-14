# 功课存在性门禁设计（app-groundbreaking gate）——七步法功课文档

日期：2026-10-14
状态：反方审查前草案（审查通过前禁写子模块代码）
关联：skill:app-groundbreaking-seven-steps、skill:hermes-plugin-development
（guard-system-layering-20260905）、plugins/guards/ 套件

---

## 0. 背景：v1 推翻记录

第一版（plugins/app-groundbreaking-guard/ 独立插件+12 测试）被用户裁定推翻重做。
v1 的三个错误（本设计的反面教材）：

| # | v1 错误 | 根治条款 |
|---|---------|----------|
| E-1 | 落位顶层独立插件，违反「同层已有等价裁定源的扩展既有插件，禁新建平行护栏」 | §5 架构挂靠：进 guards/ 套件做子模块 |
| E-2 | 证据=「文档存在」弱代理，未论证代理有效性 | §1 原子拆解 C2+§6 可验问题清单 |
| E-3 | 未过反方审查就写代码（用七步法建门禁时自己违反七步法） | §7 闭环：本文档先过审查 |

## 1. 原子（第一性拆解）

本质一句话：本门禁=「功课产物的存在性机检」——把「未做功课就编码」从事后
可辩解的违规，变成写入路径上的确定性阻断。它不评审功课质量（质量=反方
审查+用户的领域），只机检产物存在。

本质约束（不可绕过的才算）：

- C1 判定确定性：只用文件存在性/会话态，禁 LLM 判内容。绕过后果=不可复现
  +judge 通道故障时静默失效（scope_gate 实测教训：judge 五键不齐恒 fail-open，
  门禁看起来装好实际零拦截）。
- C2 证据必须是落盘产物：读了 skill、口称做过功课、会话里讨论过——都不算；
  只有磁盘上的功课文档/DDL 算。绕过后果=口头合规骗过门禁，纪律回到认知层。
- C3 拦截面与 guards 家族射程对齐：write_file/patch/execute_code(写)；
  terminal 静默写由 guards.source_write 前置把守（同射程不重复建）。
  绕过后果=terminal 通道成后门。
- C4 存量不回溯：目录已有代码文件=存量应用，一律放行。绕过后果=全库 40+
  存量应用被误拦，门禁退化为全库改造阻断器。
- C5 fail 语义分离：守卫自身异常→fail-open+WARNING（纪律增强层宁放不锁死）；
  判定语义本身→fail-closed（三证据全无=拦）。两者混用=守卫 bug 变静默失效。

概念拆开（三个易混概念）：

1. **授权**（scope_gate 域）：用户允不允许做这个方向级动作。
2. **功课**（本门禁域）：动工前置产物（架构文档/数据结构）是否已存在。
3. **过程合规**（coding_standards/diff_debt/pre_write 域）：这次写得对不对。

用户授权「门禁必须上」后我直接编码=①通过②缺失③通过——现有防线全部放行的
缺口形态，正是本门禁要堵的位置。

## 2. 资产（存量盘点，2026-10-14 实测）

### 2.1 全防线判定轴清单

| 防线 | 层 | 判定轴 | 与本门禁关系 |
|------|----|--------|--------------|
| ReadThinkGate | L1 | 写前四轴证据（assistant 正文扫描+marker） | 过程合规域，不覆盖功课域 |
| guards.four_axis | L2 | 读 marker，四轴未齐拦写 | 同上（副防线） |
| guards.pre_write | L2 | 同轮先读后写 | 调查准入，不覆盖架构功课 |
| guards.duplicate_check | L2 | 新建 .py 符号级查重 | 代码查重≠功课查重 |
| guards.batch_write | L2 | execute_code 批量改已有文件 | 过程合规域 |
| guards.diff_debt | L2 | 写后 diff 欠账追缴 | 过程合规域 |
| guards.source_write | L2 | terminal 静默写源码 | C3 依赖它守 terminal，互补 |
| guards.coding_standards | L2 | AST 编码规范 | 格式域 |
| guards.tool_blacklist | L0 | 必败工具黑名单 | 不同域 |
| guards.casebook_gate | L2 | OntoX 写前查病例库（拦一次引导） | 同族不同粒度，见 2.2 |
| guards.scope_gate | L2 | 未授权新开目录树（LLM 授权语义判定） | 授权域，互补，见 2.3 |
| discipline 8 项 | L3 | no-guessing/db-safety 等执行纪律 | 不同域 |
| devil-advocate-audit | — | 重大方案定稿前须有 delegate 反方 | 覆盖七步法第 7 步（闭环），不覆盖 1-6 步 |
| skill-drift-check | — | skill 源文件 SHA 漂移检测 | 非「skill 未加载」拦截，不覆盖第 0 步 |
| finish_guard | — | 回合形态（请示/假继续） | 回合域 |

### 2.2 与 casebook_gate 的分工判据（同族两「写前功课」并存）

- casebook_gate：触发面=OntoX 路径任意写；语义=知识检索引导（拦一次即放）；
  证据行为=本会话出现过 casebook 查询。
- 本门禁：触发面=新应用（apps/<X>/ 无代码文件）首代码文件；语义=产物常驻
  （证据存在才放，非引导）；证据物=磁盘上的功课文档/DDL。
- 判据=目标生命周期阶段：既有应用迭代（查病例防复发）vs 新应用动工
  （查功课防空壳）。两者覆盖面重叠区间（OntoX 新应用首文件）两道都触发，
  first-wins 分发下消息竞争可接受（都指向正确的下一步）。

### 2.3 与 scope_gate 的链路协同

scope_gate 五条件（仓库内∧新文件∧父目录不存在∧深度≤2∧无授权）拦「未授权
新开目录树」的首文件；有授权时放行→本门禁接棒查功课。两道顺序门：
授权（用户允许）→功课（产物存在）→格式（coding_standards）→过程（diff_debt）。
每道判一个正交维度，无一重复。

### 2.4 真缺口结论

七步法第 1-6 步（原子拆解/资产盘点/数据结构/链路/架构挂靠/产品能力）的
产物存在性，在 L0-L4 全层零机器覆盖——只有第 7 步（反方审查）有
devil-advocate-audit。缺口成立，新建（进 guards 套件，非平行新插件）合规。

## 3. 数据结构（门禁的本体）

门禁无持久化表——它是纯函数式判定器，本体=**证据判据清单**（声明在子模块
顶部的显式数据结构，可枚举可测试）。三方契约（人/代码/AI 会话）由三载体
承载：判据清单（代码常量）+block 消息文本（判据的人话投影）+skill
app-groundbreaking-seven-steps（方法论全文）。

- 证据源三元组（EvidenceSpec）：
  - E1 应用内功课文档：apps/<X>/ 下 rglob("*.md") 且非 README*
  - E2 应用内数据结构：apps/<X>/ 下 *.sql 或 migrations/alembic/db/schema/ 目录
  - E3 仓根蓝图功课：docs/blueprint/*.md 文件名含应用名（归一化=小写+连字符转下划线）
- 豁免标记：内容含 `# groundbreaking-ok: ` 且理由≥8 字。
- 代码扩展集合：.py/.ts/.vue/.go/.java。
- 会话态优化（不改语义）：证据命中后 per-app 放行标记缓存到
  plugins._shared_state（casebook_gate 同源基建），批量写时免重复磁盘扫描。
- 不引入新配置文件：guards 家族零 yaml 判据先例（scope_gate 豁免扩展名=
  常量），引入声明式配置=新范式，须单独论证（本设计不夹带）。

## 4. 链路（判定流）

```
pre_tool_call(write_file|patch|execute_code)
  → _parsed_target: 工具过滤 + 路径解析 + 代码扩展过滤 + apps/<X>/ 段定位
  → 豁免: 会话态缓存命中 → 放行
  → 存量判定: apps/<X>/ 已有代码文件 → 放行并写会话态缓存
  → 证据三源 E1/E2/E3 → 任一命中 → 放行并写会话态缓存
  → block: 三段式消息(命中证据/三源三态/分档下一步) + 指路 skill
```

- execute_code 写形态识别复用 duplicate_check 基建（batch_write/scope_gate
  同款 `_execute_code_py_targets`），不重复实现。
- block 消息含豁免标记写法与「先表后码」指引。

## 5. 架构挂靠（动工三问过堂）

- 挂哪层：L2 guards 套件新子模块 `plugins/guards/groundbreaking_gate.py`。
  规则依据=「同层已有等价裁定源的扩展既有插件，禁新建平行护栏」——guards
  即 L2 裁定源，新防线必须长在套件里（v1 错误 E-1 的根治）。
- 复用哪条骨架：casebook_gate 样板（路径门控+_shared_state 会话态+
  fail-open 契约+env 开关 HERMES_GROUNDBREAKING_GATE_WARN_ONLY/_DISABLE）。
- 注册形态：guards/__init__.py 套件 register 逐子护栏 try/except（单失败
  不连坐）+plugin.yaml 版本升号。
- first-wins 分发适配：本门禁消息自带 `[groundbreaking-gate]` 标签，
  验证单独断言 handler 返回值，不依赖聚合。

## 6. 产品能力（被拦-解锁旅程）

- 谁：未来所有会话（含本 agent 自己）在 OntoX/家族仓新建应用时。
- 进：写 apps/<新>/main.py 且三证据全无 → 拦，消息列出三源三态。
- 三档出路：①apps/<X>/docs/ 或仓根 docs/blueprint/ 落功课文档（文件名含
  应用名）②先建 DDL/migrations（先表后码）③纯适配层显式豁免标记（≥8 字
  理由，可被 grep 审计滥用）。
- 离开：功课落盘后重写放行；或豁免留痕。
- 可验问题清单：①三无新应用写首代码文件被拦？②三证据任一存在放行？
  ③存量应用零打扰？④守卫异常 fail-open 不 raise？⑤豁免标记全库可审计
  （grep groundbreaking-ok）？⑥execute_code 写形态同样被拦？

## 7. 闭环（施工顺序）

1. 本设计文档 commit。
2. 独立只读反方审查子代理（挑：证据代理有效性/first-wins 竞争/存量判定
   边界如目录里只有测试文件/豁免滥用面/shared_state 并发）。
3. 审查修正回灌本文档。
4. 子模块代码+测试（含 fresh 子进程真机探针：discover_plugins+
   list_plugins enabled+探针行为三件）。
5. preflight+rules_selfcheck 全绿。
6. 双仓收尾核对（源码仓+运行副本 ~/.hermes/hermes-agent，零未推）。

## 附：证据索引

- 分层规则与 first-wins 分发：skill:hermes-plugin-development/
  references/guard-system-layering-20260905.md [实测文档]
- guards 套件 12 子模块判定轴：plugins/guards/ 各子模块 docstring
  （2026-10-14 逐个读取）[实测]
- scope_gate 五条件：plugins/guards/scope_gate.py:1-48 [实测]
- casebook_gate 会话态+fail-open 契约：plugins/guards/casebook_gate.py:1-33 [实测]
- devil-advocate-audit pre_llm_call 语义判定：plugins/devil-advocate-audit/
  __init__.py:1-10 [实测]
- skill-drift-check=SHA 漂移非加载拦截：plugins/skill-drift-check/
  __init__.py:1-6 [实测]
- v1 推翻：用户 2026-10-14 裁定「推翻重做」；v1 文件已 rm 零残留
  （git status 核验）[实测]
