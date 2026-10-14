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

两道正交门，时序上互不依赖触发（审查 R7 修正：scope_gate 豁免 .md 等文档
扩展，先保存功课文档再写码的常见时序里 scope_gate 根本不拦，本门禁是
独立首道而非「接棒」）：scope_gate 拦「未授权新开目录树」的代码首文件，
本门禁拦「已授权/豁免路径但无功课产物」的代码首文件。语义链=授权（用户
允许）→功课（产物存在）→格式（coding_standards）→过程（diff_debt），
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
  - E1 应用内功课文档：apps/<X>/ 下 rglob("*.md") 且非 README*，且**最小内容
    判据**（审查 R1）：非空有效行数≥30 且标题行含**必备「数据库」**（用户指令
    2026-10-14「确定数据结构的时候要先确定数据库」——缺库归属裁决的功课文档
    不构成证据）+「数据结构」「链路」「架构」任一（str 方法匹配标题行，零正则
    零 LLM）。空文档/一行 touch 不构成证据。
  - E2 应用内数据结构：apps/<X>/ 下 *.sql 或 migrations/alembic/db/schema/ 目录
  - E3 仓根蓝图功课：docs/blueprint/*.md 文件名 token 边界匹配（审查 R4：
    文件名按 -_ 切分 token，与应用名归一化 token **整 token 相等**才命中，
    禁子串 containment——实测 'aml' 子串误命中 3 份无关文档；短结构词
    api/web/backend/data 入噪音词表，命中降级提示档要求 E1/E2 补强），
    且满足与 E1 相同的最小内容判据。
- 豁免载体（审查 R2）：豁免**只认仓根固定清单**
  docs/blueprint/GROUNDBREAKING_EXEMPT.md（逐应用一行=应用名+理由+日期），
  **禁在被写目标文件内生效**（否则一行 8 字注释即自豁免=后门）；门禁读该
  清单匹配应用名，命中放行并记 WARNING 留痕。
- 代码扩展集合：.py/.ts/.vue/.go/.java。存量扫描集合=触发集合=该五类扩展
  的任意文件（**含 tests/*.py 与 __init__.py**，rglob 全子树，两集合强制相等，
  审查 R5——否则「先写测试再写主码」路径判定歧义）。
- 会话态优化（审查 R6 修正后语义）：per-app 缓存存**命中证据的绝对路径**
  （非布尔值），命中时 os.path.exists 复验（单次 stat），失效即清缓存重走
  证据三源——证据文件被删后缓存不续命。
- 不引入新配置文件：guards 家族零 yaml **判据**先例（审查 R8 措辞限定：
  判据一律代码常量；路径类配置有 casebook_gate 读 config.yaml 的先例，
  本门禁路径键同此模式）。

## 4. 链路（判定流）

```
pre_tool_call(write_file|patch|execute_code)
  → _parsed_target: 工具过滤 + 路径解析 + 代码扩展过滤 + apps/<X>/ 段定位
  → 豁免: 会话态缓存命中(证据路径 exists 复验) → 放行
  → 豁免清单: GROUNDBREAKING_EXEMPT.md 含应用名 → 放行+WARNING
  → 存量判定: apps/<X>/ 已有代码文件(五扩展含tests/__init__) → 放行并写缓存
  → 证据三源 E1/E2/E3 → 任一命中 → 放行并写缓存(存证据绝对路径)
  → block: 三段式消息(命中证据/三源三态/分档下一步) + 指路 skill
```

- execute_code 写形态识别：duplicate_check 基建扩展参数化变体
  `_execute_code_write_targets(code, exts)`（审查 R3——既有
  `_execute_code_py_targets` 硬过滤 .py，非 Python 扩展经 execute_code 写入
  时不可见=射程缺口；变体满足单一事实源纪律，.py 原路径逐字不变供既有
  调用方，本门禁传五扩展全集）。
- block 消息含豁免清单写法（非行内标记）与「先表后码」指引。

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
- 三档出路：①apps/<X>/docs/ 或仓根 docs/blueprint/ 保存功课文档（token
  匹配应用名+最小内容判据）②先建 DDL/migrations（先表后码）③纯适配层
  在仓根 GROUNDBREAKING_EXEMPT.md 登记（应用名+理由+日期，集中可审计）。
- 离开：功课保存后重写放行；或豁免清单留痕。
- 可验问题清单：①三无新应用写首代码文件被拦？②三证据任一存在放行？
  ③存量应用零打扰？④守卫异常 fail-open 不 raise？⑤豁免集中于单一清单
  可审计？⑥execute_code 写形态（含 .ts 等非 Python 扩展）同样被拦？
  ⑦空文档/短文档（<30 行或无必备标题）不构成证据仍拦？⑧目录仅有
  tests/*.py 时写主码仍拦？⑨同会话删除证据文件后缓存失效重拦？

## 7. 闭环（施工顺序）

1. 本设计文档 commit。
2. 独立只读反方审查子代理。**已回（deleg_13eaea95，裁定=有条件通过）**：
   8 项修正（R1 空文档击穿/R2 自豁免后门/R3 execute_code 射程缺口三项高；
   R4 子串误命中实测/R5 集合边界两项中；R6 缓存复验/R7 scope_gate 关系
   改述/R8 yaml 措辞三项低），全部回灌本文档 §2.3/§3/§4/§6。
3. 审查修正回灌本文档（本条已执行）。
4. 子模块代码+测试（含 fresh 子进程真机探针：discover_plugins+
   list_plugins enabled+探针行为三件）。**已执行（commit 2cdd810ec5）**：
   plugins/guards/groundbreaking_gate.py（334 行）+ 套件注册（plugin.yaml
   2.6.0+__init__._SUB_GUARDS）+ duplicate_check._execute_code_write_targets
   参数化变体（.py 专用版薄委托）+ tests/plugins/test_guards_groundbreaking_
   gate.py 16 用例（覆盖 §6 可验 9 问+R1-R8）94 passed（含套件 6 既有文件
   回归）；真机探针：_SUB_GUARDS 11 子护栏含本模块、ontox 仓虚构新应用
   block、存量 aml_ng 放行。施工中修三处实现缺陷：_e2 对不存在目录
   iterdir 抛错被 fail-open 吞=静默放行；E1/E3 缓存载体由目录改具体文档
   路径（删证据后缓存失效重拦）；open().write() 链双 Call 节点重复上报
   去重。R5 审查建议内部矛盾裁定：两集合强制相等（含 tests/__init__）
   优先于其负例「仅 tests 仍拦」——门禁拦首个代码文件本身，绕过链路
   不成立。
5. preflight+rules_selfcheck 全绿。
6. 收尾核对：源码仓 origin 零未推（fork 仓唯一同步通道=origin，
   cnb remote 历史异源禁用——见 memory 1014 条目）。

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
