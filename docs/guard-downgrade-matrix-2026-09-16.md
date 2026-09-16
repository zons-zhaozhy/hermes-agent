"""评测矩阵：文本层规矩 → 动作层 guard 下沉评估（0916）。

逐条评估 .hermes-rules.md 全部规矩：现有 guard 覆盖情况、可下沉性判定、
动作清单。这是「逐条评估下沉」的结论账本，防重复造轮子（每条先查重）。
"""

## 评估结论总表

| # | 规矩（.hermes-rules.md） | 现有覆盖 | 判定 | 动作 |
|---|---|---|---|---|
| 1 | 改文件一律 patch/write_file 禁 heredoc | patch-first 插件 | ✅已动作层 | 无 |
| 2 | 写前必读目标文件 | guards/pre_write.py | ✅已动作层 | 无 |
| 3 | 同文件 patch 连败2次改 write_file | 无 | ❌不可下沉 | 文本层 |
| 4 | 新函数带 Contract/assert | scientific-programming-guard | ✅已动作层 | 无 |
| 5 | 新建文件前全库查重 | guards/duplicate_check.py | ✅已动作层 | 无 |
| 6 | 错误信息即答案/读堆栈 | discipline/error_discipline.py | ✅已动作层 | 无 |
| 7 | 报错后禁原样重试/2次换策略 | discipline/no_guessing.py R2 | ✅已动作层 | 无 |
| 8 | 服务名 --list 核验 | discipline/no_guessing.py R3 | ✅已动作工具 | 无 |
| 9 | 禁穷举试探 | discipline/no_guessing.py | ✅已动作层 | 无 |
| 10 | DB 查询前必确认 schema | discipline/db_safety.py | ✅已动作层 | 无 |
| 11 | 一律 scripts/run_tests.sh 禁裸 pytest | 无 | 🟡可下沉 | 本轮新 guard |
| 12 | 期望值独立推导禁凑绿灯 | 无 | ❌不可下沉 | 文本层 |
| 13 | [实测]必须当轮贴原始输出 | 无 | ❌不可下沉 | 文本层 |
| 14 | 长任务主动汇报 | 无 | ❌不可下沉 | 文本层 |
| 15 | 顺带缺陷当场修 | finish_guard v2 | ✅已动作层(间接) | 无 |
| 16 | 凡有改动即 commit+push 零积压 | 无 | 🟡可下沉 | 本轮新 guard |
| 17 | 反方审查纪律 | devil-advocate-audit | ✅已动作层 | 无 |

## 本轮下沉 2 条

### #11 禁裸 pytest → discipline/test_discipline.py

- pre_tool_call 拦 terminal 命令：令牌含 `pytest` 且非 `scripts/run_tests.sh` 前缀 → block，指正 run_tests.sh 正门
- 豁免：`scripts/run_tests.sh` 本身、`-k` 单测定位（允许 `run_tests.sh file -k`，禁裸 `pytest file -k`）、pytest --version 这类探针
- 实现参考 no_guessing.py：shlex 切令牌+集合判断，无正则，状态走 _shared_state

### #16 commit+push 零积压 → finish_guard 增强

- pre_verify 已有通道（改动过文件+回复收尾时触发）已拦「请示式收尾」
- 增强：改动过文件但回复不含 commit 哈希/推送证据 → 注入续跑消息要求先 commit+push 再收尾
- 豁免：回复已含 commit 哈希（7-40hex）、已说明纯文档/无代码改动、attempt 超上限

## 判定标准（沉淀）

一条规矩可下沉 = 存在机器可判定的违规信号（命令令牌/工具调用形状/状态谓词），
且正门可指（block 消息能告诉模型怎么改）。纯意图类（期望值独立性/主动汇报/
诚实标注）无判定信号，留在文本层靠审计。
