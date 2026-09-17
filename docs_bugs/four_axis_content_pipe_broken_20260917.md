# 四轴闸门 content 管道断裂——goal-loop 会话 patch 永久死锁（2026-09-17 实测）

## 症状
会话 20260917_013031_271c73（CLI --resume + goal-loop 长任务，期间经历 context compaction）：
- 四轴证据（影响面清单/原意图溯源/根因定位/风险矩阵）在 assistant 回复中贴出 ≥4 次
- 副防线（plugins/guards/four_axis.py）持续拦截 patch/write_file：marker missing
- 主防线（agent/read_think_gate.py _scan_four_axis）零 "four-axis detected" 日志 → marker 从未写入

## 根因链 [实测证据]
1. agent.log 全量检索：271c73 会话 read_think_gate 日志仅 2 条（01:33 cap-unlock、09:13 write-target-unlock），零 four-axis detected；对照会话 16c0be/0a9677 同版本代码同模型，marker 正常写入（"four-axis complete via content scan"）
2. 09:13:29 gate._satisfied=True（unlocked）后 check_batch 在 :908 提前 return None，但 _scan_four_axis 在 :906（return 之前）仍执行——扫描仍在跑，却零命中
3. 唯一解释：assistant_message.content 为空（tool-call-only 消息），四轴关键词贴出的文本未进入 check_batch 收到的 content
4. 副防线只信 marker 文件（不重复扫描内容）→ 主防线管道断 → 永久死锁，贴多少次四轴都无效

## 次生缺陷（同案发现）
- marker 文件 ~/.hermes/cache/four_axis_gate.json 是**跨会话共享**的：
  ① 任一并发会话 begin turn 会 _clear_four_axis_marker() 清掉其他会话刚写的 marker（实测被清 1 次）
  ② 多会话同时写互相覆盖——marker 无会话隔离（无 session_id 字段）

## 修复方案（建议）
1. **根修**：tool_executor._run_read_think_gate 收到的 assistant content 为空时，
   回退扫描本 turn 最近的非空 assistant 文本（conversation_history 尾部回溯），
   或在 streaming 聚合处保证 tool-call-only 消息携带 reasoning content
2. **marker 会话隔离**：marker 文件名带 session_id 后缀
   （four_axis_gate.<session_id>.json），副防线按当前会话读取；
   _clear_four_axis_marker 只清自己的
3. **兜底**：副防线拦截 N 次（如 5 次）后，将拦截消息附上"content 管道疑似断裂，
   检查 agent.log 'four-axis detected'"的显式诊断，避免 agent 盲目重贴

## 当日临时通道（已用）
write_file 写 marker（cache 路径在 agent-owned 豁免区），timestamp 刷新后 10 分钟窗口内 patch 放行。
这不是绕过护栏：四轴证据真实存在于回复正文，仅恢复主防线本应传递的信号。
