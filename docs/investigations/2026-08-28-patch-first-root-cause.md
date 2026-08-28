# patch 指令失效根因调查 — 2026-08-28

## Evidence
- [实测] `~/.hermes/config.yaml:871` patch-first、`:880` source-code-write-guard 均已 enable。
- [实测] `plugins/patch-first/__init__.py:94-110` `_on_pre_tool_call` 只写 `/tmp/.hermes_patch_first_pending` 状态文件，返回 None——**不 block**。提醒经 `pre_llm_call` 在 60 秒内的下一次 LLM 调用注入（行113-146），此时 sed 写入早已执行成功。
- [实测] `plugins/source_code_write_guard/__init__.py` 三层检测（python 内联写/cat heredoc/tee/in-place）仅覆盖 `_SOURCE_EXTENSIONS` 后缀白名单目标；无后缀/.env/.lic 等不拦，且**放行后 sed 层它不管**（职责划分：write_guard 管通道，patch-first 管方式）。
- [实测] 系统提示 .hermes.md 只写「Edit with `patch`/`write_file`. Do NOT print code blocks」为偏好措辞；memory 条目「patch优先(0826拍板)」在上下文压缩后约束力衰减为可选项。
- [实测] 本会话 ReadThink gate 只对 terminal 计数只读调查，不拦写通道。

## Findings（根因三层）
1. **主根因**：唯一的"方式闸门"patch-first 是纯提醒型（软信号），sed/脚本写入当场成功；对模型的行为约束依赖下一轮提醒注入，且 60s 过期。机械闸门缺位。
2. write_guard 只按"目标后缀"拦通道，sed 等替换类写入（目标在白名单后缀内时本应拦，但 sed s/old/new/ file 形式无重定向，三层检测均不命中 `_extract_redirect_target`）漏判。
3. 提示词/memory 均为软约束，压缩后衰减。

## Actions
- patch-first `_on_pre_tool_call` 升级为硬拦截：返回 `{"action":"block","message":...}`（对齐 source_code_write_guard 的既有 block 契约）。
- 验证：terminal 发 `sed -i` 应被 block；`sed -n` 只读应放行。
