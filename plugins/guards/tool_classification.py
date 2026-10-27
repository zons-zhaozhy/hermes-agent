"""工具写语义分类的单一事实源（C2 结构债收敛，2026-10-27）。

背景：全库曾散布 24 处互不一致的工具名写枚举（AST 扫描实测，2026-10-27：
13 处 guard 拦截面——{write_file,patch} 与 {+execute_code} 与 {+skill_manage}
三种形态互不一致；另有 11 处语义别类集合；只读白名单多份且随 browser_*
旧工具名锈化）——每新增守卫复制一份字面量，工具改名/新增写通道即漂移
（2026-10-27 审计：execute_code 收录分裂、browser 旧名三处锈化即其实际产物）。

本模块是「哪些工具具备写文件能力」语义的唯一 owner：
- 消费方一律 import 本模块常量，禁止再持有工具名字面量集合；
- 新增写能力工具（如未来新的文件写通道工具）只改本模块一处；
- 机器门禁 scripts/check_tool_classification.py 以 AST 解析执法：非
  owner 模块出现「含 ≥2 个写工具名的 frozenset 字面量」即 FAIL。

语义分层（消费方按语义选层，禁跨层借用）：
- FILE_WRITE_TOOLS      直接以路径参数写文件的核内工具
- CODE_WRITE_TOOLS      以代码参数间接写文件的工具（code=任意 Python）
- SKILL_WRITE_TOOLS     写技能资产的工具
- STANDARD_WRITE_TOOLS  通用守卫拦截面（文件写+代码写）
- FILE_AND_SKILL_WRITE_TOOLS  文件写+技能写（execute_code/terminal 写
  由 source_write 按命令特征另管的防线用）

注意：terminal 本身不在任何层——它按命令形态动态判定（见
read_think_gate.gate._terminal_writes_file / guards.source_write），
按工具名静态收录 terminal 会把只读命令全拦。
"""

from __future__ import annotations

FILE_WRITE_TOOLS: frozenset[str] = frozenset({"write_file", "patch"})
CODE_WRITE_TOOLS: frozenset[str] = frozenset({"execute_code"})
SKILL_WRITE_TOOLS: frozenset[str] = frozenset({"skill_manage"})

STANDARD_WRITE_TOOLS: frozenset[str] = FILE_WRITE_TOOLS | CODE_WRITE_TOOLS
FILE_AND_SKILL_WRITE_TOOLS: frozenset[str] = (
    FILE_WRITE_TOOLS | SKILL_WRITE_TOOLS
)
# 静态可能写文件的完整面（含 terminal 壳）。仅限「宁可多做」类消费方
# （如索引同步：多同步一次无伤害）；拦截面禁用——terminal 必须按命令
# 形态动态判定（见模块 docstring 尾注）。
MAYBE_WRITE_TOOLS: frozenset[str] = STANDARD_WRITE_TOOLS | frozenset(
    {"terminal"}
)

assert FILE_WRITE_TOOLS == frozenset({"write_file", "patch"})  # 期望: 基集两工具
assert "execute_code" in STANDARD_WRITE_TOOLS  # 期望: 通用层含代码写通道
assert "skill_manage" not in STANDARD_WRITE_TOOLS  # 期望: 技能写不在通用层
