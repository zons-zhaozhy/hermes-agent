#!/usr/bin/env bash
# pre-submit-check.sh — 提交 PR 到 NousResearch/hermes-agent 前的自动检查
# 用法: ./scripts/pre-submit-check.sh [--fix]
#   --fix: 自动修复可修复的问题（如中文替换提示）
#
# 检查项目:
#   1. 分支是否从 upstream/main 分叉
#   2. diff 中是否有跨工具 schema 静态引用
#   3. diff 中是否有与 PR 无关的文件（夹带）
#   4. diff 中是否有中文字符
#
# 退出码: 0 = 通过, 1 = 未通过（有违规项）

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

PASS=0
FAIL=0
WARN=0
DIFF_FILES=""

# ── 辅助函数 ──────────────────────────────────────────

check_header() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  $1"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

pass_msg() {
    echo -e "  ${GREEN}✓${NC} $1"
    PASS=$((PASS + 1))
}

fail_msg() {
    echo -e "  ${RED}✗${NC} $1"
    FAIL=$((FAIL + 1))
}

warn_msg() {
    echo -e "  ${YELLOW}⚠${NC} $1"
    WARN=$((WARN + 1))
}

# ── 获取 diff 文件列表 ─────────────────────────────────

get_diff_files() {
    if [ -z "$DIFF_FILES" ]; then
        # 对比 upstream/main（如果存在）或 HEAD~1
        if git rev-parse --verify upstream/main >/dev/null 2>&1; then
            DIFF_FILES=$(git diff upstream/main --name-only 2>/dev/null || echo "")
        else
            DIFF_FILES=$(git diff HEAD~1 --name-only 2>/dev/null || echo "")
        fi
    fi
}

# ── 检查 1：分支干净度 ─────────────────────────────────

check_branch_cleanliness() {
    check_header "Gate 1/4: 分支干净度"

    local branch
    branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")

    # 检查是否从 upstream/main 分叉
    if git rev-parse --verify upstream/main >/dev/null 2>&1; then
        local merge_base
        merge_base=$(git merge-base HEAD upstream/main 2>/dev/null || echo "")
        if [ -n "$merge_base" ]; then
            local upstream_head
            upstream_head=$(git rev-parse upstream/main 2>/dev/null)
            if [ "$merge_base" = "$upstream_head" ]; then
                pass_msg "分支从 upstream/main 分叉"
            else
                fail_msg "分支基点落后 upstream/main（可能从 fork 工作分支分叉）"
                echo "          merge-base: ${merge_base:0:12}"
                echo "          upstream:    ${upstream_head:0:12}"
                echo "          修复: git checkout -b fix/xxx upstream/main"
            fi
        else
            fail_msg "找不到与 upstream/main 的共同祖先（孤儿分支？）"
        fi
    else
        warn_msg "无 upstream remote，跳过分支干净度检查"
        echo "          添加: git remote add upstream git@github.com:NousResearch/hermes-agent.git"
    fi

    # 检查是否有 fork 本地特性文件泄露
    get_diff_files
    local local_features=""
    local_features=$(echo "$DIFF_FILES" | grep -E "agent/(agent_init|tool_executor|deliberation_gate|self_check)" || echo "")
    if [ -n "$local_features" ]; then
        fail_msg "diff 包含 fork 本地特性文件:"
        echo "$local_features" | while read -r f; do echo "          $f"; done
        echo "          修复: 从 upstream/main 创建干净分支，只 cherry-pick 目标改动"
    else
        pass_msg "无 fork 本地特性泄露"
    fi
}

# ── 检查 2：跨工具 schema 静态引用 ─────────────────────

CROSS_TOOL_PATTERNS=(
    # 工具名: 不应出现在其他工具的 schema description 中
    "execute_code"
    "write_file"
    "patch"
    "read_file"
    "web_search"
    "web_extract"
    "browser_navigate"
    "terminal"
    "delegate_task"
    "search_files"
    "vision_analyze"
)

check_cross_tool_refs() {
    check_header "Gate 2/4: 跨工具 schema 静态引用"

    get_diff_files
    local violations=0

    for f in $DIFF_FILES; do
        # 只检查 Python 文件中的 schema 定义
        if [[ "$f" != *.py ]]; then
            continue
        fi
        if [ ! -f "$f" ]; then
            continue
        fi

        # 检查文件是否包含 SCHEMA 定义
        if ! grep -q '_SCHEMA.*description' "$f" 2>/dev/null; then
            continue
        fi

        # 获取当前文件定义的工具名
        local this_tool=""
        this_tool=$(grep -oP '"name":\s*"\K[^"]+' "$f" | head -1 || echo "")

        # 检查 schema description 中是否引用了其他工具
        for tool in "${CROSS_TOOL_PATTERNS[@]}"; do
            if [ "$tool" = "$this_tool" ]; then
                continue  # 跳过自身引用
            fi
            # 在 diff 的 + 行中搜索（新增/修改的 schema 描述）
            if git diff upstream/main -- "$f" 2>/dev/null | python3 -c "
import sys
text = sys.stdin.read()
import re
print('MATCH' if re.search(r'^\+.*\"description\".*\\b${tool}\\b', text, re.MULTILINE) else '')
" 2>/dev/null | grep -q "MATCH"; then
                if [ $violations -eq 0 ]; then
                    echo ""
                fi
                fail_msg "$f: schema description 中包含静态引用 '$tool'"
                violations=$((violations + 1))
            fi
        done
    done

    if [ $violations -eq 0 ]; then
        pass_msg "无跨工具 schema 静态引用"
    else
        echo ""
        echo "          AGENTS.md:917-918 禁止静态跨工具 schema 引用"
        echo "          修复: 将引用移到 model_tools.py _compute_tool_definitions() 动态注入"
        echo "          参考: model_tools.py:453-464 (execute_code), :495-513 (browser_navigate)"
    fi
}

# ── 检查 3：夹带文件 ─────────────────────────────────

check_unrelated_files() {
    check_header "Gate 3/4: 夹带文件检测"

    get_diff_files
    local py_count
    py_count=$(echo "$DIFF_FILES" | grep -c '\.py$' || echo "0")

    if [ "$py_count" -eq 0 ]; then
        pass_msg "无 Python 文件变更"
        return
    fi

    # 检查是否有常见的夹带文件模式
    local suspicious=""
    # web_server.py — 常见夹带（认证绕过等）
    suspicious=$(echo "$DIFF_FILES" | grep "web_server.py" || echo "")
    # 配置文件变更
    suspicious="$suspicious"$(echo "$DIFF_FILES" | grep -E "config\.(yaml|py)|\.env" || echo "")

    if [ -n "$suspicious" ]; then
        warn_msg "以下文件经常被夹带，请确认与 PR 目标相关:"
        echo "$suspicious" | while read -r f; do
            [ -n "$f" ] && echo "          $f"
        done
    fi

    # 如果只有 1-2 个文件变更，通常是干净的
    local total_files
    total_files=$(echo "$DIFF_FILES" | grep -c '.' || echo "0")
    echo ""
    echo "  diff 统计: $total_files 个文件变更"
    echo "$DIFF_FILES" | while read -r f; do [ -n "$f" ] && echo "    $f"; done
}

# ── 检查 4：中文残留 ─────────────────────────────────

check_chinese_chars() {
    check_header "Gate 4/4: 中文残留检测"

    get_diff_files
    local violations=0

    for f in $DIFF_FILES; do
        if [ ! -f "$f" ]; then
            continue
        fi
        # 用 Python 检测 diff 中新增行的中文字符
        local chinese_lines
        chinese_lines=$(git diff upstream/main -- "$f" 2>/dev/null | python3 -c "
import sys, re
text = sys.stdin.read()
# 匹配以 + 开头且包含 CJK 字符的行
for line in text.split('\n'):
    if line.startswith('+') and re.search(r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]', line):
        print(line)
" 2>/dev/null || echo "")
        if [ -n "$chinese_lines" ]; then
            fail_msg "$f: diff 中包含中文字符"
            echo "$chinese_lines" | head -5 | while read -r line; do
                echo "          ${line:0:100}"
            done
            violations=$((violations + 1))
        fi
    done

    # 检查 commit message
    local last_commit_msg
    last_commit_msg=$(git log -1 --format='%s%n%b' 2>/dev/null || echo "")
    if echo "$last_commit_msg" | python3 -c "
import sys, re
text = sys.stdin.read()
if re.search(r'[\u4e00-\u9fff]', text):
    print('MATCH')
" 2>/dev/null | grep -q "MATCH"; then
        fail_msg "commit message 中包含中文字符"
        violations=$((violations + 1))
    fi

    if [ $violations -eq 0 ]; then
        pass_msg "无中文残留"
    else
        echo ""
        echo "          上游不接受中文化改动（含注释），这是历史被拒主因之一"
    fi
}

# ── 主流程 ────────────────────────────────────────────

main() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║  PR 提交前自动检查                                       ║"
    echo "║  目标: NousResearch/hermes-agent                         ║"
    echo "╚══════════════════════════════════════════════════════════╝"

    check_branch_cleanliness
    check_cross_tool_refs
    check_unrelated_files
    check_chinese_chars

    # ── 结果汇总 ──────────────────────────────────────
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  结果: ${GREEN}${PASS} 通过${NC}  ${RED}${FAIL} 失败${NC}  ${YELLOW}${WARN} 警告${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if [ $FAIL -gt 0 ]; then
        echo ""
        echo -e "  ${RED}未通过 — 修复以上 ${FAIL} 项问题后再提 PR${NC}"
        echo ""
        echo "  修复后重新检查: ./scripts/pre-submit-check.sh"
        exit 1
    else
        echo ""
        echo -e "  ${GREEN}全部通过 — 可以创建 PR${NC}"
        echo ""
        echo "  下一步: 完成 Gate 2（人工自审清单）后执行:"
        echo "    gh pr create --repo NousResearch/hermes-agent --base main --head zons-zhaozhy:$(git rev-parse --abbrev-ref HEAD) --body-file /tmp/pr-body.md"
        exit 0
    fi
}

main "$@"
