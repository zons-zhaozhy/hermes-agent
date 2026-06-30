#!/bin/bash
# sync-upstream.sh — 通过 cnb 中转同步 GitHub upstream，自动应用本地补丁
#
# 用法:
#   ./scripts/sync-upstream.sh          # 完整同步
#   ./scripts/sync-upstream.sh --check  # 只检查差距，不执行
#
# 前提: cnb sync-config 分支存在，含 .cnb.yml 云端拉取 pipeline
# 上游基点: refs/sync/upstream-base（独立 ref，不受 push 影响）

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CNB_REMOTE="cnb"
CNB_REPO="z.ai/hermes-agent-mirror"
ORIGIN_REMOTE="origin"
BRANCH="sync-upstream-2026"
MODE="${1:-sync}"

cd "$REPO_DIR"

# ── 颜色 ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${GREEN}[sync]${NC} $*"; }
warn() { echo -e "${YELLOW}[warn]${NC} $*"; }
err()  { echo -e "${RED}[err]${NC} $*" >&2; }

# ── 常量 ──
BASE_REF="refs/sync/upstream-base"

# ── 获取 cnb token ──
get_token() {
    printf "protocol=https\nhost=cnb.cool\n\n" | git credential fill 2>/dev/null | grep '^password=' | cut -d= -f2-
}

# ── 步骤 1: 触发 cnb 云端拉取 GitHub ──
trigger_cnb_sync() {
    log "触发 cnb 云端拉取 GitHub..."
    local TOKEN; TOKEN=$(get_token)
    if [ -z "$TOKEN" ]; then
        err "无法获取 cnb token"
        return 1
    fi

    local RESULT SN
    RESULT=$(curl -s -X POST "https://api.cnb.cool/${CNB_REPO}/-/build/start" \
        -H "Authorization: Bearer $TOKEN" \
        -H "Content-Type: application/json" \
        -d '{"event":"api_trigger","branch":"sync-config"}')
    SN=$(echo "$RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('sn',''))" 2>/dev/null || echo "")

    if [ -z "$SN" ]; then
        err "触发构建失败: $RESULT"
        return 1
    fi
    log "构建 SN: $SN"

    # 等待完成（最多 5 分钟）
    log "等待 cnb 构建完成（最多 5 分钟）..."
    local elapsed=0
    while [ $elapsed -lt 300 ]; do
        sleep 15
        elapsed=$((elapsed + 15))
        local status
        status=$(curl -s "https://api.cnb.cool/${CNB_REPO}/-/build/status/${SN}" \
            -H "Authorization: Bearer $TOKEN" \
            -H "Accept: application/json" \
            | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status','unknown'))" 2>/dev/null || echo "unknown")
        echo -e "  ${CYAN}${elapsed}s${NC} → $status"
        if [ "$status" = "success" ]; then
            log "cnb 构建成功"
            return 0
        elif [ "$status" = "error" ] || [ "$status" = "failed" ]; then
            err "cnb 构建失败"
            log "日志: https://cnb.cool/${CNB_REPO}/-/build/logs/${SN}"
            return 1
        fi
    done
    err "cnb 构建超时"
    return 1
}

# ── 步骤 2: 从 cnb fetch upstream-main ──
fetch_upstream() {
    log "从 cnb 拉取 upstream-main..."
    git fetch "$CNB_REMOTE" upstream-main 2>&1 | tail -3
    log "upstream-main 最新内容:"
    git log --oneline "$CNB_REMOTE/upstream-main" -1
}

# ── 步骤 3: 收集本地补丁 SHA ──
collect_patches() {
    # 收集当前分支相对于旧 upstream 的所有补丁（正序），每行一个 SHA。
    # 关键：补丁列表必须在 fetch 新 upstream 之前收集，
    # 因为 fetch 后 cnb/upstream-main 指向新基点，旧补丁链会丢失。
    #
    # 策略：用 reflog 找到 fetch 前的 HEAD，收集 HEAD 上的所有非 merge commit。
    # 补丁定义：不属于 upstream 孤儿 commit 链的 commit。
    # 简化：收集当前分支相对于 cnb/upstream-main 的 diff
    # 注意：这只在 fetch 前有效
    local patch_ref; patch_ref="refs/sync/local-patches"
    local list_file; list_file="${HERMES_HOME:-$HOME/.hermes}/.sync-patches.txt"

    # 优先从持久化文件读取（最可靠）
    if [ -f "$list_file" ]; then
        cat "$list_file"
        return 0
    fi

    # 回退：优先用 upstream-base ref，再用 cnb/upstream-main
    local fallback_base; fallback_base=$(git rev-parse "$BASE_REF" 2>/dev/null || echo "")
    if [ -n "$fallback_base" ]; then
        git rev-list --reverse "${fallback_base}..HEAD" 2>/dev/null || true
    else
        warn "upstream-base ref 未设置，使用 cnb/upstream-main（fetch 后可能不可靠）"
        git rev-list --reverse "${CNB_REMOTE}/upstream-main..HEAD" 2>/dev/null || true
    fi
}

# ── 从 manifest 读取补丁分类 ──
# manifest 文件: ${HERMES_HOME:-$HOME/.hermes}/.patch-manifest.yaml
# 返回补丁的分类: pr-track / local-only / upstream-absorbed / unknown
get_patch_category() {
    local sha="$1"
    local manifest="${HERMES_HOME:-$HOME/.hermes}/.patch-manifest.yaml"
    if [ ! -f "$manifest" ]; then
        echo "unknown"
        return
    fi
    python3 -c "
import sys
sha, path = sys.argv[1], sys.argv[2]
try:
    text = open(path).read()
    chunk = text.split(sha, 1)[1]
    cat_line = [l for l in chunk.splitlines() if 'category:' in l]
    if cat_line:
        print(cat_line[0].split(':', 1)[1].strip().strip('\"'))
    else:
        print('unknown')
except (IndexError, FileNotFoundError):
    print('unknown')
" "$sha" "$manifest" 2>/dev/null || echo "unknown"
}

# ── 保存当前补丁列表（在 fetch 前调用）──
save_patches() {
    local list_file; list_file="${HERMES_HOME:-$HOME/.hermes}/.sync-patches.txt"

    # 优先用持久化 ref（不受 push 影响），回退到 cnb/upstream-main
    local old_base; old_base=$(git rev-parse "$BASE_REF" 2>/dev/null || echo "")
    if [ -z "$old_base" ]; then
        old_base=$(git rev-parse "${CNB_REMOTE}/upstream-main" 2>/dev/null || echo "")
        [ -n "$old_base" ] && warn "upstream-base ref 未设置，回退到 cnb/upstream-main（push 后可能不可靠）"
    fi

    if [ -z "$old_base" ]; then
        err "无法确定 upstream 基点，同步中止"
        return 1
    fi

    git rev-list --reverse "${old_base}..HEAD" > "$list_file" 2>/dev/null
    local count; count=$(wc -l < "$list_file" | tr -d ' ')

    # 安全护栏：0 补丁但 HEAD 领先 cnb/upstream-main → 基点失效
    if [ "$count" -eq 0 ]; then
        local remote_diff; remote_diff=$(git rev-list --count "${CNB_REMOTE}/upstream-main..HEAD" 2>/dev/null || echo "0")
        if [ "$remote_diff" -gt 0 ]; then
            err "补丁列表为空但 HEAD 领先 cnb/upstream-main ${remote_diff} 个 commit"
            err "upstream-base ($(git log -1 --format='%h' "$old_base" 2>/dev/null)) 可能已失效"
            err "修复: git update-ref $BASE_REF <正确的upstream基点SHA>"
            return 1
        fi
    fi

    log "保存 $count 个补丁到 $list_file（基点: $(git log -1 --format='%h' "$old_base" 2>/dev/null || echo "${old_base:0:12}")）"
    return 0
}

# ── 步骤 4: rebase 补丁到新 upstream ──
rebase_patches() {
    local new_base; new_base="$CNB_REMOTE/upstream-main"

    # 先保存当前补丁列表
    local patch_file; patch_file="/tmp/hermes-patches-$(date +%s).txt"
    collect_patches > "$patch_file"
    local count; count=$(wc -l < "$patch_file" | tr -d ' ')

    if [ "$count" -eq 0 ]; then
        # 二次确认：HEAD 确实没有本地 commit
        local remote_diff; remote_diff=$(git rev-list --count "${CNB_REMOTE}/upstream-main..HEAD" 2>/dev/null || echo "0")
        if [ "$remote_diff" -gt 0 ]; then
            err "补丁列表为空但 HEAD 领先 cnb/upstream-main ${remote_diff} 个 commit — 补丁可能已丢失"
            err "手动恢复: git rev-list --reverse <旧基点>..HEAD > ${HERMES_HOME:-$HOME/.hermes}/.sync-patches.txt"
            return 1
        fi
        warn "无补丁，直接 reset 到 upstream"
        git checkout "$BRANCH"
        git reset --hard "$new_base"
        git update-ref "$BASE_REF" "$new_base"
        return 0
    fi

    log "$count 个本地补丁需要 rebase 到新 upstream"
    echo "  补丁列表:"
    while IFS= read -r sha; do
        [ -z "$sha" ] && continue
        local msg; msg=$(git log -1 --format='%s' "$sha" | cut -c1-50)
        local cat; cat=$(get_patch_category "$sha")
        echo "    $(git log -1 --format='%h' "$sha") [$cat] ${msg}"
    done < "$patch_file"

    # 创建临时分支，reset 到新 upstream，逐个 cherry-pick
    local tmp_branch; tmp_branch="sync-tmp-$(date +%s)"
    git checkout -b "$tmp_branch" "$new_base"

    local ok=0 fail=0
    local failed_shas=()
    while IFS= read -r sha; do
        [ -z "$sha" ] && continue
        local msg; msg=$(git log -1 --format='%s' "$sha")
        if git cherry-pick --no-commit "$sha" 2>/dev/null; then
            if [ -n "$(git diff --cached --name-only)" ]; then
                git commit --no-verify -m "$msg" -q
                log "  ✓ $(git log -1 --format='%h' HEAD) ${msg:0:50}"
                ok=$((ok + 1))
            else
                git reset -q
                warn "  ⊘ ${sha:0:12} 无变更（已被 upstream 包含）: ${msg:0:50}"
            fi
        else
            git cherry-pick --abort 2>/dev/null || true
            git reset --hard HEAD -q
            err "  ✗ 冲突: ${sha:0:12} ${msg:0:50}"
            failed_shas+=("$sha")
            fail=$((fail + 1))
        fi
    done < "$patch_file"

    # 替换分支
    git checkout "$BRANCH" 2>/dev/null || git checkout -b "$BRANCH"
    git reset --hard "$tmp_branch" -q
    git branch -D "$tmp_branch" -q

    log "rebase 完成: $ok 成功, $fail 冲突, $(grep -c . "$patch_file") 总计"

    if [ $fail -gt 0 ]; then
        warn "冲突补丁（需手动处理）:"
        for sha in "${failed_shas[@]}"; do
            local cat; cat=$(get_patch_category "$sha")
            echo "    $(git log -1 --format='%h %s' "$sha" | cut -c1-70) [$cat]"
            if [ "$cat" = "local-only" ]; then
                warn "    → local-only 补丁，如 upstream 已有类似修复可考虑丢弃"
            elif [ "$cat" = "pr-track" ]; then
                warn "    → PR-track 补丁，需手动解决冲突后重新 cherry-pick"
            fi
        done
        warn "这些补丁需要手动 cherry-pick: git cherry-pick <sha>"
        return 1
    fi
    # 更新 upstream-base ref 到新基点
    git update-ref "$BASE_REF" "$new_base"
    log "upstream-base ref 更新到 $(git log -1 --format='%h' "$new_base")"
    return 0
}

# ── 步骤 5: 验证 ──
verify() {
    log "验证..."
    source .venv/bin/activate 2>/dev/null || source venv/bin/activate 2>/dev/null || true

    # 语法检查关键文件
    local py_files=(
        agent/prompt_builder.py
        agent/system_prompt.py
        agent/context_compressor.py
        hermes_cli/main.py
        hermes_cli/goals.py
        tools/file_tools.py
        tools/mixture_of_agents_tool.py
        cli.py
    )
    local errors=0
    for f in "${py_files[@]}"; do
        if python -m py_compile "$f" 2>/dev/null; then
            echo "  ✓ $f"
        else
            err "  ✗ $f 语法错误"
            errors=$((errors + 1))
        fi
    done

    if [ $errors -gt 0 ]; then
        err "$errors 个文件语法错误"
        return 1
    fi

    # 快速测试
    log "运行核心测试..."
    if python -m pytest tests/agent/test_prompt_builder.py tests/agent/test_system_prompt.py tests/hermes_cli/test_goals.py -q --no-header --timeout=30 2>&1 | tail -3; then
        log "核心测试通过"
    else
        warn "部分测试失败，请检查"
    fi
    return 0
}

# ── 步骤 6: 推送 ──
push_all() {
    log "推送到 cnb..."
    git push "$CNB_REMOTE" "$BRANCH:main" --force 2>&1 | tail -3

    log "推送到 origin (GitHub)..."
    if git push "$ORIGIN_REMOTE" "$BRANCH:main" --force 2>&1 | tail -3; then
        log "origin 推送成功"
    else
        warn "origin 推送失败（网络问题？），cnb 已是最新"
    fi
}

# ── 主流程 ──
main() {
    echo ""
    echo "══════════════════════════════════════════════"
    echo "  Hermes Agent — Upstream Sync via cnb"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "══════════════════════════════════════════════"
    echo ""

    case "$MODE" in
        --check)
            log "检查模式：只看差距不执行"
            git fetch "$CNB_REMOTE" upstream-main 2>/dev/null || true
            local current; current=$(git log -1 --format='%h' "$CNB_REMOTE/upstream-main" 2>/dev/null)
            local local_head; local_head=$(git log -1 --format='%h' HEAD)
            local base_sha; base_sha=$(git rev-parse "$BASE_REF" 2>/dev/null || echo "")
            local base_info; [ -n "$base_sha" ] && base_info=$(git log -1 --format='%h' "$base_sha" 2>/dev/null) || base_info="未设置"
            log "cnb upstream-main: $current"
            log "本地 HEAD:          $local_head"
            log "upstream-base ref:  $base_info"
            local patches; patches=$(git rev-list --count "$CNB_REMOTE/upstream-main..HEAD" 2>/dev/null || echo "?")
            log "本地领先 (vs cnb):  $patches 个补丁"
            if [ -n "$base_sha" ]; then
                local real_patches; real_patches=$(git rev-list --count "${base_sha}..HEAD" 2>/dev/null || echo "?")
                log "本地补丁 (vs base): $real_patches 个"
            fi
            ;;
        sync|"")
            save_patches || { err "save_patches 失败，同步中止（防止丢失补丁）"; exit 1; }
            trigger_cnb_sync || exit 1
            fetch_upstream
            rebase_patches || warn "有冲突需手动处理"
            verify || warn "验证有问题但继续推送"
            push_all
            log "同步完成！"
            log "当前 HEAD: $(git log -1 --format='%h %s' HEAD | cut -c1-60)"
            ;;
        *)
            err "未知模式: $MODE"
            err "用法: $0 [--check|sync]"
            exit 1
            ;;
    esac
}

main
