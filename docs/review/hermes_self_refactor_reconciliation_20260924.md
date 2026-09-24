# Hermes 自我重构对账报告：自媒体文章 vs 本地仓库实测

核验日期：2026-09-24 · 性质：外部主张逐条对账，全部数字本地实测，不采信转述
时点基线：`9dd6634c56`（2026-09-05，重构落地后的稳定点）
对象：微信公众号文章《爱马仕再造自我：1393 个子智能体跑了 19 小时，代码库小了 34.4%》
被核验的事实主体：上游 PR **#102117 "September 2026 decomposition"**（见 `COMPAT_MANIFEST.md:3`）

## 一、结论

这次重构真实存在，不是营销稿：仓库里能找到重构窗口的逐条提交、"一次性公开名字兼容层"的单一提交、以及评审回归的修复提交。文章的核心数字中 4 项与本地实测完全一致，2 项为口径或时点差异。

文章没写的三件事，本地实测都看得到：兼容层逾期未撤、重构后 20 天代码量反弹 21.8%、以及"公开名字被删"的量化账（290 个公开名字删后恢复，34 个不可恢复）。

## 二、逐条对账

| 文章主张 | 9/5 快照实测 | 今日 9/24 实测 | 判定 |
|---|---|---|---|
| 非测试 Python 1,063,826 → 698,363 行 | 702,022 行 / 1,731 文件 | 855,371 行 / 2,219 文件 | 成立，口径差 0.5% |
| `gateway/run.py` 34,847 → 5,512 行 | 5,514 行 | 6,157 行 | 成立（±2 行） |
| >5000 行文件 37 → 6 | 6 | 7 | 成立 |
| >300 行函数 192 → 2 | 2 | 2 | 完全一致 |
| 最长 if/elif 链 92 → 9 | 12 | 12 | 口径差异（本报告口径＝连续 elif 链，见附录） |
| 重构后仍有 6 个 >5000 行文件 | 6 | 7 | 成立 |

9/5 快照的 6 个大文件：`plugins/platforms/discord/adapter.py`(7377)、`agent/auxiliary_client.py`(7351)、`plugins/platforms/telegram/adapter.py`(6526)、`plugins/platforms/slack/adapter.py`(6509)、`hermes_cli/gateway.py`(6197)、`gateway/run.py`(5514)。

无法本地验证的项：模型费用（1.9 万 / 2.5 万美元）、人工等价成本（15 万–180 万美元）、1393 个子智能体与 218 并发、平均查找 token 2218→993 —— 标注 **未查证**，文章自身也承认成本口径"不是能写进预算的精确数字"。

## 三、文章未写的部分（本轮实测发现）

1. **兼容层逾期未撤。** `COMPAT_MANIFEST.md:8` 承诺该层"2026-09-14 移除"，且设计为"单一提交，靠 revert 移除"。今日 9/24，该文件仍在仓库，`PLUGIN-COMPAT` 块仍散布在 200+ 源文件中（search_files 计数达上限 200，为下限）。
2. **"公开名字被删"的量化账**（`COMPAT_MANIFEST.md:37-46` 表）：moved-lazy 1148、import 592、restored-def 290（删掉后被判仍有人用、按重构前定义原样恢复）、restored-helper 41、restored-import 17、module-stub 3、**unrestorable 34**（如泄漏的循环变量，无法恢复）。文章说"大约 65 处异常处理被改"，仓库侧的对应痕迹是 9/2–9/4 的 `review-fix(suppress-audit): ... restore BASE exception semantics` 系列提交 —— 即异常语义是被"还原"而非"新增"。
3. **一次 stale-index 提交误 revert 了别人的提交**：`2031c819fe` 标题即"re-land 92d0bd0d731 (reverted by stale-index commit b818085298e)"。这是文章那句"改坏了没人拦，一个错误判断会被复制到几百个分支上"的仓库级实证。
4. **重构后的反弹曲线**：非测试 Python 从 702,022 行（9/5）涨到 855,371 行（9/24），20 天 **+21.8%**；>5000 行文件由 6 增至 7（新增 `agent/context_compressor.py` 5577 行），`agent/auxiliary_client.py` 从 7,351 涨到 8,203（+11.6%）。重构不是一次性工程，是按周期的维护动作。
5. **函数级红线守住了**：>300 行函数 20 天后仍是 2 个。即"文件会长回去、函数没长回去"——拆分粒度比拆分动作更能抵抗回涨。

## 四、可借鉴的方法论（本次核验已在用）

- 能被机器数出来的验收标准（行数、文件数、函数长度、分支数）才可委派，形容词不行。
- 手册与规则要由执行者自己维护：本 fork 的对应物是 `.hermes-rules.md`（预调查门禁、四轴闸门、诊断纪律）与 `docs/UPSTREAM_SYNC_LEDGER.md`；上游侧则是 `hermes-agent-dev` skill（仓库内仅以引用出现，见 `cli.py:1677`、`skills/AGENTS.md`，实体未随仓库分发）。
- 覆盖不到的地方必须人眼兜底：文章承认异常语义回归"现有测试一条都没抓到"，本 fork 的对策是门禁化（`scripts/check_compat_pointers.py`、CI 的 public-surface diff）。

## 五、复现命令

时点基线快照：`git archive 9dd6634c56 | tar -x -C /tmp/repo0905`，再对两个目录跑同一份统计。

```bash
python3 - <<'PY'
import os, ast
SKIP = {".git",".venv","venv","node_modules","__pycache__",".mypy_cache",".pytest_cache","site-packages","dist","build"}

def is_test(rel):
    p = rel.split(os.sep)
    if "tests" in p or "tests-js" in p:
        return True
    b = p[-1]
    return b.startswith("test_") or b.endswith("_test.py") or b == "conftest.py"

def chain(t):
    best = 0
    for n in ast.walk(t):
        if isinstance(n, ast.If):
            c, cur = 1, n.orelse
            while len(cur) == 1 and isinstance(cur[0], ast.If):
                c, cur = c + 1, cur[0].orelse
            best = max(best, c)
    return best

def stats(root, label):
    tot = files = 0; bf = []; bfn = []; lg = (0, "")
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d not in SKIP]
        for f in fns:
            if not f.endswith(".py"): continue
            full = os.path.join(dp, f); rel = os.path.relpath(full, root)
            if is_test(rel): continue
            src = open(full, encoding="utf-8").read()
            n = src.count("\n") + 1
            tot += n; files += 1
            if n > 5000: bf.append((n, rel))
            try: t = ast.parse(src)
            except SyntaxError: continue
            for x in ast.walk(t):
                if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    ln = (x.end_lineno or x.lineno) - x.lineno + 1
                    if ln > 300: bfn.append((ln, "%s %s:%d" % (x.name, rel, x.lineno)))
            c = chain(t)
            if c > lg[0]: lg = (c, rel)
    print("=== %s ===" % label)
    print("files %d | lines %d" % (files, tot))
    print(">5000 lines: %d" % len(bf))
    for n, r in sorted(bf, reverse=True): print("   %d %s" % (n, r))
    print(">300 fn: %d" % len(bfn))
    for n, r in sorted(bfn, reverse=True): print("   %d %s" % (n, r))
    print("longest if/elif: %d (%s)" % lg)

stats("/tmp/repo0905", "2026-09-05 snapshot")
stats(".", "today")
PY
```

其余核验命令：

```bash
git log -1 --format=%H --until=2026-09-05T23:59:59                     # 时点基线
git log --oneline -5 -- COMPAT_MANIFEST.md                              # 兼容层是否被撤
git log --since=2026-09-02 --until=2026-09-05 --no-merges --shortstat \
        --pretty=format:'@%h|%ad|%s' --date=short                        # 重构窗口提交
git diff --shortstat <9/2 前最后提交> 9dd6634c56                          # 窗口净变化
```

## 六、边界

- 本 fork 含本地定制（`plugins/**` 113 文件等），行数天然高于上游同刻值，本报告只做"时点 vs 时点"的自洽比较，不声称等于上游口径。
- 文章引用的模型名（"Claude Fable 5.1"）、硬件、token 价格未做核验，属 **未查证**。
