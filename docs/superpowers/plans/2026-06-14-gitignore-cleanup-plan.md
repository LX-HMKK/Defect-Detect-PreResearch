# .gitignore 整理与项目目录优化 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**目标:** 重写 .gitignore 使其分层清晰、消除矛盾规则，清理 temp/ 和 .playwright-mcp/ 等临时文件，保留 results/ 全部实验数据。

**架构:** 零代码任务 — 只改 .gitignore（单文件重写）和删除临时文件（非 results/）。.gitignore 按 OS / IDE / Python / PyTorch / 项目特性 / 大数据目录 六层分组，每层有注释说明。

**技术栈:** git, 文件系统操作

---

## 现状分析

### .gitignore 问题清单

1. **results/ 规则矛盾** — `!results/.gitkeep` → `results/*` → `results/` 三行相互抵消：`results/*` 忽略子文件，`results/` 又忽略目录本身，`!results/.gitkeep` 例外在中间被覆盖
2. **data/ 规则过于激进** — `data/*/` + `data/*` + `!data/.gitkeep` + `!data/README.md` + `!data/DATASET_REGISTRY.md` — 5 行才完成两个文件的白名单
3. **占位 .gitkeep 过多** — `!results/.gitkeep`、`!weights/.gitkeep`、`!data/.gitkeep` — weights 目录已不存在
4. **assets/ 被忽略但实际有 3 个提交文件** — requirements.txt、pyrightconfig.json、setup_miniforge.bat 被错误忽略
5. **docs/ 被注释掉** — `# docs/` 导致 docs 目录的所有新文件（任务书、汇报文档、图表）不被追踪
6. **格式混乱** — 分组注释不统一，混用了 glob 和严格匹配

### 需清空的临时/残留目录

| 目录 | 大小 | 操作 |
|------|------|------|
| `temp/` | ~0.5MB（残留 PNG + .py 脚本） | 清空（运行时按需重建） |
| `.playwright-mcp/` | ~50KB（日志） | 删除 |
| `.pytest_cache/` | ~100KB | 删除 |
| `.claude/worktrees/` | ~200KB（两个旧 worktree 残留） | 删除 |

### 必须保留的目录

| 目录 | 大小 | 原因 |
|------|------|------|
| `results/` | ~17GB | 所有实验数据（模型权重 .ckpt + 对比 JSON + 混淆矩阵） |
| `data/` | ~1.6GB | 训练/测试数据集 |
| `datasets/` | ~604MB | DTD 外部数据集（DRAEM 依赖） |
| `pre_trained/` | ~100MB | HuggingFace/TorchHub 权重缓存 |
| `assets/` | ~1KB | requirements.txt, pyrightconfig.json |

---

### 任务 1: 清理临时文件和残留目录

**文件:**
- 删除: `temp/` 下所有内容（运行时重建）
- 删除: `.playwright-mcp/`
- 删除: `.pytest_cache/`
- 删除: `.claude/worktrees/` 下旧残留（保留 .claude 配置本身）

- [ ] **步骤 1: 清空 temp/ 目录**

```powershell
Get-ChildItem "D:\StudyWorks\3.2\Defect-Detect-PreResearch\temp" -Recurse -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -Confirm:$false
Write-Output "temp/ 已清空"
```

- [ ] **步骤 2: 删除 .playwright-mcp/ 残留日志**

```powershell
Remove-Item "D:\StudyWorks\3.2\Defect-Detect-PreResearch\.playwright-mcp" -Recurse -Force -Confirm:$false -ErrorAction SilentlyContinue
Write-Output ".playwright-mcp/ 已删除"
```

- [ ] **步骤 3: 删除 .pytest_cache/**

```powershell
Remove-Item "D:\StudyWorks\3.2\Defect-Detect-PreResearch\.pytest_cache" -Recurse -Force -Confirm:$false -ErrorAction SilentlyContinue
Write-Output ".pytest_cache/ 已删除"
```

- [ ] **步骤 4: 清理 .claude/worktrees/ 旧残留**

```powershell
# 保留 .claude 配置（settings.local.json, scheduled_tasks.lock），仅删 worktrees
$wtPath = "D:\StudyWorks\3.2\Defect-Detect-PreResearch\.claude\worktrees"
if (Test-Path $wtPath) {
    Remove-Item $wtPath -Recurse -Force -Confirm:$false
    Write-Output ".claude/worktrees/ 已删除"
}
```

- [ ] **步骤 5: 验证清理后状态**

```powershell
Get-ChildItem "D:\StudyWorks\3.2\Defect-Detect-PreResearch" -Directory | Where-Object { $_.Name -match '^(temp|\.playwright|\.pytest)' } | Select-Object Name
# 预期输出：仅 temp/ 存在（空目录），.playwright-mcp 和 .pytest_cache 已消失
```

- [ ] **步骤 6: 提交**

```bash
git add -A
git commit -m "chore: 清理临时文件和残留目录（temp/pytest_cache/playwright/worktrees残留）"
```


### 任务 2: 重写 .gitignore — 分层清晰

**文件:**
- 重写: `.gitignore`（从零重写，6 层结构）

- [ ] **步骤 1: 写入新的 .gitignore**

```gitignore
# ============================================================
# OS 生成文件（Windows / macOS / Linux）
# ============================================================
.DS_Store
Thumbs.db
*~

# ============================================================
# IDE 和编辑器
# ============================================================
.vscode/
.idea/
*.swp
*.swo

# ============================================================
# Python 通用
# ============================================================
__pycache__/
*.py[cod]
*$py.class
*.so
*.egg-info/
.installed.cfg
*.egg
dist/
build/
eggs/
sdist/
wheels/
*.log

# 虚拟环境（始终在项目外）
env/
venv/
ENV/

# ============================================================
# PyTorch / 模型权重
# ============================================================
*.pth
*.pt

# ============================================================
# Jupyter
# ============================================================
.ipynb_checkpoints/

# ============================================================
# 项目特性 — 大数据、大模型、运行时产物
# ============================================================

# 实验输出根目录（17GB+ 模型权重 + JSON 结果）
# 仅提交占位文件和报告 markdown
results/*
!results/.gitkeep
!results/comparison/
!results/comparison/*.md
!results/comparison/*.csv
!results/comparison/*.json
!results/comparison/post_process/
!results/comparison/post_process/*.md
!results/comparison/post_process/*.json
!results/confusion_matrices/
!results/confusion_matrices/*.png
!results/confusion_matrices/*.json
!results/small_sample/
!results/small_sample/*.json

# 运行时临时文件（pycache 重定向目标、推理临时图片）
temp/
!temp/.gitkeep

# 预训练权重缓存（HuggingFace / TorchHub 下载）
# 由 anomalib 首次运行时自动下载，无需纳入版本控制
pre_trained/

# 外部数据集（DTD 等）
# 需手动下载，不提交
datasets/
!datasets/.gitkeep

# ============================================================
# Claude Code / AI 工具
# ============================================================
.claude/
!.claude/settings.json
!.claude/settings.local.json

# Playwright MCP 浏览器日志
.playwright-mcp/

# Claude 记忆目录
.remember/

# ============================================================
# 测试
# ============================================================
.pytest_cache/
htmlcov/
.coverage
coverage.xml

# ============================================================
# 项目特定大文件目录（不提交内容，仅保留结构标记）
# ============================================================

# 数据集目录 — 仅提交 .gitkeep 和注册表
data/*
!data/.gitkeep
!data/README.md
!data/DATASET_REGISTRY.md

# 文档目录大文件排除
docs/*.pptx
```

- [ ] **步骤 2: 验证 .gitignore 规则有效性**

```bash
cd "D:\StudyWorks\3.2\Defect-Detect-PreResearch" && git status --short
```

预期结果：
- `results/` 下仅显示 comparison/*.json、confusion_matrices/*.png 等白名单文件
- `data/` 下仅显示 .gitkeep、README.md、DATASET_REGISTRY.md
- `assets/` 的三个文件出现在 untracked（不再被忽略）
- `docs/` 的所有 .md 文件出现在 untracked（不再被忽略）
- `docs/*.pptx` 被忽略
- `temp/` 被忽略但 .gitkeep 被追踪
- `.playwright-mcp/` 不再出现（已删除）

- [ ] **步骤 3: 提交**

```bash
git add .gitignore
git commit -m "chore: 重写 .gitignore 为 6 层分类结构，修复 assets/docs/data 误忽略问题"
```


### 任务 3: 追踪之前被错误忽略的文件

**文件:**
- 追踪: `assets/pyrightconfig.json`, `assets/requirements.txt`, `assets/setup_miniforge.bat`
- 追踪: `docs/` 下所有 .md 文件（任务书、需求、综述、汇报、讲稿）

- [ ] **步骤 1: 添加 assets/ 到 Git**

```bash
git add assets/
```

- [ ] **步骤 2: 添加 docs/ 下的 .md 文件**

```bash
git add docs/*.md docs/superpowers/plans/*.md docs/superpowers/specs/*.md
```

- [ ] **步骤 3: 添加 .gitkeep 占位文件（确保空目录结构保留）**

```bash
# 确保 datasets/.gitkeep 存在
touch datasets/.gitkeep
git add datasets/.gitkeep

# 确保 results/.gitkeep 存在
mkdir -p results
touch results/.gitkeep
git add results/.gitkeep

# 确保 temp/.gitkeep 存在
mkdir -p temp
touch temp/.gitkeep
git add temp/.gitkeep

# 确保 data/.gitkeep 存在
touch data/.gitkeep
git add data/.gitkeep
```

- [ ] **步骤 4: 提交**

```bash
git add assets/ docs/*.md docs/superpowers/ docs/superpowers/plans/docs/superpowers/specs/
git commit -m "chore: 追踪之前被 .gitignore 误忽略的 assets/ 和 docs/ 文件"
```


### 任务 4: 验证最终状态并清理工作树残留

- [ ] **步骤 1: 最终 git status 检查**

```bash
cd "D:\StudyWorks\3.2\Defect-Detect-PreResearch" && git status
```

预期：只有有意修改的文件被显示，无 .pptx、无临时文件、无 weights、pre_trained 被忽略。

- [ ] **步骤 2: 确认 results/ 不受影响**

```bash
du -sh results/
# 预期：约 17GB，所有 .ckpt 文件和 JSON 完好
```

- [ ] **步骤 3: 提交**

```bash
# 如果还有未预期的变动，处理后再提交
git add -A
git commit -m "chore: 最终验证 — 确认 .gitignore 规则正确，实验数据完好"
```
