# UI 布局精修 — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 S1 进出动画方向冲突、S2 推理结果布局零散、S3 共享原图 64px bug + 四列比例失调

**Architecture:** 纯前端 CSS/HTML/JS 变更，不涉及后端。先做最独立且影响最小的 S3 64px bug fix，再做 S1 动画统一（改动集中在 animations.js + app.css），然后 S2/S3 布局重构（index.html DOM + app.css 新组件），最后响应式验证。

**Tech Stack:** Vanilla CSS (CSS 自定义属性双模式)、Alpine.js 3.14、WAAPI (Web Animations API)

---

### Task 1: 修复 S3 共享原图 64px 缩略图 bug

**Files:**
- Modify: `modules/ui/static/index.html:125-133`（内联 `<style>` 中的 `.compare-shared-image` 规则）
- Modify: `modules/ui/static/css/app.css:2320-2350`（`.compare-shared-image` + `.compare-shared-img`）

**Goal:** 移除 `height: 64px` 内联覆盖，共享原图恢复为合适的 160-200px 展示尺寸。

- [ ] **Step 1: 删除 index.html 内联样式中的 64px 限制**

在 `index.html` 内联 `<style>` 中，定位第 127-133 行的 `.compare-shared-image` 块：

```css
/* 删除整个 flex row 内联块（第 118-133 行），替换为： */
.compare-shared-image {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
    margin-bottom: 20px;
    padding: 12px;
    background: var(--bg-secondary);
    border-radius: 10px;
}
.compare-shared-image img {
    max-height: 200px;
    width: 100%;
    object-fit: contain;
    border-radius: 6px;
    background: var(--bg);
}
.compare-shared-label {
    font-size: 12px;
    font-weight: 500;
    color: var(--text-secondary);
}
```

即把原来的：
```css
.compare-shared-image {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 20px;
    padding: 8px 12px;
    background: var(--bg-secondary);
    border-radius: 10px;
}
.compare-shared-image img {
    height: 64px;
    width: auto;
    border-radius: 6px;
    object-fit: contain;
    background: var(--bg);
}
.compare-shared-label {
    font-size: 13px;
    font-weight: 500;
    color: var(--text-secondary);
}
```
替换为上面的新代码。

- [ ] **Step 2: 更新 app.css 中的 `.compare-shared-image` 和 `.compare-shared-img`**

定位 `app.css` 第 2319-2350 行，替换为：

```css
/* 共享原图 — column 布局居中 */
.compare-shared-image {
    position: relative;
    width: 100%;
    max-width: 480px;
    margin: 0 auto 20px;
    border-radius: var(--r-lg);
    overflow: hidden;
    background: var(--bg-secondary);
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 12px;
}

.compare-shared-label {
    font-size: 11px;
    font-weight: 500;
    color: var(--text-secondary);
    margin-top: 6px;
    text-align: center;
}

.compare-shared-img {
    display: block;
    width: 100%;
    max-height: 200px;
    object-fit: contain;
    border-radius: var(--r-md);
    background: var(--bg);
}
```

- [ ] **Step 3: 启动 UI 验证 S3 共享原图尺寸**

```bash
python scripts/run_ui.py --no-browser
# → 打开 http://127.0.0.1:8000
# 滚动到 S3（第三页），检查共享原图是否以 160-200px 高度正常显示，容器无大量留白
```

- [ ] **Step 4: Commit**

```bash
git add modules/ui/static/index.html modules/ui/static/css/app.css
git commit -F .git-msg
```

`.git-msg` 内容：
```
fix(ui): 修复 S3 共享原图 64px 缩略图 bug

移除内联 height:64px 样式，恢复 max-height:200px 正常显示。
容器改为 column 居中布局，max-width 从 640px 缩小到 480px。
```

---

### Task 2: 统一 S1 进出动画方向 + 消除 CSS/JS 双动画竞争

**Files:**
- Modify: `modules/ui/static/js/animations.js:365-389`（`snapPageExit` 方向）
- Modify: `modules/ui/static/js/animations.js:309-358`（`snapPageEnter` stagger）
- Modify: `modules/ui/static/css/app.css:756-778`（删除 CSS 退出动画）
- Modify: `modules/ui/static/css/app.css:2522-2533`（禁用 view-timeline 块）
- Modify: `modules/ui/static/js/app.js:197-227`（IntersectionObserver 退出/进入时序）

**Goal:** 离开动画统一为向下推出(+30px)，进入动画从下方推入(+24px→0)。CSS 不再驱动退出动画，全部由 JS WAAPI 控制。

- [ ] **Step 1: 修改 `snapPageExit` — 方向改为向下推出**

在 `animations.js` 第 365-389 行，修改 `snapPageExit` 函数：

```javascript
snapPageExit(section) {
    if (_prefersReducedMotion) return [];

    const inner = section.querySelector(':scope > .snap-page-inner');
    if (!inner) return [];
    
    // 收集所有可见的直接子元素
    const children = [];
    const directChildren = inner.querySelectorAll(':scope > .scroll-reveal, :scope > .scroll-reveal-stagger > .scroll-reveal, :scope > *');
    directChildren.forEach((child) => {
        // 跳过隐藏/零高度元素
        if (child.offsetHeight === 0) return;
        if (child.hasAttribute('x-show') && window.getComputedStyle(child).display === 'none') return;
        children.push(child);
    });
    if (children.length === 0) return [];

    // 取消该 section 内所有正在运行的动画
    children.forEach((child) => {
        child.getAnimations().forEach((a) => a.cancel());
    });

    const animations = [];
    children.forEach((child, i) => {
        animations.push(
            child.animate(
                [
                    { opacity: 1, transform: 'translateY(0) scale(1)' },
                    { opacity: 0, transform: 'translateY(30px) scale(0.97)' }
                ],
                {
                    duration: 350,
                    delay: i * 0.04 * 1000,  // 40ms stagger
                    easing: 'cubic-bezier(0, 0, 0.2, 1)',
                    fill: 'forwards'
                }
            )
        );
    });
    return animations;
},
```

- [ ] **Step 2: 修改 `snapPageEnter` — stagger 从 100ms 改为 80ms**

在 `animations.js` 第 309 行，将 `staggerMs` 默认值从 100 改为 80：

```javascript
snapPageEnter(section, options) {
    options = options || {};
    const { staggerMs = 80, duration = 500 } = options;  // ← 100 → 80
    // ... 其余代码不变
```

同时在该函数开头（第 322 行 `section.classList.remove` 之后）增加取消旧动画的逻辑：

```javascript
// 清除该 section 的 exiting 状态
section.classList.remove('snap-page--exiting');

// 取消该 section 内所有子元素上正在运行的动画（防止与旧退出动画叠加）
const allChildren = inner.querySelectorAll(':scope > .scroll-reveal, :scope > .scroll-reveal-stagger > .scroll-reveal, :scope > *');
allChildren.forEach((child) => {
    child.getAnimations().forEach((a) => a.cancel());
});
```

- [ ] **Step 3: 删除 app.css 中的 CSS 退出动画规则**

删除 `app.css` 第 756-770 行（保留第 772 行注释和第 774-778 行的 `.hero-title` 特殊处理）：

```css
/* ── Snap 页面进出动画 ── */

/* 进入：内容从下方淡入（由 JS snapPageEnter 驱动） */

/* 当 section 正在离开时 Hero 缩小淡出 */
.snap-page--exiting .hero-title {
    opacity: 0;
    transform: translateY(20px) scale(0.95);
    transition: opacity 0.35s cubic-bezier(0, 0, 0.2, 1),
                transform 0.35s cubic-bezier(0, 0, 0.2, 1);
}
```

注意：Hero 标题的方向从原来仅 `scale(0.95)` 改为 `translateY(20px) scale(0.95)` 并加 `transition`。

- [ ] **Step 4: 禁用 Chrome 115+ view-timeline 冲突块**

在 `app.css` 第 2522 行，添加禁用注释：

```css
/* Chrome 115+: 使用 ViewTimeline 在吸附时触发动画
   ⚠ 暂禁用 — 与 JS scroll-snap 编排的 snapPageEnter/snapPageExit 冲突
     待 Chrome 原生 scroll-driven animations 成熟后迁移 */
/* @supports (animation-timeline: view()) {
    .snap-page .scroll-reveal {
        animation: snapPageEnter 0.6s cubic-bezier(0.16, 1, 0.3, 1) both;
        animation-timeline: view(block 80% 20%);
    }
    .snap-page .section-title {
        animation: snapPageEnter 0.5s cubic-bezier(0.16, 1, 0.3, 1) both;
        animation-timeline: view(block 85% 15%);
    }
} */
```

- [ ] **Step 5: 调整 app.js IntersectionObserver 回调中的退出/进入时序**

在 `app.js` 第 210-227 行，修改 IntersectionObserver 回调：

```javascript
if (maxRatio > 0 && maxIdx !== self.currentSection) {
    var prevSection = sections[self.currentSection];
    var nextSection = sections[maxIdx];

    // 1. 先触发旧 section 的退出动画
    if (prevSection && window.Anim && window.Anim.snapPageExit) {
        window.Anim.snapPageExit(prevSection);
    }

    // 2. 更新 currentSection
    self.currentSection = maxIdx;

    // 3. 触发新 section 的入场动画（snapPageEnter 内部会移除 exiting class）
    if (nextSection && window.Anim && window.Anim.snapPageEnter) {
        window.Anim.snapPageEnter(nextSection, { staggerMs: 80, duration: 500 });
    }
}
```

- [ ] **Step 6: 启动 UI 验证 S1 进出动画**

```bash
python scripts/run_ui.py --no-browser
# → http://127.0.0.1:8000
# 测试: 鼠标滚轮慢速滚动 S0↔S1↔S2，确认内容沿滚动方向平滑推移
# 测试: 快速连续滚两页，确认无动画残留/闪烁
# 测试: 键盘 ↑↓ 导航，确认动画一致
# 测试: 向上回滚时，旧页面内容正确重新入场（不残留 opacity:0）
```

- [ ] **Step 7: Commit**

```bash
git add modules/ui/static/js/animations.js modules/ui/static/css/app.css modules/ui/static/js/app.js
git commit -F .git-msg
```

`.git-msg` 内容：
```
fix(ui): 统一 S1 进出动画为向下推送方向

- snapPageExit: 方向改为 translateY(+30px) 向下推出
- snapPageEnter: stagger 从 100ms 改为 80ms，增加旧动画取消逻辑
- 删除 CSS .snap-page--exiting 退出动画规则（消除 CSS/JS 双动画竞争）
- 禁用 Chrome 115+ @supports (animation-timeline: view()) 块
- IntersectionObserver 回调中先 exit 再 enter，保证时序
- Hero 标题退出改为 translateY(20px) 与整体方向一致
```

---

### Task 3: 重构 S2 推理结果为一体化仪表盘卡片

**Files:**
- Modify: `modules/ui/static/index.html:698-804`（结果面板 DOM）
- Modify: `modules/ui/static/css/app.css:1436-1500`（结果布局样式）
- Modify: `modules/ui/static/css/app.css`（新增 `.result-dashboard`、`.result-metrics-row`、`.pipeline-summary`）

**Goal:** 将 6:4 左右分栏替换为单张一体化卡片，流水线完成后收缩为步骤摘要。

- [ ] **Step 1: 重构 index.html 结果面板 DOM**

将 `index.html` 第 698-804 行（`<template x-if="inferenceState === 'done' && resultData">` 内的整个结果面板）替换为：

```html
<template x-if="inferenceState === 'done' && resultData">
<div class="result-dashboard scroll-reveal" x-transition>
    <!-- 标题栏：模型名 + 数据集 + 徽章 -->
    <div class="result-dashboard-header">
        <div class="result-dashboard-title">
            <span class="result-dashboard-label">推理结果</span>
            <span class="result-dashboard-meta" x-text="resultData.model_name + ' · ' + selectedDataset"></span>
        </div>
        <span class="result-badge"
              :class="resultData.is_anomaly ? 'badge-anomaly' : 'badge-normal'"
              x-text="resultData.is_anomaly ? '异常' : '正常'"></span>
    </div>

    <!-- 对比滑块：全宽 -->
    <div class="result-dashboard-compare" x-data="imageCompare">
        <div class="compare-container">
            <img :src="resultData.image_b64" class="compare-image compare-original" alt="原图">
            <img :src="resultData.heatmap_b64" class="compare-image compare-heatmap"
                 :style="{ clipPath: 'inset(0 ' + (100 - sliderPos) + '% 0 0)' }" alt="热力图">
            <div class="compare-handle" :style="{ left: sliderPos + '%' }"
                 @mousedown="startDrag" @touchstart="startDrag">
                <div class="compare-handle-line"></div>
                <div class="compare-handle-grip">
                    <svg width="20" height="20" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <circle cx="10" cy="10" r="8" fill="#ffffff" stroke="#2997ff" stroke-width="2"/>
                        <path d="M7 10h6M10 7v6" stroke="#2997ff" stroke-width="1.5" stroke-linecap="round"/>
                    </svg>
                </div>
            </div>
            <input type="range" min="0" max="100" x-model="sliderPos" class="compare-range" aria-label="对比滑块">
        </div>
        <!-- 图例 overlay（左下角） -->
        <div class="heatmap-legend heatmap-legend--overlay">
            <div class="legend-label">异常得分</div>
            <div class="legend-bar-wrap">
                <div class="legend-bar"></div>
                <div class="legend-labels">
                    <span>1.0</span><span>0.5</span><span>0.0</span>
                </div>
            </div>
        </div>
    </div>

    <!-- 三列指标行 -->
    <div class="result-metrics-row">
        <div class="result-metric-card">
            <div class="metric-label">异常得分</div>
            <div class="metric-value result-score-value" x-text="resultData.score.toFixed(4)"></div>
            <div class="metric-bar">
                <div class="metric-fill"
                     :class="resultData.is_anomaly ? 'fill-anomaly' : 'fill-normal'"
                     :style="{ width: (resultData.score * 100) + '%' }"></div>
                <div class="metric-threshold"
                     :style="{ left: (resultData.threshold * 100) + '%' }"></div>
            </div>
        </div>
        <div class="result-metric-card">
            <div class="metric-label">置信度</div>
            <div class="metric-value result-confidence-value"
                  x-text="(resultData.confidence * 100).toFixed(1) + '%'"></div>
            <div class="metric-bar">
                <div class="metric-fill fill-confidence"
                     :style="{ width: (resultData.confidence * 100) + '%' }"></div>
            </div>
        </div>
        <div class="result-metric-card">
            <div class="metric-label">阈值 τ</div>
            <div class="metric-value" x-text="resultData.threshold.toFixed(3)"></div>
            <div class="metric-bar">
                <div class="metric-fill fill-threshold"
                     :style="{ width: (resultData.threshold * 100) + '%' }"></div>
            </div>
        </div>
    </div>

    <!-- 判决条 + 操作 -->
    <div class="result-dashboard-footer"
         :class="resultData.is_anomaly ? 'footer-anomaly' : 'footer-normal'">
        <span class="footer-verdict">
            <span class="verdict-dot" :class="resultData.is_anomaly ? 'dot-anomaly' : 'dot-normal'"
                  x-text="resultData.is_anomaly ? '●' : '●'"></span>
            得分 <b x-text="resultData.score.toFixed(4)"></b>
            <span x-text="resultData.is_anomaly ? ' > ' : ' ≤ '"></span>
            阈值 <b>τ = <span x-text="resultData.threshold.toFixed(3)"></span></b>
            &rarr; <b x-text="resultData.is_anomaly ? '异常' : '正常'"></b>
        </span>
        <button class="btn-reset" @click="resetInference()">重新上传</button>
    </div>

    <!-- 隐藏数据：供 JS 读取（tooltip + bbox） -->
    <img :src="resultData?.anomaly_map_b64"
         x-ref="anomalyMapData"
         x-show="false"
         @load="setupVisualInteractions()">
    <div x-ref="bboxData"
         :data-bboxes="JSON.stringify(resultData?.bboxes || [])"
         x-show="false"></div>
</div>
</template>
```

- [ ] **Step 2: 添加流水线摘要行（推理完成后显示）**

在 `index.html` 的 `.pipeline` div（第 579 行）之后、进度条之前，插入：

```html
<!-- 流水线收缩摘要（推理完成后显示） -->
<div class="pipeline-summary" x-show="inferenceState === 'done'" x-transition>
    <span class="pipeline-summary-step pipeline-summary-step--done">
        <span class="pipeline-summary-dot">●</span> 已上传
    </span>
    <span class="pipeline-summary-arrow">→</span>
    <span class="pipeline-summary-step pipeline-summary-step--done">
        <span class="pipeline-summary-dot">●</span>
        <span x-text="(models.find(function(m){ return m.key === selectedModel }) || {}).name || 'PatchCore'"></span>
    </span>
    <span class="pipeline-summary-arrow">→</span>
    <span class="pipeline-summary-step pipeline-summary-step--done">
        <span class="pipeline-summary-dot">●</span> 推理完成 ✓
    </span>
</div>
```

同时修改 `.pipeline` 的 `x-show`（第 579 行），加上对 done 状态的排除：

```html
<div class="pipeline" x-show="inferenceState === 'idle' || inferenceState === 'uploaded' || inferenceState === 'error'">
```

- [ ] **Step 3: 替换 app.css 中的结果面板样式**

删除 `app.css` 第 1436-1472 行（`.result-layout`、`.result-left`、`.result-right`、`.result-panel`、`@keyframes resultReveal`），替换为：

```css
/* ═══════════════════════════════════════════════════════════════════════════
   结果仪表盘卡片 — 一体化布局
   ═══════════════════════════════════════════════════════════════════════════ */
.result-dashboard {
    max-width: 780px;
    margin: 0 auto;
    background: var(--bg-system);
    border: 1px solid var(--sep-subtle);
    border-radius: var(--r-xl);
    padding: 28px;
    display: flex;
    flex-direction: column;
    gap: 20px;
    box-shadow: var(--shadow-md);
}

/* ── 标题栏 ── */
.result-dashboard-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.result-dashboard-title {
    display: flex;
    flex-direction: column;
    gap: 2px;
}

.result-dashboard-label {
    font-family: var(--font-display);
    font-size: 18px;
    font-weight: 600;
    color: var(--text);
    letter-spacing: -0.01em;
}

.result-dashboard-meta {
    font-size: 13px;
    color: var(--text-tertiary);
}

/* ── 对比滑块区 ── */
.result-dashboard-compare {
    position: relative;
}

/* ── 三列指标行 ── */
.result-metrics-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
}

.result-metric-card {
    background: var(--bg-secondary);
    border: 1px solid var(--sep-subtle);
    border-radius: var(--r-md);
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 8px;
}

.result-metric-card .metric-label {
    font-size: 11px;
    font-weight: 500;
    color: var(--text-tertiary);
    text-transform: uppercase;
    letter-spacing: 0.03em;
}

.result-metric-card .metric-value {
    font-family: var(--font-mono);
    font-size: 22px;
    font-weight: 700;
    color: var(--text);
}

/* ── 判决 + 操作底部栏 ── */
.result-dashboard-footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 16px;
    border-radius: var(--r-md);
    font-size: 13px;
}

.result-dashboard-footer.footer-anomaly {
    background: var(--bad-bg);
    color: var(--bad);
}

.result-dashboard-footer.footer-normal {
    background: var(--ok-bg);
    color: var(--ok);
}

.footer-verdict {
    display: flex;
    align-items: center;
    gap: 4px;
    color: var(--text);
}

.footer-verdict b {
    font-weight: 600;
    color: var(--text);
}

/* ── 图例 overlay（左下角）── */
.heatmap-legend--overlay {
    position: absolute;
    bottom: 12px;
    left: 12px;
    background: var(--bg-system);
    border: 1px solid var(--sep-subtle);
    border-radius: var(--r-sm);
    padding: 8px 12px;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    z-index: 5;
}

/* ── 流水线收缩摘要 ── */
.pipeline-summary {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 10px;
    padding: 16px 0;
    margin-bottom: 8px;
}

.pipeline-summary-step {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 13px;
    color: var(--text-tertiary);
}

.pipeline-summary-step--done {
    color: var(--text);
}

.pipeline-summary-dot {
    font-size: 10px;
    line-height: 1;
}

.pipeline-summary-step--done .pipeline-summary-dot {
    color: var(--ok);
}

.pipeline-summary-arrow {
    color: var(--text-tertiary);
    font-size: 14px;
}
```

- [ ] **Step 4: 删除 `.result-verdict` 旧样式（已移至 `.result-dashboard-footer`）**

在 `app.css` 中搜索 `.result-verdict` 相关样式并删除（约第 1620-1650 行区域）。同时删除 `.result-card` 旧样式块中与新结构冲突的部分（`.result-header`、`.result-metrics` 中的旧布局）。**注意：** 保留 `.result-badge`、`.badge-anomaly`、`.badge-normal`、`.metric-bar`、`.metric-fill`、`.metric-threshold`、`.fill-anomaly`、`.fill-normal`、`.fill-confidence`、`.verdict-dot`、`.dot-anomaly`、`.dot-normal` 等共用原子类。

- [ ] **Step 5: 启动 UI 验证 S2 仪表盘布局**

```bash
python scripts/run_ui.py --no-browser
# → http://127.0.0.1:8000
# 1. 上传图片 + 选择模型 + 推理
# 2. 观察流水线收缩为摘要行 → 仪表盘卡片从下方升入
# 3. 检查：标题栏、对比滑块、三列指标、判决条是否都在一张卡片内
# 4. 检查：亮/暗模式切换后样式正确
```

- [ ] **Step 6: Commit**

```bash
git add modules/ui/static/index.html modules/ui/static/css/app.css
git commit -F .git-msg
```

`.git-msg` 内容：
```
feat(ui): 重构 S2 推理结果为一体化仪表盘卡片

- 替换 6:4 左右分栏为单张 .result-dashboard 卡片
- 标题栏 + 全宽对比滑块 + 三列指标行 + 底部判决/操作
- 新增 .pipeline-summary 流水线收缩摘要（推理完成时显示）
- 图例移入滑块 overlay 左下角
- 删除旧的 .result-layout/.result-left/.result-right 布局
```

---

### Task 4: 重构 S3 四模型对比布局（色标方向 + 紧凑四列 + 摘要栏）

**Files:**
- Modify: `modules/ui/static/index.html:867-939`（`.compare-grid` 内的槽位 DOM）
- Modify: `modules/ui/static/index.html:846-864`（摘要栏 DOM）
- Modify: `modules/ui/static/css/app.css:2089-2398`（网格、槽位、色标、指标样式）

**Goal:** 色标从左侧竖线改为顶部横线，槽位 `min-height: auto`，hover 减弱，摘要栏压缩为单行。

- [ ] **Step 1: 修改槽位色标 — index.html 内联样式**

将 `index.html` 第 871-872 行的色标 div：

```html
<div class="compare-slot-accent"
     :style="{ background: mk === 'patchcore' ? '#2997ff' : mk === 'padim' ? '#30d158' : mk === 'fre' ? '#ff9f0a' : '#bf5af2' }"></div>
```

保持不变（样式由 CSS 控制方向）。

- [ ] **Step 2: 减少槽位指标行数 — 3 行 → 2 行**

在 `index.html` 第 918-933 行的指标区，删除阈值行（第三行 `.compare-metric`）：

```html
<div class="compare-metrics">
    <div class="compare-metric">
        <span class="compare-metric-label">得分</span>
        <span class="compare-metric-value"
              x-text="compareSlots[mk].data.score.toFixed(4)"></span>
    </div>
    <div class="compare-metric">
        <span class="compare-metric-label">置信度</span>
        <span class="compare-metric-value"
              x-text="(compareSlots[mk].data.confidence * 100).toFixed(1) + '%'"></span>
    </div>
</div>
```

（删除第三个 `.compare-metric` 即阈值行）

- [ ] **Step 3: 更新摘要栏为单行格式**

将 `index.html` 第 847-864 行的摘要栏替换为：

```html
<template x-if="summary">
<div class="compare-summary-row" x-transition>
    <span class="compare-summary-icon">&#127942;</span>
    <span class="compare-summary-best">
        最佳：<strong x-text="summary.best_name"></strong>
        <code x-text="summary.best_score?.toFixed(4)"></code>
    </span>
    <span class="compare-summary-sep">|</span>
    <template x-for="(r, i) in summary.ranking" :key="r.model">
        <span class="compare-summary-rank" x-show="i > 0">
            <span class="rank-hash">#</span><span x-text="i + 1"></span>
            <span class="rank-name" x-text="r.name"></span>
            <code x-text="r.score?.toFixed(4) ?? '—'"></code>
        </span>
    </template>
</div>
</template>
```

- [ ] **Step 4: 更新 app.css 中的槽位和色标样式**

替换 `app.css` 第 2089-2367 行：

```css
/* ── 四列网格 ── */
.compare-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
}

/* ── 槽位卡片 ── */
.compare-slot {
    position: relative;
    overflow: hidden;
    background: var(--bg-system);
    border: 1px solid var(--sep-subtle);
    border-radius: var(--r-lg);
    padding: 16px;
    transition: border-color var(--dur-normal) var(--ease-out),
                box-shadow var(--dur-normal) var(--ease-out),
                transform 0.25s var(--ease-out);
    display: flex;
    flex-direction: column;
    /* min-height 由内容自然撑高 */
}

/* ── 槽位 hover 减弱（避免四列同时跳动）── */
.compare-slot:hover {
    transform: translateY(-2px);
    border-color: var(--sep-strong);
    box-shadow: var(--shadow-md);
}

/* ── 槽位状态修饰符 ── */
.compare-slot--active {
    border-color: var(--accent);
    box-shadow: 0 0 20px var(--accent-glow);
}

.compare-slot--done {
    animation: compareDoneFlash 0.8s var(--ease-out) forwards;
    will-change: border-color, box-shadow;
}

.compare-slot--error {
    border-color: var(--bad);
    box-shadow: 0 0 12px rgba(255, 69, 58, 0.15);
}

@keyframes compareDoneFlash {
    0% {
        border-color: var(--accent);
        box-shadow: 0 0 20px var(--accent-glow);
    }
    20% {
        border-color: var(--warn);
        box-shadow: 0 0 16px rgba(255, 159, 10, 0.4);
    }
    40% {
        border-color: var(--accent);
        box-shadow: 0 0 20px var(--accent-glow);
    }
    60% {
        border-color: var(--warn);
        box-shadow: 0 0 16px rgba(255, 159, 10, 0.4);
    }
    80% {
        border-color: var(--ok);
        box-shadow: 0 0 12px rgba(48, 209, 88, 0.2);
    }
    100% {
        border-color: var(--ok);
        box-shadow: none;
    }
}

/* ── 槽位头部 ── */
.compare-slot-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
    min-height: 28px;
}

.compare-slot-header h3 {
    font-family: var(--font-display);
    font-size: 15px;
    font-weight: 600;
    color: var(--text);
    letter-spacing: -0.01em;
}

.compare-slot-badge {
    font-size: 11px;
    font-weight: 600;
    padding: 2px 10px;
    border-radius: 100px;
    letter-spacing: 0.02em;
    flex-shrink: 0;
}

/* ── 等待状态 ── */
.compare-slot-pending {
    flex: 1;
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 120px;
}

.compare-slot-placeholder {
    font-size: 13px;
    color: var(--text-tertiary);
}

/* ── 推理中状态 ── */
.compare-slot-active {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 14px;
    min-height: 120px;
}

/* ── 错误状态 ── */
.compare-slot-error {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 8px;
    text-align: center;
    font-size: 13px;
    color: var(--bad);
    padding: 12px;
    word-break: break-word;
    min-height: 100px;
}

/* ── 完成状态 ── */
.compare-slot-result {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 10px;
}

.compare-heatmap-wrap {
    position: relative;
    width: 100%;
    border-radius: var(--r-md);
    overflow: hidden;
    background: var(--bg-secondary);
}

.compare-slot .compare-heatmap {
    position: relative;
    display: block;
    width: 100%;
    max-height: 180px;
    object-fit: contain;
    border-radius: var(--r-md);
    background: var(--bg-secondary);
}

/* ── 紧凑指标 ── */
.compare-metrics {
    display: flex;
    flex-direction: column;
    gap: 6px;
}

.compare-metric {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
}

.compare-metric-label {
    font-size: 10px;
    font-weight: 500;
    color: var(--text-tertiary);
    letter-spacing: 0.02em;
    text-transform: uppercase;
}

.compare-metric-value {
    font-family: var(--font-mono);
    font-size: 15px;
    font-weight: 600;
    color: var(--text);
}

/* ── 槽位色标（顶部横线）── */
.compare-slot-accent {
    width: 100%;
    height: 3px;
    border-radius: 2px;
    margin-bottom: 10px;
    flex-shrink: 0;
    background: var(--algo-color, var(--accent));
}

/* ── 摘要栏（单行压缩）── */
.compare-summary-row {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 12px;
    padding: 10px 20px;
    background: var(--bg-secondary);
    border: 1px solid var(--sep-subtle);
    border-radius: var(--r-md);
    margin-bottom: 20px;
    font-size: 13px;
    flex-wrap: wrap;
}

.compare-summary-icon {
    font-size: 16px;
    flex-shrink: 0;
}

.compare-summary-best {
    color: var(--text);
    white-space: nowrap;
}

.compare-summary-best strong {
    font-weight: 600;
}

.compare-summary-best code {
    font-family: var(--font-mono);
    font-size: 12px;
    color: var(--accent);
    margin-left: 4px;
}

.compare-summary-sep {
    color: var(--sep-strong);
    font-size: 14px;
}

.compare-summary-rank {
    display: flex;
    align-items: center;
    gap: 3px;
    color: var(--text-secondary);
    white-space: nowrap;
}

.rank-hash {
    color: var(--text-tertiary);
    font-size: 11px;
}

.compare-summary-rank code {
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--text-secondary);
    margin-left: 2px;
}
```

- [ ] **Step 5: 在响应式断点中更新 S3 相关规则**

在 `app.css` 的 `@media (max-width: 768px)` 中（约第 2585 行），确保：

```css
.compare-grid { grid-template-columns: repeat(2, 1fr); gap: 10px; }
.compare-shared-img { max-height: 160px; }
```

在 `@media (max-width: 480px)` 中（约第 2659 行）：

```css
.compare-grid { grid-template-columns: 1fr; }
.compare-shared-img { max-height: 140px; }
```

- [ ] **Step 6: 清理旧 CSS — 删除已废弃的规则**

- 删除 `app.css` 中原 `.compare-shared-image` 的旧位置样式（第 2319-2350 行，已在 Task 1 中替换）
- 删除 `app.css` 中原 `.compare-slot-accent` 的旧 absolute 定位样式（已被新规则覆盖）
- 删除 `app.css` 中原 `.compare-summary` 样式（替换为 `.compare-summary-row`）
- 删除 `app.css` 中原 `.compare-ranking`、`.compare-rank-item`、`.rank-num` 等旧摘要样式

**注意：** 搜索确认上述选择器不再被 HTML 引用后再删除，避免遗漏引用导致样式丢失。

- [ ] **Step 7: 启动 UI 验证 S3 布局**

```bash
python scripts/run_ui.py --no-browser
# → http://127.0.0.1:8000
# 1. 先在 S2 完成一次推理（上传图片）
# 2. 滚动到 S3，点击「四模型同时对比」
# 3. 检查：共享原图尺寸正常 (160-200px)、色标为顶部横线、四列紧凑、摘要栏单行
# 4. hover 各槽位确认反馈克制（仅 -2px 平移）
# 5. 亮/暗模式切换
```

- [ ] **Step 8: Commit**

```bash
git add modules/ui/static/index.html modules/ui/static/css/app.css
git commit -F .git-msg
```

`.git-msg` 内容：
```
feat(ui): 重构 S3 四模型对比布局 — 紧凑四列 + 顶横色标

- 色标从左侧竖线(absolute)改为顶部横线(static)，算法辨识度更高
- 槽位 min-height:auto，内容自然撑高
- 槽位指标从3行减为2行（得分+置信度）
- hover效果从 translateY(-4px)+scale(1.02) 减弱为 translateY(-2px)
- 摘要栏压缩为单行 flex row
- 响应式断点适配 2列(平板)/1列(手机)
```

---

### Task 5: 响应式验证 + 边界情况检查

**Files:** 无新增修改，仅验证

**Goal:** 确保所有变更在移动端、亮/暗双模式、reduced-motion 下正确。

- [ ] **Step 1: 平板断点验证 (768px)**

```bash
python scripts/run_ui.py --no-browser
# → 浏览器 DevTools: 设为 768×1024
# 检查: pipeline 堆叠为单列、result-metrics-row 从3列变堆叠、compare-grid 为2列
```

- [ ] **Step 2: 手机断点验证 (480px)**

```bash
# → 浏览器 DevTools: 设为 375×812 (iPhone)
# 检查: compare-grid 变1列、共享原图 max-height 140px、导航栏无溢出
```

- [ ] **Step 3: reduced-motion 验证**

在浏览器 Console 中模拟：
```js
// 开启 reduced-motion 模拟
// Chrome DevTools → Rendering → Emulate CSS media feature prefers-reduced-motion: reduce
```
验证：滚动时无动画，内容直接显示最终状态。

- [ ] **Step 4: 亮色模式验证**

切换到亮色模式，检查所有页面（S0/S1/S2）的可读性和对比度。

- [ ] **Step 5: 如果发现响应式问题，修复并 amend**

```bash
git add modules/ui/static/css/app.css
git commit --amend --no-edit  # 追加到最后一个 commit
```

---

### 实施顺序建议

```
Task 1 (64px bug fix) → Task 2 (animation) → Task 3 (S2 dashboard) → Task 4 (S3 compare) → Task 5 (verify)
```

Task 1 和 Task 2 可以独立并行（它们改不同的 CSS 段和不同的文件区域），但建议按顺序做以便逐个验证。Task 3 和 Task 4 均依赖 Task 1 的 CSS 变更（共享 CSS 变量和组件），应在其后执行。
