# Apple 风格前端重设计 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不动后端与 SSE 推理逻辑的前提下，把现有 FastAPI + Alpine.js 三页 Snap SPA 升级为 Apple 产品介绍页质感的 Cinematic Pro 风格。

**Architecture:** 通过新增独立 `apple-redesign.css` 与 `hero-visual.js` 承载新视觉与 Hero 动画，对 `index.html` 做最小类名/结构改动，复用现有 `app.css` 中的组件样式。动画揭示逻辑注入 `animations.js`，并在 `app.js` / `compare.js` 的结果回调中触发。

**Tech Stack:** HTML5, CSS3 (custom properties, grid, flex, backdrop-filter), Alpine.js, vanilla JS (WAAPI), FastAPI static files.

---

## 文件结构

| 文件 | 责任 |
|------|------|
| `modules/ui/static/css/apple-redesign.css` | 新增 Apple 风格视觉层：变量覆盖、Hero、Bento、玻璃工作台/仪表盘、对比墙、响应式 |
| `modules/ui/static/js/hero-visual.js` | Hero SVG 管线描边动画控制，暴露 `HeroVisual.play()/stop()` |
| `modules/ui/static/index.html` | 引入新资源；调整 Section 0/1/2 的类名与结构 |
| `modules/ui/static/js/animations.js` | 新增结果仪表盘/对比墙揭示动画函数 |
| `modules/ui/static/js/app.js` | 单模型推理结果出现后触发 `Anim.resultReveal()` |
| `modules/ui/static/js/compare.js` | 四模型对比全部完成后触发 `Anim.compareReveal()` |
| `tests/test_ui_static.py` | 静态资源回归测试：检查新文件存在、HTML 引用、关键结构类名 |

---

## Task 1: 创建静态资源回归测试

**Files:**
- Create: `tests/test_ui_static.py`
- Test: `tests/test_ui_static.py`

- [ ] **Step 1: 编写测试**

```python
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC = PROJECT_ROOT / "modules" / "ui" / "static"
HTML = STATIC / "index.html"


def test_apple_redesign_css_exists():
    assert (STATIC / "css" / "apple-redesign.css").exists()


def test_hero_visual_js_exists():
    assert (STATIC / "js" / "hero-visual.js").exists()


def test_index_html_links_redesign_assets():
    text = HTML.read_text(encoding="utf-8")
    assert "/static/css/apple-redesign.css" in text
    assert "/static/js/hero-visual.js" in text


def test_index_html_has_new_structure():
    text = HTML.read_text(encoding="utf-8")
    assert "hero-visual" in text
    assert "bento-grid" in text
    assert "workbench" in text
    assert "compare-wall" in text
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_ui_static.py -v`

Expected: 4 FAILED（文件不存在 / 类名未找到）

- [ ] **Step 3: 提交测试文件**

```bash
git add tests/test_ui_static.py
git commit -m "test(ui): 添加 Apple 重设计静态资源回归测试"
```

---

## Task 2: 创建 Apple 重设计样式表

**Files:**
- Create: `modules/ui/static/css/apple-redesign.css`
- Modify: `modules/ui/static/index.html`（引入，见 Task 4）

- [ ] **Step 1: 创建 CSS 文件**

Create `modules/ui/static/css/apple-redesign.css` with:

```css
/* ==========================================================================
   Apple 风格重设计覆盖层 — Cinematic Pro
   与现有 app.css 共存，仅通过新增类名和更具体选择器生效
   ========================================================================== */

/* ── 变量精细覆盖 ── */
:root {
    --bg-root: #0a0a0b;
    --bg-system: rgba(255, 255, 255, 0.055);
    --bg-secondary: rgba(255, 255, 255, 0.09);
    --bg-tertiary: rgba(255, 255, 255, 0.13);

    --sep-subtle: rgba(255, 255, 255, 0.07);
    --sep-default: rgba(255, 255, 255, 0.11);
    --sep-strong: rgba(255, 255, 255, 0.17);

    --accent: #2997ff;
    --accent-hover: #47a9ff;
    --accent-pressed: #0070d6;
    --accent-dim: rgba(41, 151, 255, 0.14);
    --accent-glow: rgba(41, 151, 255, 0.30);

    --text: rgba(255, 255, 255, 0.92);
    --text-secondary: rgba(255, 255, 255, 0.60);
    --text-tertiary: rgba(255, 255, 255, 0.34);

    --r-sm: 10px;
    --r-md: 14px;
    --r-lg: 18px;
    --r-xl: 24px;

    --ease-out-expo: cubic-bezier(0.16, 1, 0.3, 1);
    --ease-spring: cubic-bezier(0.22, 0.8, 0.3, 1.15);
}

html[data-theme="light"] {
    --bg-root: #f5f5f7;
    --bg-system: rgba(255, 255, 255, 0.62);
    --bg-secondary: rgba(255, 255, 255, 0.42);
    --bg-tertiary: rgba(255, 255, 255, 0.28);

    --sep-subtle: rgba(0, 0, 0, 0.05);
    --sep-default: rgba(0, 0, 0, 0.09);
    --sep-strong: rgba(0, 0, 0, 0.15);

    --text: rgba(0, 0, 0, 0.88);
    --text-secondary: rgba(0, 0, 0, 0.55);
    --text-tertiary: rgba(0, 0, 0, 0.32);

    --accent-dim: rgba(41, 151, 255, 0.10);
    --accent-glow: rgba(41, 151, 255, 0.14);
}

/* 降低背景装饰强度 */
.snap-container::before {
    opacity: 0.06;
}
html[data-theme="light"] .snap-container::before {
    opacity: 0.04;
}
body::before {
    opacity: 0.22;
}

/* ── Hero ── */
.hero {
    padding: 0 0 28px;
    position: relative;
    z-index: 1;
}

.hero-title {
    font-size: 88px;
    font-weight: 700;
    letter-spacing: -0.04em;
    line-height: 1.03;
    text-shadow: 0 0 100px var(--accent-glow), 0 0 200px rgba(41, 151, 255, 0.10);
    margin-bottom: 20px;
}

.hero-subtitle {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    padding: 10px 18px;
    background: var(--bg-system);
    border: 1px solid var(--sep-default);
    border-radius: 100px;
    backdrop-filter: blur(20px) saturate(150%);
    -webkit-backdrop-filter: blur(20px) saturate(150%);
    font-size: 16px;
    color: var(--text-secondary);
}

.hero-subtitle span {
    display: inline-flex;
    align-items: center;
    gap: 6px;
}
.hero-subtitle span:not(:last-child)::after {
    content: '';
    width: 4px;
    height: 4px;
    border-radius: 50%;
    background: var(--text-tertiary);
}

/* ── Hero Visual ── */
.hero-visual {
    width: 100%;
    max-width: 720px;
    margin: 40px auto 0;
    aspect-ratio: 3 / 1;
}
.hero-visual svg {
    width: 100%;
    height: 100%;
    display: block;
}
.hero-visual .hv-path {
    fill: none;
    stroke: url(#hvGrad);
    stroke-width: 2.5;
    stroke-linecap: round;
    stroke-dasharray: 900;
    stroke-dashoffset: 900;
}
.hero-visual.is-playing .hv-path {
    animation: hvDraw 2.2s var(--ease-out-expo) forwards, hvPulse 2.5s 2.2s ease-in-out infinite;
}
.hero-visual .hv-node rect {
    fill: var(--bg-secondary);
    stroke: var(--sep-default);
    stroke-width: 1.5;
    transition: stroke 0.3s var(--ease-out), fill 0.3s var(--ease-out);
}
.hero-visual.is-playing .hv-node rect {
    animation: hvNodePulse 2.5s 2.2s ease-in-out infinite;
}
.hero-visual .hv-label {
    font-family: var(--font-body);
    font-size: 13px;
    fill: var(--text-secondary);
    text-anchor: middle;
}

@keyframes hvDraw {
    to { stroke-dashoffset: 0; }
}
@keyframes hvPulse {
    0%, 100% { opacity: 0.55; }
    50% { opacity: 1; }
}
@keyframes hvNodePulse {
    0%, 100% { stroke: var(--sep-default); fill: var(--bg-secondary); }
    50% { stroke: var(--accent); fill: var(--accent-dim); }
}

/* ── Bento 算法卡片 ── */
.bento-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 24px;
    margin-top: 44px;
}

.bento-card {
    position: relative;
    background: var(--bg-system);
    border: 1px solid var(--sep-subtle);
    border-radius: var(--r-lg);
    padding: 28px;
    overflow: hidden;
    backdrop-filter: blur(18px) saturate(150%);
    -webkit-backdrop-filter: blur(18px) saturate(150%);
    transition: transform 0.35s var(--ease-spring), box-shadow 0.35s var(--ease-out), border-color 0.35s var(--ease-out);
}
.bento-card:hover {
    transform: translateY(-5px) scale(1.012);
    border-color: var(--sep-default);
    box-shadow: 0 28px 70px rgba(0, 0, 0, 0.35), 0 0 44px var(--accent-dim);
}

.bento-card .algo-card-accent {
    left: 0;
    top: 18px;
    bottom: 18px;
    width: 3px;
    border-radius: 0 3px 3px 0;
    background: var(--algo-color, var(--accent));
    transition: box-shadow 0.35s var(--ease-out);
}
.bento-card:hover .algo-card-accent {
    box-shadow: 0 0 20px var(--algo-color, var(--accent)), 0 0 40px var(--algo-color, var(--accent));
}

.bento-card h3 {
    font-size: 22px;
    font-weight: 600;
    margin-bottom: 10px;
}
.bento-card p {
    font-size: 14px;
    color: var(--text-secondary);
    line-height: 1.55;
    margin-bottom: 4px;
}
.bento-card .algo-card-kicker {
    font-size: 13px;
    color: var(--text-tertiary);
    font-style: italic;
}
.bento-card .flowchart-svg {
    margin-top: 18px;
    width: 100%;
    height: auto;
    max-height: 210px;
}

/* ── 工作台 ── */
.workbench {
    max-width: 760px;
    margin: 0 auto 32px;
}
.workbench-row {
    display: grid;
    grid-template-columns: 1.2fr 1fr auto;
    gap: 16px;
    align-items: start;
}

.pipeline.workbench-row .pipeline-step {
    background: var(--bg-system);
    border: 1px solid var(--sep-subtle);
    border-radius: var(--r-md);
    backdrop-filter: blur(14px) saturate(140%);
    -webkit-backdrop-filter: blur(14px) saturate(140%);
}

/* Drop Zone 强化 */
.upload-zone {
    border: 2px dashed var(--sep-default);
    border-radius: var(--r-lg);
    min-height: 220px;
    background: var(--bg-system);
    transition: border-color 0.3s var(--ease-out), background 0.3s var(--ease-out), box-shadow 0.3s var(--ease-out);
}
.upload-zone:hover,
.upload-zone.is-dragover {
    border-color: var(--accent);
    border-style: solid;
    background: var(--accent-dim);
    box-shadow: 0 0 30px var(--accent-glow);
}
.upload-zone.is-dragover {
    animation: dropPulse 1.2s ease-in-out infinite;
}
@keyframes dropPulse {
    0%, 100% { box-shadow: 0 0 20px var(--accent-glow); }
    50% { box-shadow: 0 0 45px var(--accent-glow); }
}

/* ── 结果仪表盘 ── */
.result-dashboard {
    max-width: 920px;
    margin: 0 auto;
    background: var(--bg-system);
    border: 1px solid var(--sep-subtle);
    border-radius: var(--r-xl);
    padding: 32px;
    backdrop-filter: blur(22px) saturate(160%);
    -webkit-backdrop-filter: blur(22px) saturate(160%);
    box-shadow: 0 28px 80px rgba(0, 0, 0, 0.38);
}

/* 指标条动画初始态 */
.result-dashboard .metric-fill {
    width: 0 !important;
    transition: width 0.9s var(--ease-out-expo);
}
.result-dashboard.is-revealed .metric-fill {
    width: var(--metric-fill-width, 0) !important;
}

/* ── 对比墙 ── */
.compare-wall {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 20px;
}
.compare-wall .compare-slot {
    background: var(--bg-system);
    border: 1px solid var(--sep-subtle);
    border-radius: var(--r-lg);
    padding: 20px;
    backdrop-filter: blur(18px) saturate(150%);
    -webkit-backdrop-filter: blur(18px) saturate(150%);
    transition: transform 0.3s var(--ease-spring), box-shadow 0.3s var(--ease-out);
}
.compare-wall .compare-slot:hover {
    transform: translateY(-3px);
    box-shadow: 0 18px 50px rgba(0, 0, 0, 0.30);
}
.compare-wall .compare-heatmap-wrap {
    border-radius: var(--r-md);
    overflow: hidden;
    background: var(--bg-secondary);
    margin-bottom: 16px;
}
.compare-wall .compare-heatmap {
    position: relative;
    display: block;
    width: 100%;
    max-height: 280px;
    object-fit: contain;
}

/* ── 响应式 ── */
@media (max-width: 768px) {
    .hero-title { font-size: 52px; }
    .hero-subtitle { font-size: 14px; gap: 8px; padding: 8px 14px; }
    .bento-grid { grid-template-columns: 1fr; margin-top: 32px; }
    .bento-card { padding: 22px; }
    .workbench-row { grid-template-columns: 1fr; }
    .result-dashboard { padding: 20px; }
    .compare-wall { grid-template-columns: 1fr; }
}
@media (max-width: 480px) {
    .hero-title { font-size: 38px; }
    .bento-card { padding: 18px; }
    .bento-card h3 { font-size: 18px; }
}
```

- [ ] **Step 2: 提交样式文件**

```bash
git add modules/ui/static/css/apple-redesign.css
git commit -m "feat(ui): 添加 Apple 风格重设计样式层"
```

---

## Task 3: 创建 Hero 视觉动画脚本

**Files:**
- Create: `modules/ui/static/js/hero-visual.js`

- [ ] **Step 1: 创建 JS 文件**

Create `modules/ui/static/js/hero-visual.js` with:

```javascript
/**
 * Hero 检测管线 SVG 动画控制
 *
 * 通过切换 .hero-visual 的 .is-playing 类触发 CSS stroke-dashoffset 描边动画。
 */
(function () {
    function getVisual() {
        return document.querySelector('.hero-visual');
    }

    function play() {
        var el = getVisual();
        if (!el) return;
        el.classList.remove('is-playing');
        // 强制重排以重新触发 CSS 动画
        void el.offsetWidth;
        el.classList.add('is-playing');
    }

    function stop() {
        var el = getVisual();
        if (el) el.classList.remove('is-playing');
    }

    window.HeroVisual = { play: play, stop: stop };

    // Alpine 渲染完成后播放一次
    document.addEventListener('alpine:initialized', function () {
        setTimeout(play, 300);
    });
})();
```

- [ ] **Step 2: 提交脚本**

```bash
git add modules/ui/static/js/hero-visual.js
git commit -m "feat(ui): 添加 Hero 检测管线 SVG 动画控制器"
```

---

## Task 4: 在 index.html 中引入新资源

**Files:**
- Modify: `modules/ui/static/index.html`

- [ ] **Step 1: 添加 CSS 链接**

Old:
```html
    <!-- 主样式表 -->
    <link rel="stylesheet" href="/static/css/app.css">
    <link rel="stylesheet" href="/static/css/flowchart.css">
```

New:
```html
    <!-- 主样式表 -->
    <link rel="stylesheet" href="/static/css/app.css">
    <link rel="stylesheet" href="/static/css/flowchart.css">
    <link rel="stylesheet" href="/static/css/apple-redesign.css">
```

- [ ] **Step 2: 添加 JS 脚本**

Old:
```html
    <script src="/static/js/animations.js"></script>
    <script src="/static/js/cursor-glow.js"></script>
    <script src="/static/js/inference.js"></script>
```

New:
```html
    <script src="/static/js/animations.js"></script>
    <script src="/static/js/hero-visual.js"></script>
    <script src="/static/js/cursor-glow.js"></script>
    <script src="/static/js/inference.js"></script>
```

- [ ] **Step 3: 提交**

```bash
git add modules/ui/static/index.html
git commit -m "feat(ui): 在 index.html 中引入重设计资源"
```

---

## Task 5: 重构 Section 0 Hero 区域

**Files:**
- Modify: `modules/ui/static/index.html`

- [ ] **Step 1: 替换 Hero 标题与副标题**

Old:
```html
                <div class="hero">
                    <h1 class="hero-title scroll-reveal">工业缺陷检测</h1>
                    <p class="hero-subtitle scroll-reveal">四种无监督算法 &middot; 实时推理 &middot; 像素级定位</p>
                </div>
```

New:
```html
                <div class="hero">
                    <h1 class="hero-title scroll-reveal">工业缺陷检测</h1>
                    <p class="hero-subtitle scroll-reveal">
                        <span>无监督</span><span>像素级</span><span>实时推理</span>
                    </p>
                    <div class="hero-visual scroll-reveal" aria-hidden="true">
                        <svg viewBox="0 0 720 240" xmlns="http://www.w3.org/2000/svg">
                            <defs>
                                <linearGradient id="hvGrad" x1="0" y1="0" x2="720" y2="0">
                                    <stop offset="0%" stop-color="#2997ff" stop-opacity="0.55"/>
                                    <stop offset="100%" stop-color="#2997ff" stop-opacity="0.05"/>
                                </linearGradient>
                            </defs>
                            <path class="hv-path" d="M 60 120 L 220 120 L 260 80 L 420 80 L 460 120 L 660 120"/>
                            <g class="hv-node"><rect x="20" y="90" width="80" height="60" rx="12"/></g>
                            <text class="hv-label" x="60" y="125">输入</text>
                            <g class="hv-node"><rect x="180" y="90" width="80" height="60" rx="12"/></g>
                            <text class="hv-label" x="220" y="125">特征</text>
                            <g class="hv-node"><rect x="340" y="50" width="80" height="60" rx="12"/></g>
                            <text class="hv-label" x="380" y="85">得分</text>
                            <g class="hv-node"><rect x="620" y="90" width="80" height="60" rx="12"/></g>
                            <text class="hv-label" x="660" y="125">热力图</text>
                        </svg>
                    </div>
                </div>
```

- [ ] **Step 2: 为算法网格和卡片添加 Bento 类名**

Old (grid opener):
```html
                <!-- 2x2 算法流程图卡片网格 -->
                <div class="algo-grid scroll-reveal-stagger">
```

New:
```html
                <!-- 2x2 算法流程图卡片网格 -->
                <div class="algo-grid bento-grid scroll-reveal-stagger">
```

Old (PatchCore card):
```html
                    <div class="algo-card algo-card--patchcore scroll-reveal">
```

New:
```html
                    <div class="algo-card bento-card algo-card--patchcore scroll-reveal">
```

Repeat for PaDiM, FRE, DRAEM cards:
- `algo-card algo-card--padim scroll-reveal` → `algo-card bento-card algo-card--padim scroll-reveal`
- `algo-card algo-card--fre scroll-reveal` → `algo-card bento-card algo-card--fre scroll-reveal`
- `algo-card algo-card--draem scroll-reveal` → `algo-card bento-card algo-card--draem scroll-reveal`

- [ ] **Step 3: 提交**

```bash
git add modules/ui/static/index.html
git commit -m "feat(ui): Section 0 Hero 升级为电影感主视觉 + Bento 卡片"
```

---

## Task 6: 重构 Section 1 推理工作台

**Files:**
- Modify: `modules/ui/static/index.html`

- [ ] **Step 1: 用 workbench 包裹流水线**

Old (pipeline opener):
```html
                    <!-- 流水线：上传 → 选择 → 推理（idle / uploaded / error 状态显示） -->
                    <div class="pipeline" x-show="inferenceState === 'idle' || inferenceState === 'uploaded' || inferenceState === 'error'">
```

New:
```html
                    <!-- 流水线：上传 → 选择 → 推理（idle / uploaded / error 状态显示） -->
                    <div class="workbench">
                    <div class="pipeline workbench-row" x-show="inferenceState === 'idle' || inferenceState === 'uploaded' || inferenceState === 'error'">
```

Old (pipeline closer — the line after the final `</div>` of pipeline, before progress bar):
```html
                    </div>

                    <!-- 进度条 -->
                    <div class="progress-bar" x-show="inferenceState === 'loading' || inferenceState === 'inferring'">
```

New:
```html
                    </div>
                    </div>

                    <!-- 进度条 -->
                    <div class="progress-bar" x-show="inferenceState === 'loading' || inferenceState === 'inferring'">
```

- [ ] **Step 2: 提交**

```bash
git add modules/ui/static/index.html
git commit -m "feat(ui): Section 1 推理区添加玻璃工作台包装"
```

---

## Task 7: 重构 Section 2 对比墙

**Files:**
- Modify: `modules/ui/static/index.html`

- [ ] **Step 1: 为四模型网格添加 compare-wall 类名**

Old:
```html
                <!-- 四列网格（仅热力图，无独立原图） -->
                <div class="compare-grid">
```

New:
```html
                <!-- 四列网格（仅热力图，无独立原图） -->
                <div class="compare-grid compare-wall">
```

- [ ] **Step 2: 提交**

```bash
git add modules/ui/static/index.html
git commit -m "feat(ui): Section 2 四模型对比升级为 2x2 对比墙"
```

---

## Task 8: 增强 animations.js 的揭示动画

**Files:**
- Modify: `modules/ui/static/js/animations.js`

- [ ] **Step 1: 在 initAllAnimations 之前添加揭示函数**

Old:
```javascript
/**
 * 全局初始化：在 Alpine 渲染完成后调用，触发滚动淡入动画。
 */
function initAllAnimations() {
    Anim.initScrollReveal();
}
window.initAllAnimations = initAllAnimations;
```

New:
```javascript
/**
 * 结果仪表盘揭示动画
 * 触发卡片淡入，并将指标条宽度从 0 动画到目标值。
 */
Anim.resultReveal = function (container) {
    if (!container) return;
    if (_prefersReducedMotion) {
        container.classList.add('is-revealed');
        return;
    }

    // 先重置
    container.classList.remove('is-revealed');

    // 设置指标条目标宽度
    container.querySelectorAll('.metric-fill').forEach(function (bar) {
        var target = bar.style.width;
        if (target) {
            bar.style.setProperty('--metric-fill-width', target);
            bar.style.width = '0';
        }
    });

    // 容器淡入
    var anim = container.animate(
        [{ opacity: 0, transform: 'translateY(20px) scale(0.98)' },
         { opacity: 1, transform: 'translateY(0) scale(1)' }],
        { duration: 500, easing: 'cubic-bezier(0.16, 1, 0.3, 1)', fill: 'both' }
    );

    anim.finished.then(function () {
        container.classList.add('is-revealed');
    });
};

/**
 * 四模型对比墙揭示动画
 * 对墙内每个 compare-slot 使用 WAAPI 逐个入场，不影响 pending/active 状态的可见性。
 */
Anim.compareReveal = function (container) {
    if (!container) return;
    var slots = container.querySelectorAll('.compare-slot');
    if (_prefersReducedMotion) {
        slots.forEach(function (s) { s.style.opacity = '1'; s.style.transform = 'none'; });
        return;
    }
    slots.forEach(function (slot, i) {
        slot.style.opacity = '0';
        slot.style.transform = 'translateY(24px) scale(0.98)';
        slot.animate(
            [{ opacity: 0, transform: 'translateY(24px) scale(0.98)' },
             { opacity: 1, transform: 'translateY(0) scale(1)' }],
            { duration: 500, delay: i * 80, easing: 'cubic-bezier(0.16, 1, 0.3, 1)', fill: 'forwards' }
        );
    });
};

/**
 * 全局初始化：在 Alpine 渲染完成后调用，触发滚动淡入动画。
 */
function initAllAnimations() {
    Anim.initScrollReveal();
}
window.initAllAnimations = initAllAnimations;
```

- [ ] **Step 2: 提交**

```bash
git add modules/ui/static/js/animations.js
git commit -m "feat(ui): 添加结果仪表盘与对比墙揭示动画"
```

---

## Task 9: 在 app.js 中触发结果仪表盘揭示

**Files:**
- Modify: `modules/ui/static/js/app.js`

- [ ] **Step 1: 在 onResult 回调中调用 resultReveal**

Old:
```javascript
                    onResult: function (data) {
                        self.resultData = data;
                        self.inferenceState = 'done';
                        // 下一帧触发数字滚动动画 + 可视化交互 + 滚动淡入
                        self.$nextTick(function () {
                            self.animateNumbers();
                            self.setupVisualInteractions();
                            // 重新触发 scroll-reveal 以捕获新出现的结果面板元素
                            setTimeout(function () {
                                if (window.initAllAnimations) window.initAllAnimations();
                            }, 100);
                        });
                    },
```

New:
```javascript
                    onResult: function (data) {
                        self.resultData = data;
                        self.inferenceState = 'done';
                        // 下一帧触发数字滚动动画 + 可视化交互 + 滚动淡入
                        self.$nextTick(function () {
                            self.animateNumbers();
                            self.setupVisualInteractions();
                            // Apple 风格结果仪表盘揭示动画
                            setTimeout(function () {
                                var dashboard = document.querySelector('.result-dashboard');
                                if (dashboard && window.Anim && window.Anim.resultReveal) {
                                    window.Anim.resultReveal(dashboard);
                                }
                            }, 80);
                            // 重新触发 scroll-reveal 以捕获新出现的结果面板元素
                            setTimeout(function () {
                                if (window.initAllAnimations) window.initAllAnimations();
                            }, 100);
                        });
                    },
```

- [ ] **Step 2: 提交**

```bash
git add modules/ui/static/js/app.js
git commit -m "feat(ui): 单模型推理结果触发仪表盘揭示动画"
```

---

## Task 10: 在 compare.js 中触发对比墙揭示

**Files:**
- Modify: `modules/ui/static/js/compare.js`

- [ ] **Step 1: 在 onDone 回调中调用 compareReveal**

Old:
```javascript
                    onDone: function () {
                        self.compareRunning = false;
                        self.compareDone = true;
                    },
```

New:
```javascript
                    onDone: function () {
                        self.compareRunning = false;
                        self.compareDone = true;
                        self.$nextTick(function () {
                            setTimeout(function () {
                                var wall = document.querySelector('.compare-wall');
                                if (wall && window.Anim && window.Anim.compareReveal) {
                                    window.Anim.compareReveal(wall);
                                }
                            }, 120);
                        });
                    },
```

- [ ] **Step 2: 提交**

```bash
git add modules/ui/static/js/compare.js
git commit -m "feat(ui): 四模型对比完成后触发对比墙揭示动画"
```

---

## Task 11: 运行回归测试

**Files:**
- Test: `tests/test_ui_static.py`

- [ ] **Step 1: 运行 pytest**

Run: `python -m pytest tests/test_ui_static.py -v`

Expected: 4 PASSED

- [ ] **Step 2: 运行完整测试套件**

Run: `python -m pytest tests/ -v`

Expected: 全部通过

- [ ] **Step 3: 提交**

```bash
git add tests/test_ui_static.py
git commit -m "test(ui): 静态资源回归测试通过"
```

---

## Task 12: 手动视觉验证

**Files:** None

- [ ] **Step 1: 启动 FastAPI UI**

Run: `python scripts/run_ui.py --no-browser`

- [ ] **Step 2: 浏览器打开 http://127.0.0.1:8000 检查以下项目**

- [ ] Section 0 Hero 显示超大标题、胶囊副标题、SVG 管线动画自动播放
- [ ] 向下滚动到 Section 0 Bento 卡片，4 张卡片 stagger 入场
- [ ] 继续滚动到 Section 1，上传图片，选择模型，开始推理
- [ ] 推理完成后结果仪表盘以玻璃卡片形式出现，指标条从 0 填充，数字滚动
- [ ] 对比滑块可正常拖动，bbox overlay 位置正确
- [ ] 滚动到 Section 2，点击「四模型同时对比」
- [ ] 四张热力图以 2×2 对比墙呈现，逐个入场
- [ ] 全部完成后出现排名摘要栏
- [ ] 切换亮/暗主题，所有元素颜色正常
- [ ] 移动端模拟器（iPhone 尺寸）下三页可滚动、标题不溢出、对比墙变为单列

- [ ] **Step 3: 截图保存到 `.superpowers/brainstorm/` 或项目 docs（可选）**

- [ ] **Step 4: 如发现问题，回到对应 Task 修复；如全部通过，提交**

```bash
git commit -m "feat(ui): Apple 风格 Cinematic Pro 前端升级完成并手动验证"
```

---

## Self-Review Checklist

- [ ] **Spec coverage**: 每个 design.md 章节都有对应 Task
  - Hero 电影感 + Bento → Task 5
  - 玻璃工作台 + 仪表盘 → Task 6 + Task 8/9
  - 2×2 对比墙 → Task 7 + Task 10
  - 动效 → Task 3 + Task 8/9/10
  - 双主题 → CSS 变量在 Task 2
  - 测试 → Task 1/11/12
- [ ] **Placeholder scan**: 无 TBD/TODO/"稍后实现"
- [ ] **Type consistency**: `Anim.resultReveal` / `Anim.compareReveal` 在 Task 8 定义，Task 9/10 调用，名称一致
- [ ] **文件路径**: 所有路径均为绝对项目内路径
