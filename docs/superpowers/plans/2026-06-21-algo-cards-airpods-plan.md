# 首页算法卡片 AirPods Pro 风格改造规划（迭代版）

## 1. 背景与目标

将当前首页（Section 0）的**四个算法 2×2 Bento 网格**改造成**横向滚动卡片轮播**：每次主要突出展示一种算法，重点强化**卡片之间的过渡动画**，并优化 SVG 流程图使其更大、对关键算法步骤更细化。

参考标杆：https://www.apple.com/airpods-pro/

---

## 2. 关于 Bento 网格的说明

**Bento 网格**就是当前首页四个算法卡片采用的 2×2 方块布局（类似 Apple 官网 Bento 风格的信息图）。它的问题在于：
- 四张卡片同时挤入视口，每张只有约一半屏幕宽度
- SVG 流程图被压到 `max-height: 210px`，细节看不清
- 用户无法聚焦单一算法

本次改造将**移除 Bento 网格**，替换为横向滚动的全宽卡片轮播。

---

## 3. 已确认的设计决策

| 问题 | 用户决策 | 规划调整 |
|------|----------|----------|
| 是否横向滚动吸附？ | ✅ 好的 | 采用 `scroll-snap-type: x mandatory` 轮播 |
| 是否明暗交替卡片背景？ | ❌ 不要 | **保持当前亮/暗主题系统**，卡片随主题自动适配，不额外做卡片级明暗切换 |
| 是否优化流程图？ | ✅ 好的 | SVG 增大至 520×320 以上，细化关键算法步骤 |
| 是否移除 Bento 网格？ | 待解释后确认 | 本规划按移除处理，SPA 已依赖 JS，无需保留降级 |
| 动画强度？ | 中 | 800-1000ms 入场 + SVG 绘制 + 微过渡，不做视差/扫描线等强装饰 |

**核心转向**：把设计精力从「卡片明暗节奏」转移到**卡片切换过渡动画**和**流程图叙事动画**上。

---

## 4. AirPods Pro 页面核心优势（与本次改造相关）

### 4.1 过渡与滚动叙事
- **startframe / endframe**：内容不是直接出现，而是随滚动逐步 reveal
- **章节切换的从容感**：大过渡 800-1200ms，不急躁
- **当前焦点清晰**：只有中心内容获得完整动画，两侧保持静态

### 4.2 排版对比
- 极端字号对比：标题极大，描述/参数极小
- 大标题行高紧（1.0-1.1），正文行高松（1.5-1.6）
- 全大写小字标签带宽字距

### 4.3 空间与留白
- 每次只讲一件事
- 产品图/视觉元素是绝对主角，文字辅助
- 大量留白营造高级感

---

## 5. 改造设计方案

### 5.1 整体结构

首页 Section 0 下半部分改为：

```
┌─────────────────────────────────────────────────────────────┐
│  工业级                                                     │
│  无监督 缺陷检测                                            │
│                     [ Hero 视觉 / 流程图 ]                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  01 / 04                                            │   │
│  │                                                     │   │
│  │  PatchCore          [ 大型 SVG 流程图 ]              │   │
│  │  CNN 特征记忆库                                     │   │
│  │  + 最近邻搜索                                       │   │
│  │  [首选]  24.9M 参数   零训练   工业首选             │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│                          ●  ○  ○  ○                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

- 轮播容器横向滚动，每次吸附一张卡片到中心
- 左右露出相邻卡片边缘约 7.5vw，暗示可滚动
- 底部居中指示点，显示当前位置
- 不另做卡片级明暗，全部跟随 `html[data-theme]`

### 5.2 单张卡片信息架构

```
┌─────────────────────────────────────────────────────────────┐
│  01 / 04                                                    │
│                                                             │
│  PatchCore                              [ 流程图 ]          │
│  CNN 特征记忆库 + 最近邻搜索                                  │
│                                                             │
│  [首选]  24.9M 参数   零训练   推理最快                     │
└─────────────────────────────────────────────────────────────┘
```

桌面端内部采用双栏：左侧文字区约占 40%，右侧流程图区约占 60%。

---

## 6. 详细实施规划

### 6.1 HTML 结构调整

**文件**：`modules/ui/static/index.html` Section 0

#### 替换容器

将现有：
```html
<div class="algo-grid bento-grid scroll-reveal-stagger">
    <!-- 四张 algo-card bento-card -->
</div>
```

替换为：
```html
<div class="algo-highlights" x-data="algoCarousel()" x-init="initCarousel()">
    <div class="algo-carousel" x-ref="algoCarousel"
         @scroll.debounce.100ms="updateActiveIndex()">
        <!-- 四张 algo-highlight-card -->
    </div>

    <div class="algo-carousel-dots" aria-label="算法轮播指示器">
        <template x-for="(m, idx) in algoModels" :key="m.key">
            <button class="algo-carousel-dot"
                    :class="{ 'is-active': activeIndex === idx }"
                    @click="scrollToIndex(idx)"
                    :aria-label="'查看 ' + m.name"
                    :aria-current="activeIndex === idx ? 'true' : 'false'"></button>
        </template>
    </div>

    <div class="algo-carousel-counter" aria-hidden="true">
        <span class="counter-current" x-text="String(activeIndex + 1).padStart(2, '0')"></span>
        <span class="counter-sep">/</span>
        <span class="counter-total">04</span>
    </div>
</div>
```

#### 单张卡片结构

```html
<div class="algo-highlight-card algo-highlight-card--patchcore"
     :class="{ 'is-active': activeIndex === 0, 'is-prev': activeIndex > 0, 'is-next': activeIndex < 3 }"
     data-index="0">
    <div class="algo-highlight-inner">
        <div class="algo-highlight-text">
            <div class="algo-highlight-eyebrow">01 / 04</div>
            <h3 class="algo-highlight-name">PatchCore</h3>
            <p class="algo-highlight-desc">CNN 特征记忆库 + 最近邻搜索</p>
            <div class="algo-highlight-stats">
                <span class="algo-highlight-tag algo-highlight-tag--rec">首选</span>
                <span class="algo-highlight-stat">24.9M 参数</span>
                <span class="algo-highlight-stat">零训练</span>
                <span class="algo-highlight-stat">推理最快</span>
            </div>
        </div>
        <div class="algo-highlight-visual">
            <svg class="flowchart-svg flowchart-svg--large" viewBox="0 0 560 340" aria-hidden="true">
                <!-- 优化后的流程图 -->
            </svg>
        </div>
    </div>
</div>
```

### 6.2 流程图优化方案

将 SVG 尺寸从 `420×270` 增大到 **560×340**，并对每个算法的关键步骤做视觉细化：

#### PatchCore 流程图（关键步骤细化）

```
输入图像 → 预训练 CNN Backbone → 多尺度 Patch 特征 → Coreset 子采样记忆库
                                              ↓
测试图像 → 特征提取 → 最近邻搜索 (k-NN) → 异常得分 → 像素级热力图
```

细化点：
- 把「CNN 特征」拆成「Backbone」+「Patch 特征」两个独立节点
- 明确标注「Coreset 子采样」是记忆库构建关键
- 从记忆库到测试流程用虚线连接，表示推理时检索
- 最终输出节点做成「热力图」样式（渐变填充）

#### PaDiM 流程图

```
正常样本 → 多尺度 CNN 特征 → 逐 Patch 高斯建模 (μ, Σ)
                                              ↓
测试样本 → 特征提取 → 马氏距离计算 → 异常得分 → 热力图
```

细化点：
- 高斯节点用 bell curve 图形暗示
- 标注 `per patch`
- 马氏距离节点强调「协方差逆」

#### FRE 流程图

```
正常样本 → ResNet50 特征 → 编码器 (Encoder)
                                     ↓
                                隐空间 (Latent)
                                     ↓
                                解码器 (Decoder) → 重构特征
                                     ↓
                           重构误差 → 异常得分 → 热力图
```

细化点：
- 用漏斗/瓶颈图形表示编码器-解码器
- 强调重构误差是核心信号

#### DRAEM 流程图

```
正常样本 ─┬─→ 合成异常生成 (DTD 纹理增强) ─┐
         │                                  ├→ 判别网络训练 (UNet)
         └─→ 原图                          │
                                            ↓
测试样本 → 判别网络推理 → 异常分割 → 异常得分 → 热力图
```

细化点：
- 用分支结构展示「正常 + 合成异常」共同训练
- 判别网络节点强调双分支（生成器/判别器）
- 异常分割用 mask 图形表示

视觉统一：
- 所有流程图节点统一圆角矩形
- 输入/输出节点用细边框
- 核心计算节点用算法强调色填充
- 箭头带方向标记，数据流清晰
- 节点尺寸适当放大，文字 13-14px

### 6.3 CSS 样式规划

**文件**：`modules/ui/static/css/apple-redesign.css`（优先，作为覆盖层）

#### 轮播容器

```css
.algo-highlights {
    position: relative;
    margin-top: 48px;
    width: 100%;
}

.algo-carousel {
    display: flex;
    gap: 24px;
    overflow-x: auto;
    scroll-snap-type: x mandatory;
    scroll-behavior: smooth;
    -webkit-overflow-scrolling: touch;
    scrollbar-width: none;
    -ms-overflow-style: none;
    padding: 24px 0 48px;
    padding-inline: calc((100vw - min(85vw, 980px)) / 2);
}
.algo-carousel::-webkit-scrollbar {
    display: none;
}
```

#### 卡片基础

```css
.algo-highlight-card {
    flex: 0 0 min(85vw, 980px);
    min-height: 480px;
    scroll-snap-align: center;
    border-radius: var(--r-xl);
    background: var(--bg-system);
    border: 1px solid var(--sep-subtle);
    overflow: hidden;
    position: relative;
    transition: transform 0.6s var(--ease-out-expo),
                opacity 0.6s var(--ease-out),
                box-shadow 0.6s var(--ease-out);
}

/* 非激活态：缩小 + 降低透明度 */
.algo-highlight-card:not(.is-active) {
    transform: scale(0.94);
    opacity: 0.55;
}

/* 激活态：正常大小 + 提升阴影 */
.algo-highlight-card.is-active {
    transform: scale(1);
    opacity: 1;
    box-shadow: var(--shadow-lg);
}

/* 左右相邻卡片微微可见 */
.algo-highlight-card.is-prev,
.algo-highlight-card.is-next {
    opacity: 0.7;
}
```

#### 卡片内部布局

```css
.algo-highlight-inner {
    display: grid;
    grid-template-columns: 1fr 1.5fr;
    gap: 40px;
    height: 100%;
    padding: 48px;
    align-items: center;
}

.algo-highlight-eyebrow {
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--algo-color, var(--accent));
    margin-bottom: 16px;
}

.algo-highlight-name {
    font-size: 64px;
    font-weight: 700;
    letter-spacing: -0.03em;
    line-height: 1.05;
    margin-bottom: 16px;
    color: var(--text);
}

.algo-highlight-desc {
    font-size: 20px;
    line-height: 1.5;
    color: var(--text-secondary);
    margin-bottom: 28px;
}

.algo-highlight-stats {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 12px;
}

.algo-highlight-tag {
    padding: 6px 14px;
    border-radius: 100px;
    font-size: 13px;
    font-weight: 600;
}
.algo-highlight-tag--rec {
    background: var(--algo-color, var(--accent));
    color: #fff;
}

.algo-highlight-stat {
    font-size: 14px;
    color: var(--text-tertiary);
    padding: 4px 0;
}
```

#### 流程图区域

```css
.algo-highlight-visual {
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 320px;
}

.flowchart-svg--large {
    width: 100%;
    height: auto;
    max-height: 340px;
}
```

#### 指示器与计数器

```css
.algo-carousel-dots {
    display: flex;
    justify-content: center;
    gap: 10px;
    margin-top: 8px;
}

.algo-carousel-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    border: none;
    background: var(--text-tertiary);
    cursor: pointer;
    transition: transform 0.3s var(--ease-spring),
                background 0.3s var(--ease-out);
}
.algo-carousel-dot.is-active {
    background: var(--accent);
    transform: scale(1.5);
}

.algo-carousel-counter {
    position: absolute;
    right: calc((100vw - min(85vw, 980px)) / 2);
    bottom: 0;
    font-size: 14px;
    font-weight: 600;
    color: var(--text-tertiary);
    letter-spacing: 0.05em;
}
.counter-current {
    color: var(--text);
}
```

### 6.4 过渡动画规格（重点）

**文件**：`modules/ui/static/js/algo-carousel.js`（新建）

#### 卡片切换过渡

当卡片进入/离开中心时：

```css
.algo-highlight-card {
    /* 基础过渡 */
    transition: transform 0.7s cubic-bezier(0.16, 1, 0.3, 1),
                opacity 0.6s cubic-bezier(0.16, 1, 0.3, 1),
                filter 0.6s cubic-bezier(0.16, 1, 0.3, 1);
}

.algo-highlight-card:not(.is-active) {
    transform: scale(0.92);
    opacity: 0.5;
    filter: blur(1px);
}

.algo-highlight-card.is-active {
    transform: scale(1);
    opacity: 1;
    filter: blur(0);
}
```

#### 激活卡片的 stagger 入场

使用 CSS 自定义属性和 `transition-delay`：

```css
.algo-highlight-card .algo-highlight-eyebrow,
.algo-highlight-card .algo-highlight-name,
.algo-highlight-card .algo-highlight-desc,
.algo-highlight-card .algo-highlight-stats,
.algo-highlight-card .flowchart-svg--large {
    opacity: 0;
    transform: translateY(24px);
    transition: opacity 0.7s var(--ease-out-expo),
                transform 0.7s var(--ease-out-expo);
}

.algo-highlight-card.is-active .algo-highlight-eyebrow { transition-delay: 0.05s; }
.algo-highlight-card.is-active .algo-highlight-name     { transition-delay: 0.12s; }
.algo-highlight-card.is-active .algo-highlight-desc     { transition-delay: 0.20s; }
.algo-highlight-card.is-active .algo-highlight-stats    { transition-delay: 0.28s; }
.algo-highlight-card.is-active .flowchart-svg--large    { transition-delay: 0.35s; }

.algo-highlight-card.is-active .algo-highlight-eyebrow,
.algo-highlight-card.is-active .algo-highlight-name,
.algo-highlight-card.is-active .algo-highlight-desc,
.algo-highlight-card.is-active .algo-highlight-stats,
.algo-highlight-card.is-active .flowchart-svg--large {
    opacity: 1;
    transform: translateY(0);
}
```

#### SVG 流程图路径绘制

为流程图路径添加 `stroke-dasharray` 动画：

```css
.flowchart-svg--large .fc-arrow,
.flowchart-svg--large .fc-connector {
    stroke-dasharray: 1200;
    stroke-dashoffset: 1200;
    opacity: 0;
}

.flowchart-svg--large .fc-node {
    opacity: 0;
    transform: scale(0.96);
    transform-origin: center;
}

.algo-highlight-card.is-active .flowchart-svg--large .fc-arrow,
.algo-highlight-card.is-active .flowchart-svg--large .fc-connector {
    animation: drawPath 0.9s cubic-bezier(0.16, 1, 0.3, 1) forwards;
}

.algo-highlight-card.is-active .flowchart-svg--large .fc-node {
    animation: popInNode 0.5s cubic-bezier(0.22, 0.8, 0.3, 1.15) forwards;
}

@keyframes drawPath {
    0% {
        opacity: 0;
        stroke-dashoffset: 1200;
    }
    10% {
        opacity: 1;
    }
    100% {
        opacity: 1;
        stroke-dashoffset: 0;
    }
}

@keyframes popInNode {
    0% {
        opacity: 0;
        transform: scale(0.9);
    }
    100% {
        opacity: 1;
        transform: scale(1);
    }
}
```

为不同路径/节点设置 stagger delay（通过 nth-child 或自定义属性）：

```css
.flowchart-svg--large .fc-node:nth-child(1) { animation-delay: 0.40s; }
.flowchart-svg--large .fc-node:nth-child(2) { animation-delay: 0.50s; }
.flowchart-svg--large .fc-node:nth-child(3) { animation-delay: 0.60s; }
/* ... */

.flowchart-svg--large .fc-arrow:nth-child(1) { animation-delay: 0.45s; }
.flowchart-svg--large .fc-arrow:nth-child(2) { animation-delay: 0.55s; }
/* ... */
```

#### 序号计数器过渡

计数器数字变化时做快速位移动画：

```css
.counter-current {
    display: inline-block;
    transition: transform 0.3s var(--ease-spring), opacity 0.3s var(--ease-out);
}
.counter-current.is-changing {
    transform: translateY(-8px);
    opacity: 0;
}
```

#### 强调色微光晕（克制）

仅在激活卡片背后添加极淡的算法色光晕：

```css
.algo-highlight-card::before {
    content: '';
    position: absolute;
    inset: -2px;
    border-radius: inherit;
    background: radial-gradient(circle at 75% 65%, var(--algo-color, var(--accent)) 0%, transparent 55%);
    opacity: 0;
    transition: opacity 0.8s var(--ease-out);
    pointer-events: none;
    z-index: 0;
}

.algo-highlight-card.is-active::before {
    opacity: 0.06;
}
```

### 6.5 JS 状态管理

新建 `modules/ui/static/js/algo-carousel.js`：

```js
document.addEventListener('alpine:init', () => {
    Alpine.data('algoCarousel', () => ({
        activeIndex: 0,
        prevIndex: 0,
        algoModels: [
            { key: 'patchcore', name: 'PatchCore', color: '#2997ff' },
            { key: 'padim', name: 'PaDiM', color: '#30d158' },
            { key: 'fre', name: 'FRE', color: '#ff9f0a' },
            { key: 'draem', name: 'DRAEM', color: '#bf5af2' }
        ],

        initCarousel() {
            this.observeActiveCard();
            this.bindKeyboard();
        },

        updateActiveIndex() {
            const carousel = this.$refs.algoCarousel;
            const card = carousel.firstElementChild;
            if (!card) return;
            const cardWidth = card.offsetWidth;
            const gap = parseInt(getComputedStyle(carousel).gap) || 24;
            const index = Math.round(carousel.scrollLeft / (cardWidth + gap));
            const newIndex = Math.max(0, Math.min(index, this.algoModels.length - 1));
            if (newIndex !== this.activeIndex) {
                this.prevIndex = this.activeIndex;
                this.activeIndex = newIndex;
            }
        },

        scrollToIndex(idx) {
            const carousel = this.$refs.algoCarousel;
            const card = carousel.children[idx];
            if (card) {
                card.scrollIntoView({ behavior: 'smooth', inline: 'center', block: 'nearest' });
            }
        },

        observeActiveCard() {
            const carousel = this.$refs.algoCarousel;
            const cards = carousel.querySelectorAll('.algo-highlight-card');
            const observer = new IntersectionObserver((entries) => {
                let bestEntry = null;
                entries.forEach(entry => {
                    if (entry.isIntersecting && (!bestEntry || entry.intersectionRatio > bestEntry.intersectionRatio)) {
                        bestEntry = entry;
                    }
                });
                if (bestEntry && bestEntry.intersectionRatio > 0.5) {
                    const idx = parseInt(bestEntry.target.dataset.index, 10);
                    if (idx !== this.activeIndex) {
                        this.prevIndex = this.activeIndex;
                        this.activeIndex = idx;
                    }
                }
            }, {
                root: carousel,
                threshold: [0, 0.25, 0.5, 0.75, 1]
            });
            cards.forEach(card => observer.observe(card));
        },

        bindKeyboard() {
            const carousel = this.$refs.algoCarousel;
            carousel.setAttribute('tabindex', '0');
            carousel.addEventListener('keydown', (e) => {
                if (e.key === 'ArrowLeft') {
                    e.preventDefault();
                    this.scrollToIndex(Math.max(0, this.activeIndex - 1));
                } else if (e.key === 'ArrowRight') {
                    e.preventDefault();
                    this.scrollToIndex(Math.min(this.algoModels.length - 1, this.activeIndex + 1));
                }
            });
        }
    }));
});
```

### 6.6 响应式策略

```css
/* 平板 */
@media (max-width: 1024px) {
    .algo-highlight-card {
        flex: 0 0 90vw;
        min-height: auto;
    }
    .algo-highlight-inner {
        grid-template-columns: 1fr;
        gap: 28px;
        padding: 36px;
    }
    .algo-highlight-name { font-size: 48px; }
    .algo-highlight-visual { min-height: 260px; }
}

/* 手机 */
@media (max-width: 768px) {
    .algo-highlight-card {
        flex: 0 0 92vw;
    }
    .algo-highlight-inner {
        padding: 28px;
    }
    .algo-highlight-name { font-size: 36px; }
    .algo-highlight-desc { font-size: 17px; }
    .algo-highlight-stats { gap: 8px; }
    .algo-highlight-stat { font-size: 13px; }
    .algo-carousel-counter { display: none; }
    .algo-highlight-visual { min-height: 220px; }
    .flowchart-svg--large { max-height: 220px; }
}
```

### 6.7 无障碍与降级

- **prefers-reduced-motion**：
  - 禁用 SVG 路径绘制动画
  - 禁用卡片缩放/模糊过渡
  - 内容直接可见，无延迟
- **键盘**：左右方向键切换卡片，Tab 聚焦轮播容器
- **屏幕阅读器**：卡片使用 `role="group"` + `aria-roledescription="slide"`
- **主题适配**：卡片不强制明暗，完全跟随 `html[data-theme]`

---

## 7. 文件变更清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `modules/ui/static/index.html` | 修改 | Section 0 替换 bento-grid 为 algo-carousel |
| `modules/ui/static/css/apple-redesign.css` | 新增大量样式 | 轮播、卡片、过渡动画、响应式 |
| `modules/ui/static/js/algo-carousel.js` | 新建 | Alpine 组件、IntersectionObserver、键盘导航 |
| `modules/ui/static/js/app.js` | 可能修改 | 确保 Section 0 横向滚动不影响全局 snap 页码计算 |
| `tests/test_ui_static.py` | 修改 | 更新结构断言，移除 bento-grid 依赖 |

---

## 8. 测试计划

1. **静态结构测试**
   - 更新 `test_index_html_has_new_structure`：检查 `algo-carousel`、`algo-highlight-card`、`algo-carousel-dots`
   - 移除 `bento-grid` 断言

2. **功能测试**
   - 横向滚动吸附是否正常
   - 指示点是否与当前卡片同步
   - 点击指示点是否平滑滚动到对应卡片
   - 键盘左右键是否可切换

3. **动画测试**
   - 当前激活卡片是否触发 stagger 入场
   - SVG 路径是否逐笔绘制
   - 切换卡片时旧卡片是否平滑淡出/缩小

4. **响应式测试**
   - 手机端垂直布局、触摸滑动
   - 平板端单栏布局

5. **无障碍测试**
   - `prefers-reduced-motion` 下动画是否禁用
   - 屏幕阅读器是否正确播报 slide 数量

---

## 9. 实施顺序

### Phase 1：结构与基础样式（约 35%）
1. 修改 `index.html` 轮播 HTML 结构
2. 编写 `apple-redesign.css` 轮播与卡片基础样式
3. 验证横向滚动与 snap

### Phase 2：流程图优化（约 25%）
4. 将四张 SVG 流程图增大到 560×340
5. 按第 6.2 节细化每个算法关键步骤
6. 统一视觉风格（节点、箭头、颜色、标签）

### Phase 3：过渡动画（约 30%）
7. 创建 `algo-carousel.js` Alpine 组件
8. 实现卡片切换 scale/opacity/blur 过渡
9. 实现 stagger 入场动画
10. 实现 SVG 路径绘制动画

### Phase 4：Polish（约 10%）
11. 响应式适配
12. 更新测试
13. 无障碍与 reduced-motion
14. 视觉走查

---

## 10. 与全站 AirPods Pro 化的关系

本次改造聚焦首页算法卡片，是全站 AirPods Pro 化的第一步：
- 验证新的横向滚动叙事模式
- 建立 `highlight-card` 组件范式（大图、大标题、 stagger 动画）
- 为后续单模型推理、Training Studio、四模型对比的重设计提供动画时序参考

---

## 11. 待用户最终确认

确认以下事项后即可开始实施：
1. ✅ 横向滚动吸附轮播
2. ✅ 保持当前亮/暗主题系统，卡片不做额外明暗切换
3. ✅ SVG 流程图增大并细化关键步骤
4. ✅ 移除 2×2 Bento 网格
5. ✅ 中强度动画（scale/opacity/blur 过渡 + stagger + SVG 绘制）
