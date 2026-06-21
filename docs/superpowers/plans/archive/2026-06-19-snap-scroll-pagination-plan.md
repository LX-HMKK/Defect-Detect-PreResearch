# Snap 滚动切页 + Apple 风格动画 — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将现有三段式连续滚动页面重构为全视口 CSS scroll-snap 吸附滚动，新增页间过渡动画、进度环导航、各 section 精细化排布。

**Architecture:** CSS `scroll-snap-type: y mandatory` 驱动原生吸附滚动；`IntersectionObserver` + WAAPI 编排内容进出动画；进度环导航点实时反映位置。后端 API 无改动。

**Tech Stack:** HTML + CSS + Vanilla JS (WAAPI) + Alpine.js 3.14.9 (不改版本)

**Spec:** `docs/superpowers/specs/2026-06-19-snap-scroll-pagination-design.md`

---

## 文件变更清单

| 文件 | 操作 | 职责 |
|------|------|------|
| `modules/ui/static/index.html` | 重写 | Snap 容器 + 三个 snap-page 内部新布局 |
| `modules/ui/static/css/app.css` | 大幅改写 | Snap 样式、新布局、进出动画、进度环 |
| `modules/ui/static/js/app.js` | 中度修改 | Snap 事件监听、进度环更新、nav bar 标题接力 |
| `modules/ui/static/js/animations.js` | 小幅扩增 | Snap 感知的 section 进出动画编排 |
| `modules/ui/static/css/flowchart.css` | 小幅修改 | SVG 自适应缩放 |

**不改的文件:** `server.py`, `theme.py`, `inference.js`, `compare.js`, `cursor-glow.js`, `flowchart.js`, `theme.js`, `styles.css`, `demo.py`, `inference-interact.js`

---

### Task 1: HTML 结构 — Snap 容器 + Section 0 重排

**Files:**
- Modify: `modules/ui/static/index.html` (全文重写)

- [ ] **Step 1: 建立 snap 容器骨架，移除过渡区，固定导航栏**

将 `<body>` 的内部结构替换为以下骨架（内部 Alpine 绑定保持不变）：

```html
<body x-data="app" x-init="init()" @mousemove="onMouseMove">

    <!-- 跳过导航链接（无障碍） -->
    <a href="#s0" class="skip-link">跳到主内容</a>

    <!-- 页面加载遮罩 -->
    <div class="page-loader" x-ref="pageLoader">
        <svg class="loader-logo" width="48" height="48" viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg">
            <defs>
                <linearGradient id="logoGradLoader" x1="18" y1="2" x2="18" y2="34" gradientUnits="userSpaceOnUse">
                    <stop offset="0%" stop-color="#2997ff"/>
                    <stop offset="100%" stop-color="#0070d6"/>
                </linearGradient>
            </defs>
            <rect class="logo-outer" x="2" y="18" width="22.63" height="22.63" rx="3"
                  transform="rotate(-45 2 18)" fill="url(#logoGradLoader)" opacity="0.15"/>
            <rect class="logo-mid" x="6" y="18" width="16.97" height="16.97" rx="2"
                  transform="rotate(-45 6 18)" fill="url(#logoGradLoader)" opacity="0.35"/>
            <rect class="logo-inner" x="9.5" y="18" width="12.02" height="12.02" rx="2"
                  transform="rotate(-45 9.5 18)" fill="url(#logoGradLoader)"/>
            <circle cx="18" cy="18" r="2.5" fill="#ffffff" opacity="0.9"/>
        </svg>
    </div>

    <!-- 导航栏 — fixed 置于 snap 容器之上 -->
    <nav class="navbar" :class="{ 'navbar--scrolled': currentSection > 0 }">
        <div class="navbar-inner">
            <div class="navbar-brand">
                <svg width="28" height="28" viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <defs>
                        <linearGradient id="logoGradNav" x1="18" y1="2" x2="18" y2="34" gradientUnits="userSpaceOnUse">
                            <stop offset="0%" stop-color="#2997ff"/>
                            <stop offset="100%" stop-color="#0070d6"/>
                        </linearGradient>
                    </defs>
                    <rect x="2" y="18" width="22.63" height="22.63" rx="3"
                          transform="rotate(-45 2 18)" fill="url(#logoGradNav)" opacity="0.15"/>
                    <rect x="6" y="18" width="16.97" height="16.97" rx="2"
                          transform="rotate(-45 6 18)" fill="url(#logoGradNav)" opacity="0.35"/>
                    <rect x="9.5" y="18" width="12.02" height="12.02" rx="2"
                          transform="rotate(-45 9.5 18)" fill="url(#logoGradNav)"/>
                    <circle cx="18" cy="18" r="2.5" fill="#ffffff" opacity="0.9"/>
                </svg>
                <span class="navbar-title">缺陷检测</span>
                <!-- section 标题接力：离开 S0 后显示当前 section 名 -->
                <span class="navbar-section" x-show="currentSection > 0" x-text="sectionNames[currentSection]" x-transition.opacity.duration.300ms></span>
            </div>

            <div class="navbar-right">
                <select class="nav-select" x-model="selectedDataset" aria-label="选择数据集">
                    <template x-for="ds in datasets" :key="ds">
                        <option x-text="ds" :value="ds"></option>
                    </template>
                </select>

                <button class="theme-capsule" @click="toggleTheme"
                        :aria-label="theme === 'dark' ? '切换到亮色模式' : '切换到暗色模式'">
                    <span class="capsule-icon capsule-icon--sun">&#9728;</span>
                    <span class="capsule-knob" :class="{ 'capsule-knob--light': theme === 'light' }"></span>
                    <span class="capsule-icon capsule-icon--moon">&#9790;</span>
                </button>
            </div>
        </div>
    </nav>

    <!-- 进度环导航点 -->
    <div class="snap-dots" aria-label="页面导航">
        <svg class="snap-ring" viewBox="0 0 40 40" width="40" height="40">
            <circle cx="20" cy="20" r="17" fill="none" stroke="var(--text-tertiary)" stroke-width="1.5" opacity="0.3"/>
            <circle cx="20" cy="20" r="17" fill="none" stroke="var(--accent)" stroke-width="1.5"
                    stroke-linecap="round" stroke-dasharray="106.8" :stroke-dashoffset="106.8 - 106.8 * snapProgress"/>
        </svg>
        <span class="snap-dot-label" x-text="(currentSection + 1) + ' / 3'"></span>
    </div>

    <!-- ══════════════════════════════════════════════════ -->
    <!-- Snap 滚动容器 -->
    <!-- ══════════════════════════════════════════════════ -->
    <div class="snap-container" x-ref="snapContainer">
```

- [ ] **Step 2: 重写 Section 0 — 算法介绍**

```html
        <!-- ═══ Section 0: 算法介绍 ═══ -->
        <section id="s0" class="snap-page" x-ref="section0">
            <div class="snap-page-inner">
                <div class="hero">
                    <h1 class="hero-title scroll-reveal">工业缺陷检测</h1>
                    <p class="hero-subtitle scroll-reveal">四算法 &middot; 无监督 &middot; 像素级定位</p>
                </div>

                <!-- 2x2 算法卡片网格 — 每张卡片带算法色标 -->
                <div class="algo-grid scroll-reveal-stagger">
                    <!-- PatchCore — 蓝色 #2997ff -->
                    <div class="algo-card algo-card--patchcore scroll-reveal" x-intersect="onCardIntersect">
                        <div class="algo-card-accent" style="--algo-color: #2997ff;"></div>
                        <div class="algo-card-body">
                            <h3><span class="algo-icon">🔍</span> PatchCore <span class="algo-tag algo-tag--rec">首选</span></h3>
                            <p>CNN 特征记忆库 + 最近邻搜索 &middot; 24.9M 参数</p>
                            <p class="algo-card-kicker">零训练、推理最快，工业首选</p>
                        </div>
                        <svg class="flowchart-svg" viewBox="0 0 420 270" xmlns="http://www.w3.org/2000/svg">
                            <!-- SVG 内容与当前相同，保持不变 -->
                            <defs>
                                <marker id="arrow-pc" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
                                    <polygon points="0 0, 8 3, 0 6" fill="var(--text-tertiary)"/>
                                </marker>
                            </defs>
                            <g class="fc-node">
                                <rect x="8" y="62" width="85" height="44" rx="8"
                                      fill="var(--bg-secondary)" stroke="var(--sep-default)" stroke-width="1.5"/>
                                <text class="fc-label" x="50" y="89" text-anchor="middle"
                                      font-family="var(--font-body)" font-size="12" fill="var(--text)">正常样本</text>
                            </g>
                            <line class="fc-arrow" x1="93" y1="84" x2="116" y2="84"
                                  stroke="var(--text-tertiary)" stroke-width="1.5" marker-end="url(#arrow-pc)"/>
                            <g class="fc-node">
                                <rect x="118" y="62" width="85" height="44" rx="8"
                                      fill="var(--bg-secondary)" stroke="var(--sep-default)" stroke-width="1.5"/>
                                <text class="fc-label" x="160" y="89" text-anchor="middle"
                                      font-family="var(--font-body)" font-size="12" fill="var(--text)">CNN 特征</text>
                            </g>
                            <line class="fc-arrow" x1="203" y1="84" x2="226" y2="84"
                                  stroke="var(--text-tertiary)" stroke-width="1.5" marker-end="url(#arrow-pc)"/>
                            <g class="fc-node">
                                <rect x="228" y="62" width="85" height="44" rx="8"
                                      fill="var(--bg-secondary)" stroke="var(--sep-default)" stroke-width="1.5"/>
                                <text class="fc-label" x="270" y="89" text-anchor="middle"
                                      font-family="var(--font-body)" font-size="12" fill="var(--text)">Patch 特征</text>
                            </g>
                            <line class="fc-arrow" x1="313" y1="84" x2="336" y2="84"
                                  stroke="var(--text-tertiary)" stroke-width="1.5" marker-end="url(#arrow-pc)"/>
                            <g class="fc-node">
                                <rect x="338" y="58" width="74" height="52" rx="8"
                                      fill="var(--accent-dim)" stroke="var(--accent)" stroke-width="1.5"/>
                                <text class="fc-label" x="375" y="81" text-anchor="middle"
                                      font-family="var(--font-body)" font-size="12" font-weight="600" fill="var(--accent)">记忆库</text>
                                <text class="fc-label" x="375" y="97" text-anchor="middle"
                                      font-family="var(--font-body)" font-size="10" fill="var(--text-secondary)">(coreset)</text>
                            </g>
                            <polyline class="fc-arrow" points="375,110 375,158 175,158 175,186"
                                      fill="none" stroke="var(--text-tertiary)" stroke-width="1.5" marker-end="url(#arrow-pc)"/>
                            <g class="fc-node">
                                <rect x="8" y="188" width="95" height="44" rx="8"
                                      fill="var(--bg-secondary)" stroke="var(--sep-default)" stroke-width="1.5"/>
                                <text class="fc-label" x="55" y="215" text-anchor="middle"
                                      font-family="var(--font-body)" font-size="12" fill="var(--text)">测试样本特征</text>
                            </g>
                            <line class="fc-arrow" x1="103" y1="210" x2="128" y2="210"
                                  stroke="var(--text-tertiary)" stroke-width="1.5" marker-end="url(#arrow-pc)"/>
                            <g class="fc-node">
                                <rect x="130" y="188" width="90" height="44" rx="8"
                                      fill="var(--bg-secondary)" stroke="var(--sep-default)" stroke-width="1.5"/>
                                <text class="fc-label" x="175" y="215" text-anchor="middle"
                                      font-family="var(--font-body)" font-size="12" fill="var(--text)">最近邻距离</text>
                            </g>
                            <line class="fc-arrow" x1="220" y1="210" x2="246" y2="210"
                                  stroke="var(--text-tertiary)" stroke-width="1.5" marker-end="url(#arrow-pc)"/>
                            <g class="fc-node">
                                <rect x="248" y="184" width="85" height="52" rx="8"
                                      fill="var(--ok-bg)" stroke="var(--ok)" stroke-width="1.5"/>
                                <text class="fc-label" x="290" y="207" text-anchor="middle"
                                      font-family="var(--font-body)" font-size="12" font-weight="600" fill="var(--ok)">异常得分</text>
                                <text class="fc-label" x="290" y="223" text-anchor="middle"
                                      font-family="var(--font-body)" font-size="10" fill="var(--text-secondary)">→ 热力图</text>
                            </g>
                        </svg>
                    </div>

                    <!-- PaDiM — 绿色 #30d158 -->
                    <div class="algo-card algo-card--padim scroll-reveal" x-intersect="onCardIntersect">
                        <div class="algo-card-accent" style="--algo-color: #30d158;"></div>
                        <div class="algo-card-body">
                            <h3><span class="algo-icon">📊</span> PaDiM <span class="algo-tag algo-tag--alt">轻量</span></h3>
                            <p>Patch 高斯分布建模 + 马氏距离 &middot; 2.8M 参数</p>
                            <p class="algo-card-kicker">参数量最少，适合边缘部署</p>
                        </div>
                        <!-- SVG 流程图不变 -->
                        <svg class="flowchart-svg" viewBox="0 0 420 270" xmlns="http://www.w3.org/2000/svg">
                            <defs>
                                <marker id="arrow-pdim" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
                                    <polygon points="0 0, 8 3, 0 6" fill="var(--text-tertiary)"/>
                                </marker>
                            </defs>
                            <g class="fc-node">
                                <rect x="10" y="72" width="82" height="44" rx="8"
                                      fill="var(--bg-secondary)" stroke="var(--sep-default)" stroke-width="1.5"/>
                                <text class="fc-label" x="51" y="99" text-anchor="middle"
                                      font-family="var(--font-body)" font-size="12" fill="var(--text)">正常样本</text>
                            </g>
                            <line class="fc-arrow" x1="92" y1="94" x2="116" y2="94"
                                  stroke="var(--text-tertiary)" stroke-width="1.5" marker-end="url(#arrow-pdim)"/>
                            <g class="fc-node">
                                <rect x="118" y="72" width="112" height="44" rx="8"
                                      fill="var(--bg-secondary)" stroke="var(--sep-default)" stroke-width="1.5"/>
                                <text class="fc-label" x="174" y="99" text-anchor="middle"
                                      font-family="var(--font-body)" font-size="12" fill="var(--text)">CNN 多尺度特征</text>
                            </g>
                            <line class="fc-arrow" x1="230" y1="94" x2="254" y2="94"
                                  stroke="var(--text-tertiary)" stroke-width="1.5" marker-end="url(#arrow-pdim)"/>
                            <g class="fc-node">
                                <rect x="256" y="68" width="100" height="52" rx="8"
                                      fill="var(--accent-dim)" stroke="var(--accent)" stroke-width="1.5"/>
                                <text class="fc-label" x="306" y="89" text-anchor="middle"
                                      font-family="var(--font-body)" font-size="12" font-weight="600" fill="var(--accent)">Patch 高斯分布</text>
                                <text class="fc-label" x="306" y="106" text-anchor="middle"
                                      font-family="var(--font-body)" font-size="10" fill="var(--text-secondary)">(μ, Σ per patch)</text>
                            </g>
                            <polyline class="fc-arrow" points="306,120 306,154 232,154 232,186"
                                      fill="none" stroke="var(--text-tertiary)" stroke-width="1.5" marker-end="url(#arrow-pdim)"/>
                            <g class="fc-node">
                                <rect x="192" y="188" width="80" height="44" rx="8"
                                      fill="var(--bg-secondary)" stroke="var(--sep-default)" stroke-width="1.5"/>
                                <text class="fc-label" x="232" y="215" text-anchor="middle"
                                      font-family="var(--font-body)" font-size="12" fill="var(--text)">马氏距离</text>
                            </g>
                            <line class="fc-arrow" x1="272" y1="210" x2="296" y2="210"
                                  stroke="var(--text-tertiary)" stroke-width="1.5" marker-end="url(#arrow-pdim)"/>
                            <g class="fc-node">
                                <rect x="298" y="184" width="85" height="52" rx="8"
                                      fill="var(--ok-bg)" stroke="var(--ok)" stroke-width="1.5"/>
                                <text class="fc-label" x="340" y="207" text-anchor="middle"
                                      font-family="var(--font-body)" font-size="12" font-weight="600" fill="var(--ok)">异常得分</text>
                                <text class="fc-label" x="340" y="223" text-anchor="middle"
                                      font-family="var(--font-body)" font-size="10" fill="var(--text-secondary)">→ 热力图</text>
                            </g>
                        </svg>
                    </div>

                    <!-- FRE — 橙色 #ff9f0a -->
                    <div class="algo-card algo-card--fre scroll-reveal" x-intersect="onCardIntersect">
                        <div class="algo-card-accent" style="--algo-color: #ff9f0a;"></div>
                        <div class="algo-card-body">
                            <h3><span class="algo-icon">🔄</span> FRE</h3>
                            <p>ResNet50 特征提取 + 自编码器重构误差 &middot; 23.0M 参数</p>
                            <p class="algo-card-kicker">高解释性，适合质量追溯</p>
                        </div>
                        <svg class="flowchart-svg" viewBox="0 0 420 270" xmlns="http://www.w3.org/2000/svg">
                            <defs>
                                <marker id="arrow-fre" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
                                    <polygon points="0 0, 8 3, 0 6" fill="var(--text-tertiary)"/>
                                </marker>
                            </defs>
                            <g class="fc-node">
                                <rect x="8" y="75" width="82" height="44" rx="8"
                                      fill="var(--bg-secondary)" stroke="var(--sep-default)" stroke-width="1.5"/>
                                <text class="fc-label" x="49" y="102" text-anchor="middle"
                                      font-family="var(--font-body)" font-size="12" fill="var(--text)">正常样本</text>
                            </g>
                            <line class="fc-arrow" x1="90" y1="97" x2="114" y2="97"
                                  stroke="var(--text-tertiary)" stroke-width="1.5" marker-end="url(#arrow-fre)"/>
                            <g class="fc-node">
                                <rect x="116" y="75" width="82" height="44" rx="8"
                                      fill="var(--bg-secondary)" stroke="var(--sep-default)" stroke-width="1.5"/>
                                <text class="fc-label" x="157" y="102" text-anchor="middle"
                                      font-family="var(--font-body)" font-size="12" fill="var(--text)">CNN 特征</text>
                            </g>
                            <line class="fc-arrow" x1="198" y1="97" x2="222" y2="97"
                                  stroke="var(--text-tertiary)" stroke-width="1.5" marker-end="url(#arrow-fre)"/>
                            <g class="fc-node">
                                <rect x="224" y="71" width="110" height="52" rx="8"
                                      fill="var(--accent-dim)" stroke="var(--accent)" stroke-width="1.5"/>
                                <text class="fc-label" x="279" y="92" text-anchor="middle"
                                      font-family="var(--font-body)" font-size="12" font-weight="600" fill="var(--accent)">线性自编码器</text>
                                <text class="fc-label" x="279" y="109" text-anchor="middle"
                                      font-family="var(--font-body)" font-size="10" fill="var(--text-secondary)">(encoder→decoder)</text>
                            </g>
                            <polyline class="fc-arrow" points="279,123 279,158 147,158 147,186"
                                      fill="none" stroke="var(--text-tertiary)" stroke-width="1.5" marker-end="url(#arrow-fre)"/>
                            <g class="fc-node">
                                <rect x="105" y="188" width="84" height="44" rx="8"
                                      fill="var(--bg-secondary)" stroke="var(--sep-default)" stroke-width="1.5"/>
                                <text class="fc-label" x="147" y="215" text-anchor="middle"
                                      font-family="var(--font-body)" font-size="12" fill="var(--text)">重构误差</text>
                            </g>
                            <line class="fc-arrow" x1="189" y1="210" x2="214" y2="210"
                                  stroke="var(--text-tertiary)" stroke-width="1.5" marker-end="url(#arrow-fre)"/>
                            <g class="fc-node">
                                <rect x="216" y="184" width="85" height="52" rx="8"
                                      fill="var(--ok-bg)" stroke="var(--ok)" stroke-width="1.5"/>
                                <text class="fc-label" x="258" y="207" text-anchor="middle"
                                      font-family="var(--font-body)" font-size="12" font-weight="600" fill="var(--ok)">异常得分</text>
                                <text class="fc-label" x="258" y="223" text-anchor="middle"
                                      font-family="var(--font-body)" font-size="10" fill="var(--text-secondary)">→ 热力图</text>
                            </g>
                        </svg>
                    </div>

                    <!-- DRAEM — 紫色 #bf5af2 -->
                    <div class="algo-card algo-card--draem scroll-reveal" x-intersect="onCardIntersect">
                        <div class="algo-card-accent" style="--algo-color: #bf5af2;"></div>
                        <div class="algo-card-body">
                            <h3><span class="algo-icon">🎯</span> DRAEM</h3>
                            <p>合成异常 + UNet 判别网络 &middot; 97.4M 参数</p>
                            <p class="algo-card-kicker">无需真实异常样本，微小缺陷灵敏</p>
                        </div>
                        <svg class="flowchart-svg" viewBox="0 0 420 270" xmlns="http://www.w3.org/2000/svg">
                            <defs>
                                <marker id="arrow-draem" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
                                    <polygon points="0 0, 8 3, 0 6" fill="var(--text-tertiary)"/>
                                </marker>
                            </defs>
                            <g class="fc-node">
                                <rect x="8" y="66" width="82" height="44" rx="8"
                                      fill="var(--bg-secondary)" stroke="var(--sep-default)" stroke-width="1.5"/>
                                <text class="fc-label" x="49" y="93" text-anchor="middle"
                                      font-family="var(--font-body)" font-size="12" fill="var(--text)">正常样本</text>
                            </g>
                            <line class="fc-arrow" x1="90" y1="88" x2="114" y2="88"
                                  stroke="var(--text-tertiary)" stroke-width="1.5" marker-end="url(#arrow-draem)"/>
                            <g class="fc-node">
                                <rect x="116" y="62" width="102" height="52" rx="8"
                                      fill="var(--warn-bg)" stroke="var(--warn)" stroke-width="1.5"/>
                                <text class="fc-label" x="167" y="83" text-anchor="middle"
                                      font-family="var(--font-body)" font-size="12" font-weight="600" fill="var(--warn)">合成异常</text>
                                <text class="fc-label" x="167" y="100" text-anchor="middle"
                                      font-family="var(--font-body)" font-size="10" fill="var(--text-secondary)">(纹理增强)</text>
                            </g>
                            <line class="fc-arrow" x1="218" y1="88" x2="244" y2="88"
                                  stroke="var(--text-tertiary)" stroke-width="1.5" marker-end="url(#arrow-draem)"/>
                            <g class="fc-node">
                                <rect x="246" y="62" width="100" height="52" rx="8"
                                      fill="var(--accent-dim)" stroke="var(--accent)" stroke-width="1.5"/>
                                <text class="fc-label" x="296" y="83" text-anchor="middle"
                                      font-family="var(--font-body)" font-size="12" font-weight="600" fill="var(--accent)">判别网络训练</text>
                                <text class="fc-label" x="296" y="100" text-anchor="middle"
                                      font-family="var(--font-body)" font-size="10" fill="var(--text-secondary)">(UNet-like)</text>
                            </g>
                            <polyline class="fc-arrow" points="296,114 296,152 162,152 162,186"
                                      fill="none" stroke="var(--text-tertiary)" stroke-width="1.5" marker-end="url(#arrow-draem)"/>
                            <g class="fc-node">
                                <rect x="112" y="188" width="100" height="44" rx="8"
                                      fill="var(--bg-secondary)" stroke="var(--sep-default)" stroke-width="1.5"/>
                                <text class="fc-label" x="162" y="215" text-anchor="middle"
                                      font-family="var(--font-body)" font-size="12" fill="var(--text)">异常分割</text>
                            </g>
                            <line class="fc-arrow" x1="212" y1="210" x2="240" y2="210"
                                  stroke="var(--text-tertiary)" stroke-width="1.5" marker-end="url(#arrow-draem)"/>
                            <g class="fc-node">
                                <rect x="242" y="184" width="85" height="52" rx="8"
                                      fill="var(--ok-bg)" stroke="var(--ok)" stroke-width="1.5"/>
                                <text class="fc-label" x="284" y="207" text-anchor="middle"
                                      font-family="var(--font-body)" font-size="12" font-weight="600" fill="var(--ok)">异常得分</text>
                                <text class="fc-label" x="284" y="223" text-anchor="middle"
                                      font-family="var(--font-body)" font-size="10" fill="var(--text-secondary)">→ 热力图</text>
                            </g>
                        </svg>
                    </div>
                </div>

                <!-- 底部滚动提示（首次访问显示） -->
                <div class="scroll-hint" x-show="currentSection === 0" x-transition.opacity.duration.500ms>
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M12 5v14M5 12l7 7 7-7" stroke="var(--text-tertiary)" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
                    </svg>
                    <span>向下滚动</span>
                </div>
            </div>
        </section>
```

- [ ] **Step 3: 重写 Section 1 — 三列流水线 + 左右分栏结果**

```html
        <!-- ═══ Section 1: 单模型推理 ═══ -->
        <section id="s1" class="snap-page" x-ref="section1">
            <div class="snap-page-inner">
                <h2 class="section-title scroll-reveal">单模型推理</h2>

                <div class="inference-area">
                    <!-- 三列流水线 -->
                    <div class="pipeline">
                        <!-- 第 1 列：上传 -->
                        <div class="pipeline-step" :class="{ 'pipeline-step--done': uploadedFile }">
                            <div class="pipeline-step-num">1</div>
                            <div class="pipeline-step-body">
                                <div class="upload-zone"
                                     :class="{ 'has-file': uploadedFile }"
                                     x-on:click="!uploadedFile && $refs.fileInput.click()"
                                     x-on:dragover.prevent
                                     x-on:drop.prevent="onDrop">
                                    <input type="file" x-ref="fileInput"
                                           accept="image/png,image/jpeg,image/bmp,image/webp"
                                           @change="onFileSelected" hidden>

                                    <template x-if="!uploadedFile">
                                        <div class="upload-placeholder">
                                            <svg class="upload-icon-svg" width="36" height="36" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
                                                <rect x="4" y="10" width="40" height="32" rx="4" stroke="currentColor" stroke-width="1.5" opacity="0.4"/>
                                                <path d="M4 26l10-8 8 6 10-10 12 12" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" opacity="0.4"/>
                                                <circle cx="16" cy="18" r="3" stroke="currentColor" stroke-width="1.5" opacity="0.4"/>
                                            </svg>
                                            <div class="upload-text">拖拽或点击上传</div>
                                            <div class="upload-hint">PNG / JPG / BMP</div>
                                        </div>
                                    </template>

                                    <template x-if="uploadedFile">
                                        <div class="upload-preview">
                                            <img :src="uploadPreviewUrl" alt="预览">
                                            <button class="upload-clear-btn" @click.stop="resetInference" aria-label="清除">&times;</button>
                                        </div>
                                    </template>
                                </div>
                            </div>
                        </div>

                        <!-- 第 2 列：选择模型 -->
                        <div class="pipeline-step">
                            <div class="pipeline-step-num">2</div>
                            <div class="pipeline-step-body">
                                <select class="model-select" x-model="selectedModel" aria-label="选择算法">
                                    <template x-for="m in models" :key="m.key">
                                        <option :value="m.key" x-text="m.name + ' — ' + m.direction"></option>
                                    </template>
                                </select>
                            </div>
                        </div>

                        <!-- 第 3 列：推理 -->
                        <div class="pipeline-step">
                            <div class="pipeline-step-num">3</div>
                            <div class="pipeline-step-body">
                                <button class="btn-inference"
                                        :class="inferenceState"
                                        :disabled="!uploadedFile || inferenceState === 'loading' || inferenceState === 'inferring'"
                                        @click="startInference()">
                                    <span x-show="inferenceState === 'idle' || inferenceState === 'uploaded' || inferenceState === 'error'">开始推理</span>
                                    <span x-show="inferenceState === 'loading' || inferenceState === 'inferring'" class="btn-spinner-text">
                                        <span class="btn-spinner"></span> 推理中...
                                    </span>
                                    <span x-show="inferenceState === 'done'" class="btn-done-text">完成 &#10003;</span>
                                </button>
                            </div>
                        </div>
                    </div>

                    <!-- 进度条 -->
                    <div class="progress-bar" x-show="inferenceState === 'loading' || inferenceState === 'inferring'">
                        <div class="progress-track">
                            <div class="progress-fill" :style="{ width: inferenceProgress.pct + '%' }"></div>
                        </div>
                        <div class="progress-message" x-text="inferenceProgress.message"></div>
                    </div>

                    <!-- 错误卡片 -->
                    <div class="error-card" x-show="inferenceState === 'error'" x-transition>
                        <span class="error-icon">&#9888;</span>
                        <span class="error-text" x-text="errorMessage"></span>
                    </div>

                    <!-- 占位提示 -->
                    <div class="result-placeholder" x-show="inferenceState === 'idle' || (inferenceState === 'uploaded' && !resultData)">
                        <div class="result-empty">
                            <p>选择模型并上传图片后，点击「开始推理」</p>
                        </div>
                    </div>

                    <!-- 结果面板：左右分栏 -->
                    <template x-if="inferenceState === 'done' && resultData">
                    <div class="result-panel scroll-reveal" x-transition>
                        <div class="result-layout">
                            <!-- 左侧：对比滑块 -->
                            <div class="result-left">
                                <div class="compare-slider" x-data="imageCompare">
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
                                    <div class="heatmap-legend">
                                        <div class="legend-label">异常得分</div>
                                        <div class="legend-bar-wrap">
                                            <div class="legend-bar"></div>
                                            <div class="legend-labels">
                                                <span>1.0</span><span>0.8</span><span>0.6</span><span>0.4</span><span>0.2</span><span>0.0</span>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            <!-- 右侧：指标卡 -->
                            <div class="result-right">
                                <div class="result-card">
                                    <div class="result-header">
                                        <span class="result-model" x-text="resultData.model_name"></span>
                                        <span class="result-badge"
                                              :class="resultData.is_anomaly ? 'badge-anomaly' : 'badge-normal'"
                                              x-text="resultData.is_anomaly ? '异常' : '正常'"></span>
                                    </div>

                                    <div class="result-metrics">
                                        <div class="metric">
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

                                        <div class="metric">
                                            <div class="metric-label">置信度</div>
                                            <div class="metric-value result-confidence-value"
                                                  x-text="(resultData.confidence * 100).toFixed(1) + '%'"></div>
                                            <div class="metric-bar">
                                                <div class="metric-fill fill-confidence"
                                                     :style="{ width: (resultData.confidence * 100) + '%' }"></div>
                                            </div>
                                        </div>
                                    </div>

                                    <div class="result-verdict"
                                         :class="resultData.is_anomaly ? 'verdict-anomaly' : 'verdict-normal'">
                                        <span class="verdict-dot"
                                              :class="resultData.is_anomaly ? 'dot-anomaly' : 'dot-normal'"
                                              x-text="resultData.is_anomaly ? '●' : '●'"></span>
                                        <span>
                                            得分 <b x-text="resultData.score.toFixed(4)"></b>
                                            <span x-text="resultData.is_anomaly ? ' > ' : ' ≤ '"></span>
                                            阈值 <b>τ = <span x-text="resultData.threshold.toFixed(3)"></span></b>
                                            &rarr; <b x-text="resultData.is_anomaly ? '异常' : '正常'"></b>
                                        </span>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <!-- 隐藏数据：供 JS 读取 -->
                        <img :src="resultData?.anomaly_map_b64"
                             x-ref="anomalyMapData"
                             x-show="false"
                             @load="setupVisualInteractions()">
                        <div x-ref="bboxData"
                             :data-bboxes="JSON.stringify(resultData?.bboxes || [])"
                             x-show="false"></div>
                    </div>
                    </template>

                    <!-- 重试/重置 -->
                    <div class="inference-actions" x-show="inferenceState === 'done' || inferenceState === 'error'">
                        <button class="btn-reset"
                                @click="inferenceState === 'error' ? startInference() : resetInference()">
                            <span x-show="inferenceState === 'error'">重试</span>
                            <span x-show="inferenceState === 'done'">重新上传</span>
                        </button>
                    </div>
                </div>
            </div>
        </section>
```

- [ ] **Step 4: 重写 Section 2 — 共享原图 + 四列热力图**

```html
        <!-- ═══ Section 2: 四模型对比 ═══ -->
        <section id="s2" class="snap-page" x-ref="section2" x-data="compare">
            <div class="snap-page-inner">
                <h2 class="section-title scroll-reveal">四模型对比</h2>

                <!-- 引导提示 -->
                <div class="compare-guide" x-show="!compareRunning && !compareDone">
                    <p>先完成单模型推理上传图片，然后点击下方按钮一键运行四种算法</p>
                </div>

                <!-- 对比按钮 -->
                <button class="btn-compare" @click="startCompare()"
                        :disabled="compareRunning"
                        x-show="!compareRunning || compareDone">
                    <span x-show="!compareRunning && !compareDone">&#9654; 四模型同时对比</span>
                    <span x-show="compareDone">&#8635; 重新对比</span>
                </button>

                <!-- 进度指示器 -->
                <div class="compare-progress" x-show="compareRunning">
                    <div class="compare-progress-text">
                        正在对比中... <span x-text="completedCount"></span>/4
                    </div>
                    <div class="compare-progress-track">
                        <div class="compare-progress-fill"
                             :style="{ width: (completedCount / 4 * 100) + '%' }"></div>
                    </div>
                </div>

                <!-- 摘要栏（横向） -->
                <template x-if="summary">
                <div class="compare-summary" x-transition>
                    <span class="compare-summary-trophy">&#127942;</span>
                    <span class="compare-summary-best">
                        最佳：<strong x-text="summary.best_name"></strong>
                        <code x-text="summary.best_score?.toFixed(4)"></code>
                    </span>
                    <div class="compare-ranking">
                        <template x-for="(r, i) in summary.ranking" :key="r.model">
                            <span class="compare-rank-item">
                                <span class="rank-num" x-text="'#' + (i + 1)"></span>
                                <span class="rank-name" x-text="r.name"></span>
                                <code x-text="r.score.toFixed(4)"></code>
                            </span>
                        </template>
                    </div>
                </div>
                </template>

                <!-- 共享原图 -->
                <div class="compare-shared-image" x-show="compareDone">
                    <span class="compare-shared-label">原图</span>
                    <img :src="compareSlots.patchcore?.data?.image_b64 || compareSlots.padim?.data?.image_b64 || compareSlots.fre?.data?.image_b64 || compareSlots.draem?.data?.image_b64"
                         alt="原始图片" class="compare-shared-img">
                </div>

                <!-- 四列热力图网格 -->
                <div class="compare-grid">
                    <template x-for="mk in modelOrder">
                        <div class="compare-slot" :class="getSlotClass(mk)" :key="mk">
                            <!-- 色标竖线 -->
                            <div class="compare-slot-accent"
                                 :style="{ '--algo-color': mk === 'patchcore' ? '#2997ff' : mk === 'padim' ? '#30d158' : mk === 'fre' ? '#ff9f0a' : '#bf5af2' }"></div>

                            <!-- 头部：模型名 + 徽章 -->
                            <div class="compare-slot-header">
                                <h3 x-text="getModelName(mk)"></h3>
                                <span class="compare-slot-badge"
                                      x-show="slotIsDone(mk) && compareSlots[mk].data"
                                      :class="compareSlots[mk]?.data?.is_anomaly ? 'badge-anomaly' : 'badge-normal'"
                                      x-text="compareSlots[mk]?.data?.is_anomaly ? '异常' : '正常'"></span>
                            </div>

                            <!-- 等待状态 -->
                            <div class="compare-slot-pending" x-show="slotIsPending(mk)">
                                <span class="compare-slot-placeholder">等待推理...</span>
                            </div>

                            <!-- 推理中状态 -->
                            <div class="compare-slot-active" x-show="slotIsActive(mk)">
                                <div class="compare-skeleton">
                                    <div class="skeleton-row w-80 h-12"></div>
                                    <div class="skeleton-row w-60 h-12"></div>
                                    <div class="skeleton-row w-90 h-8"></div>
                                </div>
                                <div class="compare-spinner"></div>
                            </div>

                            <!-- 错误状态 -->
                            <div class="compare-slot-error" x-show="slotIsError(mk)">
                                <span class="error-icon">&#9888;</span>
                                <span x-text="compareSlots[mk].error"></span>
                            </div>

                            <!-- 完成状态：仅热力图 + 指标 -->
                            <template x-if="slotIsDone(mk) && compareSlots[mk].data">
                            <div class="compare-slot-result">
                                <div class="compare-heatmap-wrap"
                                     :id="'compare-wrap-' + mk"
                                     :data-bboxes="JSON.stringify(compareSlots[mk].data.bboxes || [])">
                                    <img :src="compareSlots[mk].data.heatmap_b64"
                                         :alt="getModelName(mk) + ' 热力图'"
                                         class="compare-heatmap"
                                         :id="'compare-heatmap-' + mk"
                                         @load="setupCompareBbox(mk)">
                                </div>
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
                            </div>
                            </template>
                        </div>
                    </template>
                </div>

                <!-- 页脚（内嵌于 S2） -->
                <footer class="footer scroll-reveal">
                    <p>工业缺陷检测系统 &middot; Anomalib 2.3 &middot; 无监督缺陷检测</p>
                    <p class="footer-sub">Powered by PatchCore / PaDiM / FRE / DRAEM</p>
                </footer>
            </div>
        </section>

    </div><!-- /.snap-container -->

    <!-- Toast 通知容器 -->
    <div class="toast-area" style="position:fixed;bottom:24px;left:50%;transform:translateX(-50%);z-index:10001;display:flex;flex-direction:column;gap:8px;pointer-events:none;">
        <template x-for="t in toasts" :key="t.id">
            <div class="toast-pill" :class="'toast--' + t.type" x-text="t.message"
                 style="pointer-events:auto;"
                 x-show="true"
                 x-transition.duration.300ms></div>
        </template>
    </div>

    <!-- JavaScript（app.js 必须在 Alpine CDN 之前加载） -->
    <script src="/static/js/animations.js"></script>
    <script src="/static/js/cursor-glow.js"></script>
    <script src="/static/js/inference.js"></script>
    <script src="/static/js/compare.js"></script>
    <script src="/static/js/app.js"></script>
    <script src="/static/js/flowchart.js"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3.14.9/dist/cdn.min.js"></script>

    <script>
        document.addEventListener('alpine:initialized', function () {
            setTimeout(function () {
                if (window.initAllAnimations) window.initAllAnimations();
            }, 300);
        });
    </script>
</body>
```

- [ ] **Step 5: 提交**

```bash
git add modules/ui/static/index.html
git commit -F .git-msg
```

`.git-msg` 内容：
```
feat(ui): 重构 HTML — snap 滚动容器 + 三 section 新布局

- 添加 .snap-container 包裹三个 snap-page
- 移除 transition-zone，footer 移入 S2 底部
- 导航栏改为 fixed，新增 section 标题接力
- S0: 算法卡片增加色标竖线和图标
- S1: 三列流水线布局 (上传→选择→推理) + 结果左右分栏
- S2: 共享原图上置 + 四列仅显示热力图
- 导航点改为进度环 + 页码显示
```

---

### Task 2: CSS — Snap 容器 + 全局布局 + 导航变更

**Files:**
- Modify: `modules/ui/static/css/app.css` (新增 snap 相关样式 + 修改导航部分)

- [ ] **Step 1: 在 `:root` 块后、全局重置前，添加 snap 滚动容器样式**

```css
/* ═══════════════════════════════════════════════════════════════════════════
   Snap 滚动容器 — 全视口吸附滚动
   ═══════════════════════════════════════════════════════════════════════════ */
html {
    height: 100%;
    overflow: hidden;
}

body {
    height: 100%;
    overflow: hidden;  /* body 不滚动，滚动交给 .snap-container */
}

.snap-container {
    height: 100%;
    overflow-y: scroll;
    overflow-x: hidden;
    scroll-snap-type: y mandatory;
    scroll-behavior: smooth;
    -webkit-overflow-scrolling: touch;
    /* 隐藏滚动条 — 吸附页面不需要 */
    scrollbar-width: none;
    -ms-overflow-style: none;
}

.snap-container::-webkit-scrollbar {
    display: none;
}

.snap-page {
    min-height: 100dvh;
    scroll-snap-align: start;
    display: flex;
    flex-direction: column;
    justify-content: center;
    position: relative;
    padding: 0;
}

.snap-page-inner {
    width: 100%;
    max-width: 1200px;
    margin: 0 auto;
    padding: max(48px, 6vh) 32px;
}
```

- [ ] **Step 2: 修改导航栏为 fixed**

将现有 `.navbar` 的 `position: sticky` 改为 `position: fixed`，调整宽度和层级：

```css
.navbar {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    z-index: 100;
    backdrop-filter: blur(12px) saturate(140%);
    -webkit-backdrop-filter: blur(12px) saturate(140%);
    --nav-opacity: 0;
    background: var(--bg-root);
    border-bottom: 1px solid transparent;
    transition: background 400ms var(--ease-out),
                border-color 400ms var(--ease-out);
    padding: 0 32px;
    height: 48px;
}

.navbar--scrolled {
    border-bottom-color: var(--sep-subtle);
}

html[data-theme="light"] .navbar {
    background: rgba(240, 240, 240, 0.65);
}

html[data-theme="light"] .navbar--scrolled {
    background: rgba(240, 240, 240, 0.85);
    border-bottom-color: rgba(0, 0, 0, 0.08);
}

.navbar-inner {
    display: flex;
    justify-content: space-between;
    align-items: center;
    max-width: 1200px;
    margin: 0 auto;
    height: 48px;
}

/* section 标题接力 */
.navbar-section {
    font-family: var(--font-display);
    font-size: 13px;
    font-weight: 500;
    color: var(--text-secondary);
    padding-left: 12px;
    border-left: 1px solid var(--sep-default);
    margin-left: 8px;
    white-space: nowrap;
}
```

- [ ] **Step 3: 修改导航点为进度环**

替换现有 `.nav-dots` 样式：

```css
/* ═══════════════════════════════════════════════════════════════════════════
   进度环导航点
   ═══════════════════════════════════════════════════════════════════════════ */
.snap-dots {
    position: fixed;
    right: 24px;
    top: 50%;
    transform: translateY(-50%);
    z-index: 99;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 6px;
}

.snap-ring {
    display: block;
    filter: drop-shadow(0 0 6px var(--accent-dim));
    transition: filter 0.3s var(--ease-out);
}

.snap-ring circle:last-child {
    transition: stroke-dashoffset 0.3s var(--ease-out);
}

.snap-dot-label {
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--text-tertiary);
    font-weight: 500;
    letter-spacing: 0.02em;
}
```

- [ ] **Step 4: 删除旧的 transition-zone 和 nav-dots 样式**

删除 `.transition-zone`, `.transition-line`, `@keyframes transitionLinePulse`, `.nav-dots`, `.nav-dot`, `.nav-dot.active`, `.nav-dot:hover` 规则。

- [ ] **Step 5: 修改 section 通用样式**

```css
/* ═══════════════════════════════════════════════════════════════════════════
   Section 通用样式
   ═══════════════════════════════════════════════════════════════════════════ */
/* 删除旧 .section 规则，snap-page-inner 替代 */

.section-title {
    font-family: var(--font-display);
    font-size: 32px;
    font-weight: 600;
    letter-spacing: -0.02em;
    text-align: center;
    margin-bottom: 40px;
}

/* 滚动提示箭头 */
.scroll-hint {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
    margin-top: 40px;
    animation: scrollHintBounce 2s ease-in-out infinite;
}

.scroll-hint span {
    font-size: 12px;
    color: var(--text-tertiary);
    letter-spacing: 0.04em;
}

@keyframes scrollHintBounce {
    0%, 100% { transform: translateY(0); opacity: 0.5; }
    50%      { transform: translateY(8px); opacity: 1; }
}
```

- [ ] **Step 6: 提交**

```bash
git add modules/ui/static/css/app.css
git commit -F .git-msg
```

`.git-msg` 内容：
```
feat(ui): 添加 snap 滚动容器 + 固定导航栏 + 进度环样式
```

---

### Task 3: CSS — Section 0 算法卡片重新设计

**Files:**
- Modify: `modules/ui/static/css/app.css` (追加 algo-card 新样式)

- [ ] **Step 1: 重写 Hero 区域样式**

```css
/* ═══════════════════════════════════════════════════════════════════════════
   Hero 区域 — Section 0
   ═══════════════════════════════════════════════════════════════════════════ */
.hero {
    padding: 0 0 48px;
    text-align: center;
}

.hero-title {
    font-family: var(--font-display);
    font-size: 64px;
    font-weight: 700;
    letter-spacing: -0.03em;
    line-height: 1.1;
    color: var(--text);
}

.hero-subtitle {
    font-family: var(--font-body);
    font-size: 19px;
    color: var(--text-secondary);
    margin-top: 12px;
    line-height: 1.5;
}
```

- [ ] **Step 2: 重写算法卡片网格和配色**

```css
/* ═══════════════════════════════════════════════════════════════════════════
   算法卡片网格
   ═══════════════════════════════════════════════════════════════════════════ */
.algo-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 32px;
    padding: 0;
}

/* 卡片基础 — 替换旧 .flowchart-card */
.algo-card {
    background: var(--bg-system);
    border: 1px solid var(--sep-subtle);
    border-radius: var(--r-lg);
    padding: 36px;
    position: relative;
    overflow: hidden;
    transition: transform 0.35s var(--ease-spring),
                box-shadow var(--dur-normal) var(--ease-out),
                border-color var(--dur-normal) var(--ease-out);
    display: flex;
    flex-direction: column;
}

.algo-card:hover {
    transform: translateY(-2px);
    border-color: var(--sep-default);
    box-shadow: var(--shadow-md);
}

/* 左侧色标竖线 */
.algo-card-accent {
    position: absolute;
    left: 0;
    top: 20px;
    bottom: 20px;
    width: 3px;
    border-radius: 0 3px 3px 0;
    background: var(--algo-color, var(--accent));
    transition: box-shadow 0.35s var(--ease-out);
}

.algo-card:hover .algo-card-accent {
    box-shadow: 0 0 12px var(--algo-color, var(--accent));
}

/* 卡片内容 */
.algo-card-body {
    padding-left: 8px;
}

.algo-card h3 {
    font-family: var(--font-display);
    font-size: 22px;
    font-weight: 600;
    margin: 0 0 8px 0;
    color: var(--text);
    display: flex;
    align-items: center;
    gap: 8px;
    letter-spacing: -0.01em;
}

.algo-icon {
    font-size: 20px;
    line-height: 1;
}

.algo-card p {
    font-size: 14px;
    color: var(--text-secondary);
    line-height: 1.6;
    margin: 0 0 4px 0;
}

.algo-card-kicker {
    font-size: 13px !important;
    color: var(--text-tertiary) !important;
    font-style: italic;
}

/* 标签 */
.algo-tag {
    font-size: 11px;
    font-weight: 500;
    padding: 2px 8px;
    border-radius: 20px;
    margin-left: 4px;
    letter-spacing: 0;
}

.algo-tag--rec {
    background: rgba(41, 151, 255, 0.15);
    color: #2997ff;
}

.algo-tag--alt {
    background: rgba(48, 209, 88, 0.15);
    color: #30d158;
}
```

- [ ] **Step 3: 删除旧的 .flowchart-card 样式（从 flowchart.css 移除到 app.css 的 .algo-card 替代）**

`.flowchart-card` 被 `.algo-card` 替代。保留 `.flowchart-card` 在 flowchart.css 中但不再使用。

- [ ] **Step 4: 提交**

```bash
git add modules/ui/static/css/app.css modules/ui/static/css/flowchart.css
git commit -F .git-msg
```

`.git-msg` 内容：
```
feat(ui): 重写 Section 0 — Hero 64px 标题 + 色标卡片 + 算法图标
```

---

### Task 4: CSS — Section 1 三列流水线 + 左右分栏结果

**Files:**
- Modify: `modules/ui/static/css/app.css` (追加 pipeline + result-layout 样式)

- [ ] **Step 1: 添加三列流水线样式**

```css
/* ═══════════════════════════════════════════════════════════════════════════
   三列流水线 — Section 1
   ═══════════════════════════════════════════════════════════════════════════ */
.pipeline {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 20px;
    margin-bottom: 24px;
    align-items: start;
}

.pipeline-step {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 20px;
    background: var(--bg-secondary);
    border-radius: var(--r-md);
    border: 1px solid var(--sep-subtle);
    transition: border-color var(--dur-normal) var(--ease-out),
                box-shadow var(--dur-normal) var(--ease-out);
}

.pipeline-step--done {
    border-color: var(--ok);
    box-shadow: 0 0 0 1px var(--ok-bg);
}

.pipeline-step-num {
    width: 28px;
    height: 28px;
    border-radius: 50%;
    background: var(--bg-tertiary);
    color: var(--text-secondary);
    font-family: var(--font-mono);
    font-size: 13px;
    font-weight: 600;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    transition: background var(--dur-fast) var(--ease-out),
                color var(--dur-fast) var(--ease-out);
}

.pipeline-step--done .pipeline-step-num {
    background: var(--ok-bg);
    color: var(--ok);
}

.pipeline-step-body {
    flex: 1;
    min-width: 0;
}

/* 流水线内的上传区缩小 */
.pipeline-step .upload-zone {
    min-height: auto;
    border: 1px dashed var(--sep-default);
    border-radius: var(--r-sm);
    padding: 12px;
}

.pipeline-step .upload-placeholder {
    padding: 16px 12px;
}

.pipeline-step .upload-icon-svg {
    width: 28px;
    height: 28px;
    margin-bottom: 8px;
}

.pipeline-step .upload-text {
    font-size: 13px;
}

.pipeline-step .upload-hint {
    font-size: 11px;
}

.pipeline-step .upload-preview {
    padding: 8px;
}

.pipeline-step .upload-preview img {
    max-height: 120px;
}

.pipeline-step .model-select {
    width: 100%;
    font-size: 13px;
    padding: 8px 28px 8px 10px;
}

.pipeline-step .btn-inference {
    width: 100%;
    padding: 10px 16px;
    font-size: 14px;
}

/* 推理控制区移入 .inference-actions */
.inference-actions {
    display: flex;
    justify-content: center;
    margin-top: 20px;
}
```

- [ ] **Step 2: 添加结果左右分栏样式**

```css
/* ═══════════════════════════════════════════════════════════════════════════
   结果面板 — 左右分栏
   ═══════════════════════════════════════════════════════════════════════════ */
.result-layout {
    display: grid;
    grid-template-columns: 6fr 4fr;
    gap: 24px;
    align-items: start;
}

.result-left {
    min-width: 0;
}

.result-right {
    min-width: 0;
}

/* 右栏指标卡自适应 */
.result-right .result-card {
    height: 100%;
}

/* 阈值行移除（紧凑化） */
.compare-metric-value--small {
    font-size: 14px;
    color: var(--text-secondary);
}

/* 推理区域 outer — 不再需要大卡片包裹 */
.inference-area {
    /* 去掉旧的大边框卡片，改为透明 */
    background: transparent;
    border: none;
    border-radius: 0;
    padding: 0;
}
```

- [ ] **Step 3: 提交**

```bash
git add modules/ui/static/css/app.css
git commit -F .git-msg
```

`.git-msg` 内容：
```
feat(ui): 添加三列流水线 + 结果左右分栏样式
```

---

### Task 5: CSS — Section 2 共享原图 + 四列色标

**Files:**
- Modify: `modules/ui/static/css/app.css` (追加 compare-shared-image + compare-slot-accent 样式)

- [ ] **Step 1: 添加共享原图和色标样式**

```css
/* ═══════════════════════════════════════════════════════════════════════════
   四模型对比 — Section 2 重排
   ═══════════════════════════════════════════════════════════════════════════ */

/* 共享原图 */
.compare-shared-image {
    position: relative;
    width: 100%;
    max-width: 640px;
    margin: 0 auto 24px;
    border-radius: var(--r-lg);
    overflow: hidden;
    background: var(--bg-secondary);
}

.compare-shared-label {
    position: absolute;
    top: 8px;
    left: 8px;
    font-size: 10px;
    font-weight: 500;
    color: var(--text-secondary);
    background: var(--bg-secondary);
    padding: 3px 10px;
    border-radius: 4px;
    opacity: 0.8;
    pointer-events: none;
    z-index: 1;
}

.compare-shared-img {
    display: block;
    width: 100%;
    max-height: 240px;
    object-fit: contain;
}

/* 槽位色标竖线 */
.compare-slot-accent {
    position: absolute;
    left: 0;
    top: 16px;
    bottom: 16px;
    width: 3px;
    border-radius: 0 3px 3px 0;
    background: var(--algo-color, var(--accent));
    transition: box-shadow 0.3s var(--ease-out);
}

.compare-slot {
    position: relative;  /* 为色标提供定位参考 */
    overflow: hidden;
}

.compare-slot--done .compare-slot-accent {
    opacity: 1;
}

/* 紧凑指标字体调整 */
.compare-metric-value {
    font-size: 16px;
}

/* 热力图容器微调 */
.compare-slot .compare-heatmap {
    max-height: 180px;
}

/* 移除旧的 .compare-mini-label（原图标签移到共享原图上，热力图不再需要标签） */
.compare-mini-label,
.compare-mini-label--heat {
    display: none;
}

/* 摘要栏优化：横向紧凑 */
.compare-summary {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 10px 16px;
    padding: 14px 20px;
    background: var(--bg-secondary);
    border: 1px solid rgba(255, 159, 10, 0.25);
    border-radius: var(--r-md);
    margin-bottom: 20px;
}

.compare-summary-trophy {
    font-size: 20px;
    line-height: 1;
}

.compare-summary-best {
    font-size: 14px;
    color: var(--text);
}

.compare-summary-best strong {
    color: var(--warn);
}

.compare-summary-best code {
    font-family: var(--font-mono);
    background: var(--bg-tertiary);
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 12px;
    color: var(--text);
    margin-left: 6px;
}
```

- [ ] **Step 2: 提交**

```bash
git add modules/ui/static/css/app.css
git commit -F .git-msg
```

`.git-msg` 内容：
```
feat(ui): 添加共享原图 + 四列色标竖线 + 摘要栏横向优化
```

---

### Task 6: CSS — Snap 页面进出动画

**Files:**
- Modify: `modules/ui/static/css/app.css` (追加动画关键帧 + 滚动驱动规则)

- [ ] **Step 1: 添加页面级进出动画**

```css
/* ═══════════════════════════════════════════════════════════════════════════
   Snap 页面进出动画
   ═══════════════════════════════════════════════════════════════════════════ */

/* 页面进入：内容从下方淡入 */
@keyframes snapPageEnter {
    from {
        opacity: 0;
        transform: translateY(32px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* 页面离开：内容向上淡出 */
@keyframes snapPageExit {
    from {
        opacity: 1;
        transform: translateY(0);
    }
    to {
        opacity: 0;
        transform: translateY(-16px);
    }
}

/* 逐级延迟入场 */
.snap-page .scroll-reveal-stagger > *:nth-child(1) { animation-delay: 0ms; }
.snap-page .scroll-reveal-stagger > *:nth-child(2) { animation-delay: 120ms; }
.snap-page .scroll-reveal-stagger > *:nth-child(3) { animation-delay: 240ms; }
.snap-page .scroll-reveal-stagger > *:nth-child(4) { animation-delay: 360ms; }

/* Chrome 115+: 使用 ViewTimeline 在吸附时触发动画 */
@supports (animation-timeline: view()) {
    .snap-page .scroll-reveal {
        animation: snapPageEnter 0.6s cubic-bezier(0.16, 1, 0.3, 1) both;
        animation-timeline: view(block 80% 20%);
    }

    .snap-page .section-title {
        animation: snapPageEnter 0.5s cubic-bezier(0.16, 1, 0.3, 1) both;
        animation-timeline: view(block 85% 15%);
    }
}

/* Hero 标题接力过渡 */
.hero-title {
    transition: opacity 0.4s var(--ease-out),
                transform 0.5s var(--ease-out-expo);
}

/* 当 section 不完全可见时 Hero 缩小淡出（由 JS 添加 class） */
.snap-page--exiting .hero-title {
    opacity: 0;
    transform: scale(0.95);
}
```

- [ ] **Step 2: 更新 scroll-reveal 基类**

```css
/* 更新旧 .scroll-reveal（保留兼容 animations.js 的 Observer 驱动方案） */
.scroll-reveal {
    opacity: 0;
    transform: translateY(24px);
    /* WAAPI 由 animations.js 的 initScrollReveal 在元素进入视口时覆盖 */
}
```

- [ ] **Step 3: 提交**

```bash
git add modules/ui/static/css/app.css
git commit -F .git-msg
```

`.git-msg` 内容：
```
feat(ui): 添加 snap 页面进出动画 + ViewTimeline 渐进增强
```

---

### Task 7: JS — app.js Snap 事件 + 进度环 + 标题接力

**Files:**
- Modify: `modules/ui/static/js/app.js` (修改导航逻辑，新增 snapProgress 状态)

- [ ] **Step 1: 在 Alpine data 中添加 snapProgress 状态**

在 `app.js` 的 `Alpine.data('app', function () { return {` 块中，修改以下部分：

```javascript
// 在 currentSection: 0 之后添加：
snapProgress: 0,  // 0.0 ~ 1.0，表示两页之间的滚动进度
sectionCount: 3,
```

- [ ] **Step 2: 修改导航相关逻辑**

替换 `setupScrollObserver` 方法：

```javascript
setupScrollObserver: function () {
    var self = this;
    var container = self.$refs.snapContainer;
    if (!container) return;

    // 导航栏加深（基于 snap 容器滚动位置）
    container.addEventListener('scroll', function () {
        self.scrolled = container.scrollTop > 50;
    }, { passive: true });

    // ── IntersectionObserver 检测当前 section ──
    var sections = [];
    if (self.$refs.section0) sections.push(self.$refs.section0);
    if (self.$refs.section1) sections.push(self.$refs.section1);
    if (self.$refs.section2) sections.push(self.$refs.section2);

    if (sections.length === 0) return;

    var observer = new IntersectionObserver(
        function (entries) {
            var maxRatio = 0;
            var maxIdx = self.currentSection;

            entries.forEach(function (entry) {
                if (entry.intersectionRatio > maxRatio) {
                    maxRatio = entry.intersectionRatio;
                    var idx = sections.indexOf(entry.target);
                    if (idx >= 0) maxIdx = idx;
                }
            });

            if (maxRatio > 0 && maxIdx !== self.currentSection) {
                // 为离开的 section 添加 exiting class
                var prevSection = sections[self.currentSection];
                if (prevSection) {
                    prevSection.classList.add('snap-page--exiting');
                }
                // 为新进入的 section 移除 exiting class
                var nextSection = sections[maxIdx];
                if (nextSection) {
                    nextSection.classList.remove('snap-page--exiting');
                }
                self.currentSection = maxIdx;
            }

            // 计算 snap 进度（用于进度环）
            var totalHeight = container.scrollHeight - container.clientHeight;
            if (totalHeight > 0) {
                self.snapProgress = Math.min(1, container.scrollTop / totalHeight);
            }
        },
        { threshold: [0, 0.15, 0.3, 0.5, 0.7, 0.85, 1] }
    );

    sections.forEach(function (s) {
        observer.observe(s);
    });

    // ── 滚动事件更新进度环 ──
    container.addEventListener('scroll', function () {
        var totalHeight = container.scrollHeight - container.clientHeight;
        if (totalHeight > 0) {
            self.snapProgress = Math.min(1, container.scrollTop / totalHeight);
        }
    }, { passive: true });

    // ── 键盘导航（↑↓ 切换 section）──
    window.addEventListener('keydown', function (e) {
        if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
            var tag = document.activeElement ? document.activeElement.tagName : '';
            if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;

            e.preventDefault();
            var dir = e.key === 'ArrowDown' ? 1 : -1;
            var next = Math.max(0, Math.min(sections.length - 1, self.currentSection + dir));
            self.currentSection = next;
            self.scrollToSection(next);
        }
    });
},
```

- [ ] **Step 3: 修改 scrollToSection**

```javascript
scrollToSection: function (idx) {
    var sections = [];
    if (this.$refs.section0) sections.push(this.$refs.section0);
    if (this.$refs.section1) sections.push(this.$refs.section1);
    if (this.$refs.section2) sections.push(this.$refs.section2);

    var target = sections[idx];
    if (target) {
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
},
```

- [ ] **Step 4: 提交**

```bash
git add modules/ui/static/js/app.js
git commit -F .git-msg
```

`.git-msg` 内容：
```
feat(ui): app.js 适配 snap 滚动 — 进度环计算 + section exiting class + 标题接力
```

---

### Task 8: JS — animations.js Snap 感知的页面过渡

**Files:**
- Modify: `modules/ui/static/js/animations.js` (追加 snapTransition 方法)

- [ ] **Step 1: 追加 snap section 过渡编排函数**

在 `Anim` 对象末尾（`springTo` 方法之后）、全局 `initAllAnimations` 之前，添加：

```javascript
/**
 * Snap 页面过渡编排
 * 当进入新 section 时，触发对应 section 内子元素的逐级入场动画。
 *
 * @param {Element} section - 进入的 section 元素
 * @param {Object} options
 * @param {number} options.staggerMs - 子元素间延迟 (ms)
 * @param {number} options.duration - 单个动画时长 (ms)
 */
snapPageEnter(section, options = {}) {
    const { staggerMs = 100, duration = 500 } = options;
    // 对 section 内所有带 .scroll-reveal 的子元素触发入场
    const children = section.querySelectorAll(':scope > .snap-page-inner > * > .scroll-reveal, :scope > .snap-page-inner > .scroll-reveal');
    const animations = [];
    children.forEach((child, i) => {
        // 重置初始状态
        child.style.opacity = '0';
        child.style.transform = 'translateY(24px)';
        animations.push(
            child.animate(
                [
                    { opacity: 0, transform: 'translateY(24px)' },
                    { opacity: 1, transform: 'translateY(0)' }
                ],
                {
                    duration,
                    delay: i * staggerMs,
                    easing: 'cubic-bezier(0.16, 1, 0.3, 1)',
                    fill: 'forwards'
                }
            )
        );
    });
    return animations;
},

/**
 * Snap 页面离开动画
 * @param {Element} section - 离开的 section 元素
 */
snapPageExit(section) {
    const children = section.querySelectorAll(':scope > .snap-page-inner > * > .scroll-reveal, :scope > .snap-page-inner > .scroll-reveal');
    const animations = [];
    children.forEach((child) => {
        animations.push(
            child.animate(
                [
                    { opacity: 1, transform: 'translateY(0)' },
                    { opacity: 0, transform: 'translateY(-16px)' }
                ],
                {
                    duration: 300,
                    easing: 'cubic-bezier(0, 0, 0.2, 1)',
                    fill: 'forwards'
                }
            )
        );
    });
    return animations;
}
```

- [ ] **Step 2: 提交**

```bash
git add modules/ui/static/js/animations.js
git commit -F .git-msg
```

`.git-msg` 内容：
```
feat(ui): animations.js 新增 snapPageEnter/snapPageExit 过渡编排
```

---

### Task 9: 响应式调整

**Files:**
- Modify: `modules/ui/static/css/app.css` (追加/修改媒体查询)

- [ ] **Step 1: 更新平板响应式规则**

在现有 `@media (max-width: 768px)` 块中追加：

```css
@media (max-width: 768px) {
    /* Snap pages: reduce padding */
    .snap-page-inner {
        padding: 32px 16px;
    }

    /* Pipeline: stack vertically */
    .pipeline {
        grid-template-columns: 1fr;
        gap: 12px;
    }

    /* Result layout: stack */
    .result-layout {
        grid-template-columns: 1fr;
        gap: 16px;
    }

    /* Hero smaller */
    .hero-title { font-size: 40px; }
    .hero-subtitle { font-size: 16px; }

    /* Algo grid: single column */
    .algo-grid { grid-template-columns: 1fr; gap: 20px; }

    /* Compare grid: 2x2 */
    .compare-grid { grid-template-columns: repeat(2, 1fr); gap: 12px; }

    /* Section title smaller */
    .section-title { font-size: 26px; }

    /* Progress ring: move to bottom */
    .snap-dots {
        position: fixed;
        right: auto; top: auto;
        bottom: 0; left: 0;
        transform: none;
        flex-direction: row;
        justify-content: center;
        gap: 12px;
        padding: 12px 0;
        background: var(--bg-root);
        border-top: 1px solid var(--sep-subtle);
        width: 100%;
    }
    .snap-ring { width: 28px; height: 28px; }

    /* Add bottom padding for nav dots */
    .snap-container { padding-bottom: 56px; }
}
```

- [ ] **Step 2: 更新手机响应式规则**

在现有 `@media (max-width: 480px)` 块中追加：

```css
@media (max-width: 480px) {
    .hero-title { font-size: 32px; }
    .snap-page-inner { padding: 24px 12px; }
    .section-title { font-size: 22px; }
    .algo-grid { gap: 12px; }
    .algo-card { padding: 24px; }
    .algo-card h3 { font-size: 18px; }
    .compare-grid { grid-template-columns: 1fr; }
    .compare-shared-img { max-height: 160px; }
    .pipeline-step { padding: 14px; }
    .pipeline-step-num { width: 24px; height: 24px; font-size: 11px; }
}
```

- [ ] **Step 3: 提交**

```bash
git add modules/ui/static/css/app.css
git commit -F .git-msg
```

`.git-msg` 内容：
```
feat(ui): 更新 snap 布局响应式 — 平板堆叠 + 手机单列 + 底部进度环
```

---

### Task 10: 浏览器验证

- [ ] **Step 1: 启动服务**

```bash
python scripts/run_ui.py
```

- [ ] **Step 2: 验证清单**

在浏览器中打开 `http://127.0.0.1:8000`，逐项检查：

| # | 检查项 | 预期行为 |
|---|--------|---------|
| 1 | 页面加载 | 加载遮罩淡出，S0 全屏显示 |
| 2 | 滚动吸附 | 每次滚轮/swipe 吸附到整页，无半页停留 |
| 3 | 进度环 | 右侧进度环显示 "1/3"，随滚动填充 |
| 4 | S0 算法卡片 | 左侧色标竖线可见，hover 发光 |
| 5 | S0 流程图 | SVG 进入视口时绘制动画触发 |
| 6 | S1 三列流水线 | 上传→选择→推理，水平排列 |
| 7 | S1 上传图片 | 拖拽/点击上传正常，预览显示 |
| 8 | S1 推理 | 选择模型→开始推理→进度→结果面板 |
| 9 | S1 结果左右分栏 | 左侧滑块可拖动，右侧指标卡显示 |
| 10 | S1 热力图 tooltip | 鼠标 hover 显示异常得分 |
| 11 | S2 共享原图 | 对比完成后显示原图 |
| 12 | S2 四列热力图 | 每列色标竖线匹配算法颜色，指标数据正确 |
| 13 | 亮/暗切换 | 胶囊开关正常，所有颜色跟随 |
| 14 | 键盘导航 | ↑↓ 键切换 section |
| 15 | 导航栏标题接力 | 离开 S0 后显示"单模型推理"/"四模型对比" |
| 16 | 手机布局 (≤480px) | 流水线单列、对比单列 |
| 17 | 暗色模式 | 所有元素暗色正确，无白色闪烁 |
| 18 | 亮色模式 | 切换到亮色，所有元素正确 |

- [ ] **Step 3: 修复发现的问题**

根据验证结果修复 CSS/JS bug。

- [ ] **Step 4: 最终提交**

```bash
git add -A
git commit -F .git-msg
```

`.git-msg` 内容：
```
fix(ui): 浏览器验证后修复 — snap 滚动切页最终调整
```

---

## 自检

- [x] Spec 覆盖：所有设计规范章节均有对应 Task（架构→Task 1, S0→Task 1+3, S1→Task 1+4, S2→Task 1+5, 切页动画→Task 6+7+8, 响应式→Task 9, 验证→Task 10）
- [x] 无占位符：所有代码步骤均为完整可运行的代码块
- [x] 类型一致：`snapProgress` 在 Task 7 的 app.js 中定义为 `0`（number），在 Task 1 的 HTML 中通过 `:stroke-dashoffset="106.8 - 106.8 * snapProgress"` 使用——类型匹配
- [x] CSS class 向后兼容：`inference.js` / `compare.js` 依赖的所有 CSS class 均保留（`.compare-container`, `.compare-heatmap`, `.compare-slot`, `#compare-wrap-*`, `#compare-heatmap-*` 等）
- [x] Alpine refs 保留：`$refs.section0/1/2`, `$refs.pageLoader`, `$refs.fileInput`, `$refs.anomalyMapData`, `$refs.bboxData` 均在 HTML 中保留
- [x] API 无变化：后端无需任何修改
