# 全站 AirPods Pro 风格 UI 改造综合规划（修订版）

## 1. 项目目标

在首页算法卡片 AirPods Pro 化改造基础上，将全站四个主要页面（算法介绍、单模型推理、Training Studio、四模型对比）以及训练动画统一推向 Apple AirPods Pro 式视觉语言：

- **明亮自信的主调**
- **产品/结果作为视觉主角**
- **大量留白与清晰层级**
- **从容的过渡动画**
- **全局统一的颜色模式**

参考标杆：https://www.apple.com/airpods-pro/

---

## 2. 设计原则（摘自 AirPods Pro 分析）

| 原则 | 表现 | 应用方式 |
|------|------|----------|
| **一次只说一件事** | 每个模块聚焦一个核心信息 | 每页只有一个主角，其他控件退后 |
| **极端字号对比** | 标题极大，辅助信息极小 | 结果图/算法名最大，参数标签最小 |
| **大量留白** | 元素间距是行高的 2-3 倍 | 增加 section padding 和卡片间距 |
| **产品为视觉中心** | 产品图居中，文字环绕 | 结果热力图、训练曲线、对比图居中 |
| **从容的动效** | 大过渡 800-1200ms | 页面切换、卡片切换用慢速高级曲线 |
| **全局一致性** | 全站使用统一的亮/暗模式 | 不单独为某个章节切换主题 |
| **去 chrome 化** | 少用边框，多用阴影和留白 | 按钮/卡片/输入框简化 |

---

## 3. 全局系统改造

### 3.1 主题系统

**决策**：全站使用**统一的颜色模式**，默认**跟随系统偏好**，用户可手动切换。

理由：
- AirPods Pro 页面虽然以白色为主，但 Apple 全系产品都支持系统亮暗模式跟随
- 统一主题避免用户在不同页面间产生割裂感
- 便于未来接入 `prefers-color-scheme` 系统级切换

实现方式：

```javascript
// modules/ui/static/theme.js
(function() {
    var t = localStorage.getItem('theme');
    if (!t) {
        // 优先跟随系统
        t = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    }
    document.documentElement.setAttribute('data-theme', t);

    // 监听系统主题变化
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', function(e) {
        if (!localStorage.getItem('theme')) {
            document.documentElement.setAttribute('data-theme', e.matches ? 'dark' : 'light');
        }
    });
})();
```

手动切换按钮保留，切换后写入 `localStorage`，优先级高于系统偏好。

### 3.2 色彩系统精修

调整 `modules/ui/static/css/app.css` `:root` 与 `html[data-theme="dark"]`：

```css
/* 亮色默认 */
:root {
    --bg-root: #f5f5f7;
    --bg-system: #ffffff;
    --bg-secondary: #f2f2f4;
    --bg-tertiary: #e8e8ed;

    --sep-subtle: rgba(0, 0, 0, 0.06);
    --sep-default: rgba(0, 0, 0, 0.10);
    --sep-strong: rgba(0, 0, 0, 0.16);

    --text: rgba(0, 0, 0, 0.88);
    --text-secondary: rgba(0, 0, 0, 0.55);
    --text-tertiary: rgba(0, 0, 0, 0.30);

    --accent: #0071e3;        /* Apple 官网蓝 */
    --accent-hover: #0077ed;
    --accent-pressed: #005bb5;
    --accent-dim: rgba(0, 113, 227, 0.10);
    --accent-glow: rgba(0, 113, 227, 0.18);
}

/* 暗色覆盖 */
html[data-theme="dark"] {
    --bg-root: #0a0a0b;
    --bg-system: rgba(255, 255, 255, 0.055);
    --bg-secondary: rgba(255, 255, 255, 0.09);
    --bg-tertiary: rgba(255, 255, 255, 0.13);

    --sep-subtle: rgba(255, 255, 255, 0.07);
    --sep-default: rgba(255, 255, 255, 0.11);
    --sep-strong: rgba(255, 255, 255, 0.17);

    --text: rgba(255, 255, 255, 0.92);
    --text-secondary: rgba(255, 255, 255, 0.60);
    --text-tertiary: rgba(255, 255, 255, 0.34);

    --accent: #2997ff;
    --accent-hover: #47a9ff;
    --accent-pressed: #0070d6;
    --accent-dim: rgba(41, 151, 255, 0.14);
    --accent-glow: rgba(41, 151, 255, 0.30);
}
```

### 3.3 字体比例系统

建立六级字体比例，参考 AirPods Pro：

```css
:root {
    --text-hero: clamp(56px, 8vw, 112px);      /* Hero 标题 */
    --text-display: clamp(40px, 5vw, 72px);    /* Section 大标题 */
    --text-headline: clamp(28px, 3vw, 44px);   /* 卡片标题/算法名 */
    --text-title: 24px;                         /* 面板标题 */
    --text-body: 17px;                          /* 正文 */
    --text-caption: 13px;                       /* 标签/参数 */
    --text-footnote: 11px;                      /* 脚注 */
}
```

大标题行高：`0.95 - 1.05`
正文行高：`1.5 - 1.6`
字重：标题 600-700，正文 400，标签 500-600

### 3.4 按钮组件统一

**主按钮（Primary）**：Apple 式黑色填充药丸

```css
.btn-primary {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    padding: 12px 24px;
    border-radius: 100px;
    background: #1d1d1f;
    color: #ffffff;
    font-size: 15px;
    font-weight: 500;
    border: none;
    cursor: pointer;
    transition: transform 0.2s var(--ease-spring),
                background 0.2s var(--ease-out),
                box-shadow 0.2s var(--ease-out);
}
.btn-primary:hover {
    background: #000000;
    transform: scale(1.02);
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.15);
}
.btn-primary:active {
    transform: scale(0.98);
}
.btn-primary:disabled {
    opacity: 0.5;
    cursor: not-allowed;
    transform: none;
}
```

**次按钮（Secondary）**：细边框透明

```css
.btn-secondary {
    padding: 10px 20px;
    border-radius: 100px;
    background: transparent;
    color: var(--text);
    font-size: 15px;
    font-weight: 500;
    border: 1px solid var(--sep-default);
    cursor: pointer;
    transition: background 0.2s var(--ease-out), border-color 0.2s var(--ease-out);
}
.btn-secondary:hover {
    background: var(--bg-secondary);
    border-color: var(--sep-strong);
}
```

**文字按钮（Tertiary）**：带箭头

```css
.btn-text {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    color: var(--accent);
    font-size: 15px;
    font-weight: 500;
    background: none;
    border: none;
    cursor: pointer;
    transition: gap 0.2s var(--ease-out);
}
.btn-text:hover {
    gap: 8px;
}
```

### 3.5 输入框与选择器统一

```css
.input-field {
    width: 100%;
    padding: 10px 14px;
    border-radius: var(--r-md);
    background: var(--bg-secondary);
    border: 1px solid var(--sep-subtle);
    color: var(--text);
    font-size: 15px;
    transition: border-color 0.2s var(--ease-out), box-shadow 0.2s var(--ease-out);
}
.input-field:focus {
    outline: none;
    border-color: var(--accent);
    box-shadow: 0 0 0 3px var(--accent-dim);
}
```

自定义下拉选择器简化为更轻量的样式，减少玻璃模糊和装饰。

### 3.6 导航栏简化

当前导航栏已经较好，但可进一步：
- 标题更简洁：从「缺陷检测」改为品牌 logo + 名称
- 数据集选择器在不需要时隐藏或弱化
- 主题切换胶囊保留，但采用更 Apple 的样式

### 3.7 背景、光标光晕与噪点纹理的融合方式

**决策**：保留光标光晕和噪点纹理，但重新设计为更克制、更 AirPods Pro 式的环境细节。

#### 噪点纹理（Noise Texture）

问题：当前噪点可能过强，像胶片颗粒，与 AirPods Pro 的纯净感冲突。

优化方案：
- **仅在背景层使用**，绝不覆盖在卡片/文字上
- 透明度降到 **0.015 - 0.025**（当前可能是 0.04-0.06）
- 使用更细腻的颗粒（更小的噪点尺寸）
- **亮暗模式差异化**：
  - light 模式：噪点几乎不可见（opacity 0.01），避免破坏纯净感
  - dark 模式：噪点稍明显（opacity 0.025），增加材质深度
- 使用 `pointer-events: none` 和 `mix-blend-mode: overlay`

```css
body::after {
    content: '';
    position: fixed;
    inset: 0;
    background-image: url("data:image/svg+xml,..."); /* 细密噪点 */
    opacity: 0.015;
    pointer-events: none;
    z-index: 9999;
    mix-blend-mode: overlay;
}
html[data-theme="dark"] body::after {
    opacity: 0.025;
}
```

#### 光标光晕（Cursor Glow）

问题：当前光标光晕可能过大、过亮，像游戏 UI，分散注意力。

优化方案：
- **大幅缩小光晕尺寸**：从可能 400-600px 降到 **200-300px**
- **降低不透明度**：从 0.15-0.25 降到 **0.04-0.08**
- **只在 dark 模式显著**：light 模式下几乎不可见（opacity 0.02）
- **颜色改为跟随当前主题强调色或中性白光**：不要饱和度过高
- **仅在空白区域明显**：当光标经过卡片/文字上方时，光晕进一步减弱
- **增加滞后/平滑跟随**：使用 lerp 让移动更柔和

```css
body::before {
    content: '';
    position: fixed;
    left: calc(var(--cursor-x) * 1px);
    top: calc(var(--cursor-y) * 1px);
    width: 280px;
    height: 280px;
    border-radius: 50%;
    background: radial-gradient(circle, var(--accent-glow) 0%, transparent 70%);
    transform: translate(-50%, -50%);
    opacity: calc(var(--glow-intensity) * 0.06);
    pointer-events: none;
    z-index: 0;
    filter: blur(40px);
    transition: opacity 0.3s var(--ease-out);
}
html[data-theme="light"] body::before {
    opacity: calc(var(--glow-intensity) * 0.02);
}
```

JS 中增加平滑跟随：

```javascript
var cursorX = window.innerWidth / 2;
var cursorY = window.innerHeight / 2;
var targetX = cursorX;
var targetY = cursorY;

function updateCursor() {
    cursorX += (targetX - cursorX) * 0.08;
    cursorY += (targetY - cursorY) * 0.08;
    document.documentElement.style.setProperty('--cursor-x', cursorX);
    document.documentElement.style.setProperty('--cursor-y', cursorY);
    requestAnimationFrame(updateCursor);
}
```

#### 背景渐变光晕

在 Hero 区域添加极淡的径向渐变，与光标光晕形成层次：

```css
.hero {
    position: relative;
}
.hero::before {
    content: '';
    position: absolute;
    inset: -20%;
    background: radial-gradient(circle at 50% 40%, rgba(0, 113, 227, 0.04) 0%, transparent 50%);
    pointer-events: none;
    z-index: 0;
}
```

### 3.8 动效系统

建立统一的动画时长阶梯：

```css
:root {
    --dur-instant: 100ms;   /* hover 反馈 */
    --dur-fast: 200ms;      /* UI 状态切换 */
    --dur-normal: 400ms;    /* 内容 reveal */
    --dur-slow: 800ms;      /* 页面/卡片切换 */
    --dur-cinematic: 1200ms;/* Hero 入场 */
}
```

统一缓动：
- 入场：`cubic-bezier(0.16, 1, 0.3, 1)`（ease-out-expo）
- 交互反馈：`cubic-bezier(0.22, 0.8, 0.3, 1.15)`（spring）
- 平滑过渡：`cubic-bezier(0.25, 0.1, 0.25, 1)`

---

## 4. 分页改造方案

### 4.1 Section 0：算法介绍页

**状态**：已有详细规划，见 `docs/superpowers/plans/2026-06-21-algo-cards-airpods-plan.md`。

**要点回顾**：
- 2×2 Bento 网格 → 横向滚动卡片轮播
- 每次突出一个算法
- 卡片切换 scale/opacity/blur 过渡
- SVG 流程图增大到 560×340 并细化关键步骤
- 底部指示点 + `01 / 04` 计数器
- 中强度动画

---

### 4.2 Section 1：单模型推理页

**当前问题**：
- 三列流水线布局信息密度高
- 结果图被挤压在右侧
- 控件（下拉框、按钮、步骤数字）过于显眼

**改造方向**：**Workbench 工作台**

#### 布局重构

```
┌─────────────────────────────────────────────────────────────┐
│  模型推理                                                   │
│  一次推理，定位缺陷。                                       │
│                                                             │
│  [算法 ▼]  [预训练/自训练 ▼]  [数据集 ▼]  [测试图片 ▼]  [开始推理] │
│                                                             │
│                                                             │
│                    [  大 型 结 果 图  ]                     │
│                    [  原图 / 热力图对比  ]                   │
│                                                             │
│  异常得分  0.8234    置信度  92.3%    阈值 τ  0.450        │
│  ─────────────────────────────────────────────────          │
│                    ● 异常                                    │
└─────────────────────────────────────────────────────────────┘
```

#### 详细结构

顶部工具栏（`inference-toolbar`）：
- 算法选择（轻量 pill 下拉）
- 数据来源切换（预训练 / 自训练）
- 数据集选择
- 测试图片选择（带缩略图）
- 主按钮「开始推理」

中央结果区（`inference-stage`）：
- 大型结果图容器，最大宽度 900px
- 默认状态显示提示文案 + 占位图
- 推理完成后显示原图/热力图对比滑块
- 滑块手柄更细更 Apple（2px 线 + 小圆点）

底部指标栏（`inference-metrics`）：
- 三列大字指标：异常得分、置信度、阈值
- 每个指标只有数字 + 小标签，去除进度条
- 判定 badge：「正常」或「异常」大色块

#### CSS 要点

```css
.inference-page {
    background: var(--bg-root);
}

.inference-toolbar {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 12px;
    justify-content: center;
    padding: 16px 24px;
    background: var(--bg-system);
    border: 1px solid var(--sep-subtle);
    border-radius: var(--r-xl);
    margin: 0 auto 32px;
    max-width: 980px;
}

.inference-stage {
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 420px;
    margin: 0 auto;
    max-width: 920px;
}

.inference-result-image {
    width: 100%;
    border-radius: var(--r-lg);
    overflow: hidden;
    background: var(--bg-system);
    box-shadow: var(--shadow-lg);
}

.inference-metrics {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 24px;
    max-width: 720px;
    margin: 32px auto 0;
    text-align: center;
}

.inference-metric-value {
    font-size: 40px;
    font-weight: 600;
    letter-spacing: -0.02em;
    line-height: 1.1;
}

.inference-metric-label {
    font-size: 13px;
    color: var(--text-tertiary);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 6px;
}

.inference-verdict {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    margin-top: 24px;
    padding: 10px 20px;
    border-radius: 100px;
    font-size: 15px;
    font-weight: 600;
}
.inference-verdict--normal {
    background: var(--ok-bg);
    color: var(--ok);
}
.inference-verdict--anomaly {
    background: var(--bad-bg);
    color: var(--bad);
}
```

#### 动画

- 结果图出现：`scale(0.96) + opacity(0)` → `scale(1) + opacity(1)`，800ms
- 指标数字：`Anim.numberRoll` 弹性滚动
- 判定 badge：`springScale` 弹入
- 工具栏下拉：`fadeIn` 200ms

---

### 4.3 Section 2：Training Studio

**当前问题**：
- 布局与全站风格一致，但训练过程是长时交互，需要更强的视觉焦点
- 训练曲线和监控面板可以更突出
- 训练状态反馈可以更强

**改造方向**：**训练控制中心（Training Control Center）**

**注意**：取消独立的深色沉浸章节，Training Studio 跟随全局主题。通过以下方式让它在统一主题下仍然突出：
- 更大的监控曲线区域
- 更强的强调色光晕
- 训练状态脉冲动画
- 训练完成时的仪式感动画

#### 布局重构

```
┌─────────────────────────────────────────────────────────────┐
│  TRAINING STUDIO                                            │
│  训练你的检测模型                                            │
│                                                             │
│  ┌──────────────┐  ┌──────────────────────────────────┐    │
│  │  算法选择     │  │                                  │    │
│  │  ● PatchCore │  │      [ Loss 曲线实时绘制 ]        │    │
│  │  ○ PaDiM     │  │                                  │    │
│  │  ○ FRE       │  ├──────────────────────────────────┤    │
│  │  ○ DRAEM     │  │  Epoch  Loss  LR  val AUROC      │    │
│  │              │  │  12/30   0.12  1e-4  0.89        │    │
│  │  Epoch       │  │                                  │    │
│  │  [    ]      │  │  [ 脉冲状态：训练中... ]          │    │
│  │              │  └──────────────────────────────────┘    │
│  │  Batch       │                                            │
│  │  [    ]      │  ┌──────────────────────────────────┐    │
│  │              │  │  样本画廊（可排除）                │    │
│  │  [开始训练]   │  │  ◻ ◻ ◻ ◻ ◻ ◻ ◻ ◻ ◻ ◻            │    │
│  └──────────────┘  └──────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

#### 跟随全局主题的设计

亮色模式下：
- 面板背景：纯白 `#ffffff` + 细边框
- Loss 曲线：蓝色 `#0071e3` + 轻微阴影
- 状态脉冲：蓝色

暗色模式下：
- 面板背景：半透明玻璃
- Loss 曲线：蓝色 `#2997ff` + 明显发光
- 状态脉冲：蓝色发光

```css
.training-monitor-card {
    background: var(--bg-system);
    border: 1px solid var(--sep-subtle);
    border-radius: var(--r-xl);
    box-shadow: var(--shadow-md);
}

/* 亮色下增强曲线可见性 */
html[data-theme="light"] .training-chart-line {
    stroke: var(--accent);
    stroke-width: 2.5;
    filter: drop-shadow(0 2px 4px rgba(0, 113, 227, 0.15));
}

/* 暗色下增强发光 */
html[data-theme="dark"] .training-chart-line {
    stroke: var(--accent);
    stroke-width: 2.5;
    filter: drop-shadow(0 0 10px var(--accent-glow));
}
```

#### 训练状态视觉

```css
.training-status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--accent);
    display: inline-block;
    margin-right: 8px;
}

.training-status-dot--training {
    animation: statusPulse 2s ease-in-out infinite;
}

@keyframes statusPulse {
    0%, 100% { opacity: 0.4; transform: scale(1); }
    50% { opacity: 1; transform: scale(1.3); }
}
```

#### Loss 曲线区域

```css
.training-chart-area {
    background: var(--bg-secondary);
    border-radius: var(--r-lg);
    padding: 20px;
}
```

---

### 4.4 Section 3：四模型对比页

**当前问题**：
- 四列网格信息密集
- 每个槽位控件过多
- 缺少「画廊」式的轻盈感

**改造方向**：**Gallery 画廊式对比**

#### 布局重构

```
┌─────────────────────────────────────────────────────────────┐
│  对比分析                                                   │
│  四款算法，同台竞技。                                       │
│                                                             │
│              [ 共享输入图像 ]                               │
│                                                             │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐      │
│  │ PatchCore│ │  PaDiM   │ │   FRE    │ │  DRAEM   │      │
│  │  0.9448  │ │  0.9231  │ │  0.8912  │ │  0.8567  │      │
│  │[热力图]  │ │[热力图]  │ │[热力图]  │ │[热力图]  │      │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘      │
│                                                             │
│              [ 开始对比 / 重新对比 ]                         │
└─────────────────────────────────────────────────────────────┘
```

#### CSS 要点

```css
.compare-page {
    background: var(--bg-root);
}

.compare-shared-image {
    max-width: 400px;
    margin: 0 auto 32px;
    text-align: center;
}

.compare-shared-image img {
    width: 100%;
    border-radius: var(--r-lg);
    box-shadow: var(--shadow-md);
}

.compare-gallery {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 20px;
    max-width: 1200px;
    margin: 0 auto;
}

.compare-slot {
    background: var(--bg-system);
    border: 1px solid var(--sep-subtle);
    border-radius: var(--r-lg);
    padding: 20px;
    transition: transform 0.35s var(--ease-spring),
                box-shadow 0.35s var(--ease-out);
}

.compare-slot:hover {
    transform: translateY(-4px) scale(1.01);
    box-shadow: var(--shadow-lg);
}

.compare-slot-name {
    font-size: 18px;
    font-weight: 600;
    margin-bottom: 4px;
}

.compare-slot-score {
    font-size: 28px;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: var(--accent);
    margin-bottom: 12px;
}

.compare-slot-heatmap {
    width: 100%;
    border-radius: var(--r-md);
    overflow: hidden;
    background: var(--bg-secondary);
}

.compare-slot-heatmap img {
    width: 100%;
    height: auto;
    object-fit: contain;
}

/* 不同模型色标 */
.compare-slot--patchcore { border-top: 3px solid #2997ff; }
.compare-slot--padim     { border-top: 3px solid #30d158; }
.compare-slot--fre       { border-top: 3px solid #ff9f0a; }
.compare-slot--draem     { border-top: 3px solid #bf5af2; }
```

#### 动画

- 四张卡片依次 stagger 入场：每张延迟 100ms
- 得分数字滚动动画
- hover 时卡片上浮
- 共享原图先出现，结果图后 stagger 出现

#### 响应式

- 平板：2×2 网格
- 手机：1 列堆叠

---

## 5. 训练动画详细规格

### 5.1 Loss 曲线实时绘制

当前 `training.js` 中的 canvas 绘制是逐点连线。改为带发光尾迹的动画绘制：

```javascript
// 新增长度限制，只显示最近 N 个点
const MAX_POINTS = 60;

function drawChart() {
    const points = metricsHistory.slice(-MAX_POINTS);
    ctx.clearRect(0, 0, width, height);

    drawGrid();

    const gradient = ctx.createLinearGradient(0, 0, 0, height);
    gradient.addColorStop(0, 'rgba(0, 113, 227, 0.20)');
    gradient.addColorStop(1, 'rgba(0, 113, 227, 0)');

    ctx.beginPath();
    ctx.moveTo(0, height);
    points.forEach((p, i) => {
        const x = (i / (MAX_POINTS - 1)) * width;
        const y = height - (p.loss / maxLoss) * height;
        ctx.lineTo(x, y);
    });
    ctx.lineTo(width, height);
    ctx.closePath();
    ctx.fillStyle = gradient;
    ctx.fill();

    ctx.beginPath();
    points.forEach((p, i) => {
        const x = (i / (MAX_POINTS - 1)) * width;
        const y = height - (p.loss / maxLoss) * height;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = 'var(--accent)';
    ctx.lineWidth = 2.5;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.shadowColor = 'var(--accent-glow)';
    ctx.shadowBlur = 12;
    ctx.stroke();
    ctx.shadowBlur = 0;

    const last = points[points.length - 1];
    const lastY = height - (last.loss / maxLoss) * height;
    ctx.beginPath();
    ctx.arc(width, lastY, 4, 0, Math.PI * 2);
    ctx.fillStyle = '#ffffff';
    ctx.fill();
    ctx.strokeStyle = 'var(--accent)';
    ctx.lineWidth = 2;
    ctx.stroke();
}
```

### 5.2 Epoch 数字滚动

每次 epoch 更新时：

```javascript
Anim.numberRoll(epochEl, prevEpoch, currentEpoch, 500, {
    format: v => Math.round(v).toString()
});
```

### 5.3 状态脉冲

训练中状态 badge 添加脉冲圆点：

```html
<span class="training-status-badge is-training">
    <span class="training-status-dot training-status-dot--training"></span>
    训练中...
</span>
```

### 5.4 训练完成动画

训练完成时：
1. 状态 badge 从「训练中」变为「完成」，颜色从蓝变绿
2. 最终指标面板从 `translateY(20px) + opacity(0)` 弹入
3. 「开始训练」按钮变为「重新训练」
4. 触发全局模型列表刷新（已有逻辑）

### 5.5 样本画廊排除动画

点击样本排除/恢复时：
- 缩略图 `scale(1.05)` 反馈 150ms
- 排除状态添加灰度 + 半透明过渡 250ms
- 已排除样本计数更新时数字滚动

### 5.6 Reduced Motion

所有训练动画在 `prefers-reduced-motion` 下：
- 曲线直接显示最终状态，不逐帧绘制
- 数字直接显示，不滚动
- 脉冲圆点变为静态实心点

---

## 6. 共享组件库

为保持一致性，抽取以下共享组件：

### 6.1 `HighlightCard`
用于首页算法卡片，可复用于未来其他 highlight 模块。

### 6.2 `MetricBlock`
```
大数字
小标签
```
用于单模型推理指标、四模型对比得分。

### 6.3 `PillSelect`
药丸形状选择器，替代当前较重的自定义下拉框。

### 6.4 `StatusBadge`
带脉冲点的状态标签，用于训练状态。

### 6.5 `GlowLineChart`
发光折线图组件，用于 Training Studio loss 曲线。

---

## 7. 实施批次

### 批次 1：全局基础（约 20% 工作量）
1. 主题系统改为默认跟随系统 + 手动切换
2. 色彩系统精修
3. 字体比例系统
4. 按钮/输入框组件统一
5. 光标光晕和噪点纹理克制化融合
6. 更新测试

### 批次 2：首页算法卡片（约 25% 工作量）
7. HTML 结构改造
8. 轮播 CSS
9. SVG 流程图优化
10. 卡片切换动画
11. 测试更新

### 批次 3：单模型推理页（约 20% 工作量）
12. 工作台布局
13. 工具栏简化
14. 结果图放大
15. 指标块组件
16. 结果出现动画

### 批次 4：Training Studio（约 20% 工作量）
17. 跟随全局主题的训练控制中心
18. Loss 曲线发光 + 实时绘制
19. 状态脉冲动画
20. Epoch 数字滚动
21. 训练完成动画

### 批次 5：四模型对比页（约 12% 工作量）
22. 画廊式网格
23. 共享原图区域
24. 卡片 stagger 入场
25. 得分数字滚动

### 批次 6：Polish（约 3% 工作量）
26. 响应式统一检查
27. Reduced motion 验证
28. 视觉走查与微调

---

## 8. 文件变更清单

| 文件 | 操作 | 批次 | 说明 |
|------|------|------|------|
| `modules/ui/static/css/app.css` | 大量修改 | 1 | 色彩、字体、按钮、输入框、全局背景 |
| `modules/ui/static/css/apple-redesign.css` | 大量修改 | 2-5 | 页面级覆盖样式 |
| `modules/ui/static/theme.js` | 修改 | 1 | 系统主题跟随 |
| `modules/ui/static/js/cursor-glow.js` | 修改 | 1 | 缩小光晕、平滑跟随、主题感知 |
| `modules/ui/static/js/animations.js` | 扩展 | 1,4 | 新增/优化动画工具 |
| `modules/ui/static/js/algo-carousel.js` | 新建 | 2 | 算法轮播组件 |
| `modules/ui/static/js/inference.js` | 修改 | 3 | 推理工作台交互 |
| `modules/ui/static/js/training.js` | 修改 | 4 | 训练动画 |
| `modules/ui/static/js/compare.js` | 修改 | 5 | 四模型对比动画 |
| `modules/ui/static/index.html` | 大量修改 | 2-5 | 各 section 结构 |
| `modules/ui/static/js/app.js` | 修改 | 1 | 导航、滚动逻辑适配 |
| `tests/test_ui_static.py` | 修改 | 各批次 | 结构断言更新 |

---

## 9. 测试计划

### 9.1 静态测试
- 更新 `test_index_html_has_new_structure`
- 检查新增资源文件存在
- 检查关键类名存在

### 9.2 功能测试
- 主题跟随系统 / 手动切换
- 算法轮播滚动、指示点、键盘导航
- 单模型推理完整流程
- Training Studio 训练 SSE 流
- 四模型对比 SSE 流

### 9.3 视觉测试
- 各页面在桌面/平板/手机下的布局
- light/dark 模式下的对比度
- 动画是否正常触发
- reduced-motion 下是否降级

### 9.4 回归测试
- `python -m pytest tests/ -v`
- 启动 UI 进行手动验证

---

## 10. 风险与注意事项

1. **默认 light 主题** 可能需要调整 favicon/logo 在 light 下的可见性
2. **光标光晕缩小后** 需确保 still 有存在感但不干扰
3. **Training Studio 跟随全局主题** 后，需要通过曲线发光和状态脉冲来保持视觉焦点
4. **大量 HTML/CSS 改动** 需要同步更新 `test_ui_static.py`
5. **scroll-snap 与横向轮播** 需要测试滚动冲突
6. **SVG 流程图增大** 可能影响首屏加载，建议懒加载或内联优化

---

## 11. 待用户最终确认

确认以下事项后即可开始分批实施：
1. ✅ 全站统一主题，默认**跟随系统**亮暗模式，可手动切换
2. ✅ 保留当前暗色切换能力
3. ✅ Training Studio **不单独做深色沉浸**，跟随全局主题，通过动画和光晕突出
4. ✅ 单模型推理改为**工作台布局**（大图 + 顶部工具栏）
5. ✅ 四模型对比改为**画廊式网格**
6. ✅ **中强度动画**（800ms 入场 + 数字滚动 + 脉冲 + SVG 绘制）
7. ✅ **保留**光标光晕和噪点纹理，但克制化、主题感知化
8. ✅ 按上述 **6 个批次** 顺序推进

