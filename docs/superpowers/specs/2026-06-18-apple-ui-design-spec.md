# Apple 风格 UI 设计规范 — 工业缺陷检测系统

> 版本 1.0 · 2026-06-18 · 亮/暗双模式 · 写给开发者的实用 checklist

---

## 1. 设计原则

1. **克制优于丰富。** 单强调色 `#2997ff`，其余全部黑/灰/白透明度。不确定加的元素，不加。
2. **留白优于填满。** 内边距 28px 起步，段间距 40–56px，容器 1200px 居中。内容漂浮，不挤画布。
3. **文字即界面。** 不用装饰图、图标堆砌、分割线。56px 大标题 + 字号层级 + SF 字体栈完成视觉表达。
4. **动效有功能。** 方向性（下方淡入）、渐进式（80ms 间隔）、物理性（弹簧缓动）。不飘，不炫。
5. **深色不是浅色反色。** 暗色文字不用纯白 `#fff`，亮色文字不用纯黑 `#000`。阴影分层模拟物理边缘。

---

## 2. CSS 变量速查

### 2.1 暗色模式（默认）

| 类别 | 变量 | 值 | 用途 |
|------|------|----|------|
| 基底 | `--bg-root` | `#000000` | 页面根背景 |
| 基底 | `--bg-system` | `#1c1c1e` | 卡片/面板 |
| 基底 | `--bg-secondary` | `#2c2c2e` | 选中态/悬浮面板 |
| 基底 | `--bg-tertiary` | `#3a3a3c` | 输入框底 |
| 分隔 | `--sep-subtle` | `rgba(255,255,255,0.06)` | 卡片边线 |
| 分隔 | `--sep-default` | `rgba(255,255,255,0.10)` | 输入框边线 |
| 分隔 | `--sep-strong` | `rgba(255,255,255,0.16)` | 需突出的边线 |
| 文字 | `--text` | `rgba(255,255,255,0.92)` | 正文/标题 |
| 文字 | `--text-secondary` | `rgba(255,255,255,0.60)` | 辅助说明 |
| 文字 | `--text-tertiary` | `rgba(255,255,255,0.36)` | 占位/禁用 |
| 强调 | `--accent` | `#2997ff` | 按钮/链接 |
| 强调 | `--accent-hover` | `#47a9ff` | 按钮 hover |
| 强调 | `--accent-pressed` | `#0070d6` | 按钮按下 |
| 强调 | `--accent-dim` | `rgba(41,151,255,0.12)` | 强调浅底 |
| 强调 | `--accent-glow` | `rgba(41,151,255,0.25)` | 按钮光晕 |
| 状态 | `--ok` | `#30d158` | 正常 |
| 状态 | `--ok-bg` | `rgba(48,209,88,0.10)` | 正常浅底 |
| 状态 | `--bad` | `#ff453a` | 异常 |
| 状态 | `--bad-bg` | `rgba(255,69,58,0.10)` | 异常浅底 |
| 状态 | `--warn` | `#ff9f0a` | 警告 |
| 状态 | `--warn-bg` | `rgba(255,159,10,0.08)` | 警告浅底 |
| 圆角 | `--r-sm` | `8px` | 输入框/小按钮 |
| 圆角 | `--r-md` | `12px` | 面板/图例 |
| 圆角 | `--r-lg` | `16px` | 主卡片 |
| 阴影 | `--shadow-sm` | `0.5px 亮边 + 1px 4px 近影` | 小卡片/标签 |
| 阴影 | `--shadow-md` | `0.5px 亮边 + 2px 8px + 8px 32px` | 卡片默认 |
| 阴影 | `--shadow-lg` | `0.5px 亮边 + 4px 16px + 16px 48px` | 下拉/悬浮 |
| 阴影 | `--shadow-glow` | `--shadow-md + 32px 蓝色光晕` | 按钮 hover |
| 动效 | `--ease-out` | `cubic-bezier(0,0,0.2,1)` | 通用淡入 |
| 动效 | `--ease-spring` | `cubic-bezier(0.22,0.8,0.3,1.15)` | 按钮/悬停 |
| 动效 | `--ease-out-expo` | `cubic-bezier(0.16,1,0.3,1)` | 进度条/大动画 |
| 动效 | `--dur-fast` | `180ms` | 颜色/边框过渡 |
| 动效 | `--dur-normal` | `300ms` | 悬停过渡 |
| 动效 | `--dur-slow` | `500ms` | 入场动画 |
| 动效 | `--dur-glacial` | `800ms` | 首屏渐进 |

### 2.2 阴影定义

```css
/* 暗色 — 亮边模拟物理边缘反射 */
--shadow-sm:
    0 0 0 0.5px rgba(255, 255, 255, 0.04),
    0 1px 4px rgba(0, 0, 0, 0.2);
--shadow-md:
    0 0 0 0.5px rgba(255, 255, 255, 0.06),
    0 2px 8px rgba(0, 0, 0, 0.3),
    0 8px 32px rgba(0, 0, 0, 0.4);
--shadow-lg:
    0 0 0 0.5px rgba(255, 255, 255, 0.08),
    0 4px 16px rgba(0, 0, 0, 0.35),
    0 16px 48px rgba(0, 0, 0, 0.5);
--shadow-glow:
    0 0 0 0.5px rgba(255, 255, 255, 0.06),
    0 2px 8px rgba(0, 0, 0, 0.3),
    0 0 32px rgba(41, 151, 255, 0.25);

/* 亮色 — 暗边更轻，阴影更散 */
--shadow-sm:
    0 0 0 0.5px rgba(0, 0, 0, 0.04),
    0 1px 4px rgba(0, 0, 0, 0.06);
--shadow-md:
    0 0 0 0.5px rgba(0, 0, 0, 0.04),
    0 1px 4px rgba(0, 0, 0, 0.06),
    0 8px 24px rgba(0, 0, 0, 0.08);
--shadow-lg:
    0 0 0 0.5px rgba(0, 0, 0, 0.06),
    0 4px 12px rgba(0, 0, 0, 0.08),
    0 16px 40px rgba(0, 0, 0, 0.12);
--shadow-glow:
    0 0 0 0.5px rgba(0, 0, 0, 0.04),
    0 2px 8px rgba(0, 0, 0, 0.08),
    0 0 32px rgba(41, 151, 255, 0.20);
```

### 2.3 亮色模式覆盖

```css
@media (prefers-color-scheme: light) {
    :root {
        --bg-root: #f0f0f0;          /* ← #000000 */
        --bg-system: #ffffff;         /* ← #1c1c1e */
        --bg-secondary: #f5f5f7;      /* ← #2c2c2e */
        --bg-tertiary: #e8e8ed;       /* ← #3a3a3c */
        --sep-subtle: rgba(0,0,0,0.06);
        --sep-default: rgba(0,0,0,0.10);
        --sep-strong: rgba(0,0,0,0.16);
        --text: rgba(0,0,0,0.88);
        --text-secondary: rgba(0,0,0,0.55);
        --text-tertiary: rgba(0,0,0,0.30);
        --shadow-card:
            0 0 0 0.5px rgba(0,0,0,0.04),
            0 1px 4px rgba(0,0,0,0.06),
            0 8px 24px rgba(0,0,0,0.08);
        --shadow-elevated:
            0 0 0 0.5px rgba(0,0,0,0.06),
            0 4px 12px rgba(0,0,0,0.08),
            0 16px 40px rgba(0,0,0,0.12);
    }
}
```

**不变量**（两模式共用）：`--accent` / `--ok` / `--bad` / `--warn` / 圆角 / 字体 / 动效。

---

## 3. 组件规范

### 3.1 标题区 · Hero

```html
<div class="reveal reveal-1" style="padding: 48px 0 8px 0;">
    <div class="title">缺陷检测</div>
    <div class="subtitle">无监督异常检测系统 · Anomalib 2.3</div>
</div>
```

| 属性 | title | subtitle |
|------|-------|----------|
| 字号 | `56px` | `19px` |
| 字重 | `700` | `400` |
| 字距 | `-0.025em` | `-0.01em` |
| 颜色 | `var(--text)` | `var(--text-secondary)` |
| 行高 | `1.05` | 默认 |
| 底部距 | `0` | `56px` |

⚠️ 标题区顶部 padding `48px` 给足呼吸空间。文字左对齐，不居中。

### 3.2 标签页 · Tabs

容器：`background: var(--bg-system); border-radius: var(--r-md); padding: 3px;`

Tab 按钮：
```css
button[role="tab"] {
    font: var(--font-body) 13px / 500;
    color: var(--text-secondary);
    border-radius: 10px;
    padding: 7px 20px;
    transition: all var(--dur-fast) var(--ease-out);
}
button[role="tab"]:hover {
    color: var(--text);
    background: rgba(128,128,128,0.08);
}
button[role="tab"][aria-selected="true"] {
    color: var(--text);
    background: var(--bg-secondary);
    box-shadow: 0 0 0 0.5px rgba(128,128,128,0.15), 0 1px 3px rgba(0,0,0,0.2);
    animation: tabPop 300ms var(--ease-spring);
}
```

⚠️ 选中态有 `tabPop` 弹簧动画（`scale(0.94) → scale(1)`），但仅用时 `300ms`，不拖沓。

### 3.3 算法卡片 · AlgoCard

```html
<div class="algo-card">
    <h4 class="recommended">PatchCore <span>— 特征建模</span></h4>
    <p>CNN 提取局部特征 → 记忆库存储 → 最近邻判别。零训练、推理最快。</p>
</div>
```

| 属性 | h4 | span (副标题) | p |
|------|----|---------------|----|
| 字号 | `18px` | `13px` | `14px` |
| 字重 | `600` | `400` | `400` |
| 颜色 | `var(--text)` | `var(--text-tertiary)` | `var(--text-secondary)` |
| 底部距 | `6px` | — | `0` |

`.recommended` 颜色改为 `var(--accent)`，后跟 `::after` 生成「推荐」圆角标签。

⚠️ 卡片无背景、无边框、无装饰线。信息靠字号和颜色层级表达，不靠"框"。

### 3.4 输入控件

```css
.gradio-dropdown, select, input, textarea {
    background: var(--bg-system);
    border: 1px solid var(--sep-default);
    border-radius: var(--r-sm);
    color: var(--text);
    font-family: var(--font-body);
    font-size: 14px;
    transition: border-color var(--dur-fast) var(--ease-out),
                box-shadow var(--dur-fast) var(--ease-out);
}
:focus {
    border-color: var(--accent);
    box-shadow: 0 0 0 4px var(--accent-dim);
}
```

| 属性 | 值 |
|------|-----|
| 字号 | `14px` |
| 内边距 | `9px 14px` |
| 边线 | `1px solid var(--sep-default)` |
| 圆角 | `8px` |
| 聚焦环 | `4px var(--accent-dim)` |

⚠️ 聚焦环用 `4px`（不是 2px），是为了视觉上明显但不过分。图片上传区 hover 时虚线变实线 + 聚焦环 + 微蓝底色。

### 3.5 按钮

```css
button.primary {
    font: var(--font-body) 15px / 500;
    background: var(--accent);
    color: #ffffff;
    border: none;
    border-radius: 100px;
    padding: 12px 30px;
    transition: background var(--dur-fast) var(--ease-out),
                transform 350ms var(--ease-spring),
                box-shadow var(--dur-fast) var(--ease-out);
}
```

| 状态 | 背景 | 变换 |
|------|------|------|
| 默认 | `var(--accent)` #2997ff | — |
| Hover | `var(--accent-hover)` #47a9ff | `scale(1.03)` + `box-shadow: 0 0 20px var(--accent-glow)` |
| Active | `var(--accent-pressed)` #0070d6 | `scale(0.95)` + 过渡压缩到 100ms |
| Focus | — | `2px solid var(--accent)` 环, offset 2px |

⚠️ **必须 100px 圆角**（胶囊形），不是 8px。Hover 必须同时有 scale 和 glow。Active 缩到 0.95（物理按压感），过渡加速。

对比按钮 (`.compare-btn`)：背景透明 → hover 变 `var(--accent-dim)`，其他同上。

### 3.6 状态面板 · StatusPanel

```css
.status-panel {
    background: var(--bg-system);
    border: 1px solid var(--sep-subtle);
    border-radius: var(--r-md);
    padding: 16px;
    min-height: 48px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    transition: background var(--dur-normal) var(--ease-out),
                border-color var(--dur-normal) var(--ease-out);
}
```

加载旋转：`18px` 圆环，边框 `2px solid var(--sep-default)`，顶边 `var(--accent)`。动画 `spinnerRotate 0.55s linear infinite`。

成功/失败/警告变体在各底色（`--ok-bg`/`--bad-bg`/`--warn-bg`）上显示对应色文字。

### 3.7 结果卡片 · ResultCard

```css
.result-card {
    background: var(--bg-system);
    backdrop-filter: blur(20px) saturate(120%);
    border-radius: var(--r-lg);        /* 16px */
    padding: 28px;
    border: 1px solid var(--sep-subtle);
    box-shadow: var(--shadow-md);
    transition: transform 0.4s var(--ease-spring),
                box-shadow 0.4s var(--ease-out),
                border-color var(--dur-normal) var(--ease-out);
}
.result-card:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-lg);
    border-color: var(--sep-default);
}
```

| 属性 | 值 |
|------|-----|
| 圆角 | `16px` |
| 内边距 | `28px` |
| 背景 | `var(--bg-system)` + 磨砂玻璃 |
| Hover | 上浮 2px + 阴影加深 |

空状态占位：内边距加大到 `48px 28px`，居中显示 `var(--text-tertiary)` 色提示。

### 3.8 状态徽章 · StatusBadge

```css
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 14px;
    border-radius: 100px;
    font: var(--font-body) 13px / 500;
    transition: transform var(--dur-fast) var(--ease-spring);
}
.status-badge.normal { background: var(--ok-bg); color: var(--ok); }
.status-badge.anomaly {
    background: var(--bad-bg);
    color: var(--bad);
    animation: pulseWarn 2.5s ease-in-out infinite;
}
```

⚠️ 异常徽章有呼吸式透明度脉冲（`1 → 0.65 → 1`），持续提醒但不刺眼。

### 3.9 度量数字 · CoreMetric

```
┌─────────────────────────────┐
│    异常得分                   │  ← 12px var(--text-tertiary)
│    0.9876                    │  ← 42px mono, var(--bad) 或 var(--ok)
│    ████████████────          │  ← 进度条 4px 高
│            τ 0.823           │  ← 阈值标记线
└─────────────────────────────┘
```

| 属性 | label | value | 进度条 |
|------|-------|-------|--------|
| 字号 | `12px` | `42px` | `4px` 高 |
| 颜色 | 三级文字 | 异常红/正常绿 | 对应状态色 |
| 动效 | — | — | width 0.7s `--ease-out-expo` |

进度条末端有 7px 白色圆点（`::after`）。阈值标记线带圆点指示器。

Hover：背景微升 + `translateY(-1px)`。

### 3.10 热力图图例 · HeatmapLegend

```
┌──────────────┐
│     得分      │  ← 11px, 500 weight, var(--text-tertiary)
│     ██       │
│     ██  1.0  │
│     ██  0.8  │
│     ██  0.6  │  ← 200px 渐变条, 8px 宽
│     ██  0.4  │
│     ██  0.2  │
│     ██  0.0  │
└──────────────┘
```

| 属性 | 值 |
|------|-----|
| 渐变 | `#ff3b30 → #ff9f0a → #ffd60a → #30d158 → #0a84ff` |
| 条宽 | `8px` |
| 条高 | `200px` |
| 圆角 | `4px` |
| 标签字号 | `10px mono` |
| 容器圆角 | `12px` + 1px 边线 |

### 3.11 对比模式 · Compare

对比卡片 `.compare-result-card` 与结果卡片同材质：`var(--bg-system)` + 磨砂玻璃 + 16px 内边距 + `var(--shadow-md)`。

Hover：`translateY(-1px)` + 阴影加深。空状态居中显示 `var(--text-tertiary)`。

对比按钮 `.compare-btn` 是 outline 风格：透明底 `var(--bg-system)` + `1px solid var(--sep-default)` 边线 + `var(--accent)` 色文字。Hover 变 `var(--accent-dim)` 底 + `scale(1.02)` + 蓝色光晕。

### 3.12 底部说明 · Footer

```css
.footer-section {
    background: var(--bg-system);
    backdrop-filter: blur(20px) saturate(120%);
    border: 1px solid var(--sep-subtle);
    border-radius: var(--r-lg);
    padding: 28px 32px;
    margin-top: 48px;
    transition: border-color var(--dur-normal) var(--ease-out),
                box-shadow var(--dur-normal) var(--ease-out);
}
.footer-section:hover {
    border-color: var(--sep-default);
    box-shadow: var(--shadow-md);
}
```

| 属性 | 值 |
|------|-----|
| 标题字号 | `12px` / 500 / `var(--text-tertiary)` / letter-spacing `0.04em` |
| 内容标题 | `13px` / 600 / `var(--text)` |
| 内容正文 | `13px` / `var(--text-secondary)` / line-height `1.7` |
| 列间距 | `56px` |

---

## 4. 动效参数表

| 场景 | 属性 | 时长 | 缓动 | 效果 |
|------|------|------|------|------|
| 页面加载入场 | `animation: revealUp` | 800ms | `ease-out-expo` | `opacity 0→1` + `translateY(28→0)` + `blur(2→0)` |
| 入场延迟级联 | `animation-delay` | 0/80/160/240/320/400ms | — | 6 层依次出现 |
| 按钮 hover | `transform` | 350ms | `ease-spring` | `scale(1.03)` + glow |
| 按钮 press | `transform` | 100ms | `ease-out` | `scale(0.95)` |
| 卡片 hover | `transform` | 400ms | `ease-spring` | `translateY(-2px)` + shadow↑ |
| 标签页选中 | `animation: tabPop` | 300ms | `ease-spring` | `scale(0.94→1)` |
| 下拉菜单弹出 | `animation: dropdownReveal` | 200ms | `ease-out` | `opacity 0→1` + `translateY(-4→0)` + `scale(0.98→1)` |
| 进度条伸缩 | `transition: width` | 700ms | `ease-out-expo` | 平滑伸缩 |
| 异常徽章呼吸 | `animation: pulseWarn` | 2.5s | `ease-in-out` | `opacity 1 ↔ 0.65` |
| 滚动露出 | `animation: scrollReveal` | 线性 | `view()` | `opacity 0→1` + `translate 36→0` |

### CSS 缓动函数定义

```css
--ease-out:      cubic-bezier(0, 0, 0.2, 1);        /* 通用淡出 */
--ease-spring:   cubic-bezier(0.22, 0.8, 0.3, 1.15); /* 弹性反弹 */
--ease-out-expo: cubic-bezier(0.16, 1, 0.3, 1);      /* 大范围移动 */
--ease-in-out:   cubic-bezier(0.4, 0, 0.2, 1);       /* 呼吸循环 */
```

### 时长层级

| 层级 | 值 | 用途 |
|------|-----|------|
| 即时 | `100ms` | 按钮按下 |
| 快速 | `180ms` | 颜色/边框/背景过渡 |
| 正常 | `280-300ms` | 悬停反馈 |
| 缓慢 | `500ms` | 入场动画 |
| 极慢 | `800ms` | 首屏渐进加载 |

---

## 5. 反模式（不该做的事）

- ❌ 不要加网格背景、扫描线、粒子效果（破坏纯粹感）
- ❌ 不要用 Google Fonts — 字体必须本地栈，避免被墙
- ❌ 不要加 JS MutationObserver 改样式（用 CSS 解决 CSS 问题）
- ❌ 不要用 `!important` 除非覆盖 Gradio 内部样式
- ❌ 不要用 `#fff` 纯白文字（用 `rgba(255,255,255,0.92)`）
- ❌ 不要用 `#000` 纯黑文字亮色模式（用 `rgba(0,0,0,0.88)`）
- ❌ 不要加 `box-shadow` 给普通按钮（破坏扁平感，只在 hover 出现）
- ❌ 不要加边框给卡片（已经用背景色差 + 阴影区分层次）
- ❌ 不要用 `scale()` 以外的方式实现按钮反馈（不用 `translate`、不用 `border` 变色）
- ❌ 不要用奇数圆角值（3/5/7/9px），用 8/12/16/20px
- ❌ 不要在动效中改变 `width`/`height`（用 `transform`，避免 layout thrashing）
- ❌ 不要忽略 `prefers-reduced-motion`（未来应添加，当前暂未实现）

---

## 6. 附录：Apple 设计参考

**暗色模式**：iOS Settings → Display & Brightness → Dark. Background = `#000000`, Card = `#1c1c1e` (systemBackground), Secondary = `#2c2c2e` (secondarySystemBackground).

**字体比率**：Apple 大字标题在 48–64px 范围，正文 15–17px，辅助文 11–13px。字距大标题 -0.02em 到 -0.03em，正文 -0.01em 或 0。

**按钮动画**：iOS 按钮按下时 `scale(0.97)` 持续 0.1s，松手弹簧回弹 0.35s。

**磨砂玻璃**：macOS 使用 `backdrop-filter: blur(30px) saturate(180%)`，iOS 用 `blur(20px) saturate(120%)`。项目用后者。

**阴影哲学**：Apple 不用单一大阴影。总是三层：0.5px 亮边（模拟物理边缘聚光）+ 近影 + 远影。
