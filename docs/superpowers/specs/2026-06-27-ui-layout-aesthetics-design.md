# UI 排印 + 布局对齐 AirPods Pro — 设计规范

> 日期：2026-06-27
> 基线：实测 Apple AirPods Pro 3 官网计算样式 vs 本地 UI（`127.0.0.1:8000`）计算样式与几何。
> 关联：`2026-06-21-airpods-pro-ui-direction.md`（方向）、`2026-06-21-full-ui-airpods-plan.md`（综合规划）。

## 范围

**纳入**：排印数值（A1–A6）、装饰精简（B7–B8）、全局布局对齐（G1–G3）、Hero 重排（H1–H2）、page-3 推理与结果展示分栏（I1–I5）、page-4 四模型对比（C1–C2）。

**排除**（用户指定）：
- **颜色系统**——默认主题、`--accent`、亮/暗色板、章节背景明暗交替均不动。
- **字体自托管**——SF Pro 子集加载单独一轮再评估，本规范不改 `font-family` 栈。

## 设计基线（实测值）

| 维度 | Apple AirPods Pro 3（实测） | 本地 UI（实测） | 处理 |
|---|---|---|---|
| Hero 字号 | 96px | 72px（clamp 上限） | A1 |
| Hero 字重 | 600 | 700 | A1 |
| Hero 行高 | 1.04 | 1.05 | 不变 |
| Hero 字距 | -1.44px ≈ -0.015em | -0.04em | A1 |
| Hero 文案 | 多行断句 | 单行 | A2 |
| Hero 阴影 | 无 | `0 0 80px` 蓝色辉光 | B8 |
| Eyebrow | 28px/600/非大写/+0.196px | 12px/大写/+1.44px/蓝 | A4 |
| 正文 | 17px | 15px | A3 |
| CTA | 17px/400/`11px 21px`/-0.374px | 14px/600/`12px`/normal | A6 |
| 滚动吸附 | none，高度自由 | `y proximity`，四段 `100dvh` | G2 |
| 玻璃模糊 | 几乎不用 | 全站 `backdrop-filter` | B7 |
| 导航/内容左对齐 | 一致 | 导航 1200 / 内容 1400（错位 68px） | G1 |

## 内容宽度基准

**统一 `max-width: 1200px`**（用户拍板）。涉及所有居中容器：`.snap-page-inner`、`.navbar-inner`、`.algo-carousel`、`.inference-toolbar`、`.result-dashboard`、`.compare-wall`、`.compare-shared-image` 容器。

## 1. 排印（A1–A6）

| 编号 | 改动 | 位置 |
|---|---|---|
| A1 | Hero `.hero-title`：字重 `700→600`、字距 `-0.04em→-0.01em`、`clamp` 上限 `72→96px`、行高维持 `1.05` | `apple-redesign.css:75-82` |
| A2 | Hero 文案多行断句：`工业缺陷检测` → `工业级 / 无监督 / 缺陷检测`（`<br>`） | `index.html:275` |
| A3 | 正文 `html { font-size: 15px → 17px }` | `app.css:221` |
| A4 | `.section-kicker`：字号 `12→22px`、去 `text-transform:uppercase`、字距 `+1.44px→+0.1px`、颜色 `--accent→--text-secondary` | `app.css` |
| A5 | `.hero-subtitle` 去胶囊：删 `background/border/border-radius/backdrop-filter`，改普通段落（21px/400/`--text-secondary`），保留三词 `·` 分隔 | `apple-redesign.css:84-109` |
| A6 | 按钮几何（`.btn-primary`/`.btn-inference`/`.training-start-btn`）：字重 `600→500`、内距 `12px→11px 21px`、字距加 `-0.37px`；胶囊形与圆角不变 | `apple-redesign.css` + `app.css` |

> 注：A1 行高规范历史上曾被写成 0.92，实测 Apple 为 1.04；本地已是 1.05，**不改回 0.92**。

## 2. 装饰精简（B7–B8）

| 编号 | 改动 | 位置 |
|---|---|---|
| B7 | 移除全局 `backdrop-filter: blur() saturate()`：算法卡、工具栏、下拉触发器、结果卡、对比槽、来源选择器。亮色改纯白卡 + 1px `--sep-subtle` + `--shadow-sm`；暗色保持 `--bg-system` 卡面，仅去模糊。 | `apple-redesign.css:212,443,678,705,1180` 等 |
| B8 | 删 `.hero-title { text-shadow: 0 0 80px var(--accent-glow)... }` | `apple-redesign.css:80` |

## 3. 全局布局（G1–G3）

| 编号 | 改动 | 位置 |
|---|---|---|
| G1 | `.navbar-inner { max-width: 1200px }`（现已是 1200，无需改数值，但确认其与内容基准一致）。使导航 logo 与页面标题同左 x。 | `app.css:442` |
| G2 | 非 Hero 页放开吸附：`.snap-page`（非 `.snap-page--home`）`min-height: 100dvh → auto`，上下 `padding: max(120px,12vh) 32px`。保留 `.snap-page--home` 的 `scroll-snap-align/stop`。`app.js` 的 `IntersectionObserver` 逻辑不变（仍以 `.snap-container` 为 root）。 | `app.css:185-192` + `index.html:43-52` |
| G3 | 统一内容宽 `max-width: 1200px`：`.snap-page-inner`（1400→1200）、`.navbar-inner`（1200，已对齐）、`.algo-carousel`（1200，已对齐）、`.inference-toolbar`（1000→1200）、`.result-dashboard`（见 I2）、`.compare-wall`（1400→1200）、`.compare-shared-image` 容器（见 C1）。 | 多处 `max-width` |

> `.snap-page--home`（首页）保留 `min-height:100dvh` + 吸附，是唯一全屏 snap 页。

## 4. Hero（H1–H2）

| 编号 | 改动 | 位置 |
|---|---|---|
| H1 | 算法轮播下移：首页只留 Hero 标题/副标题 + 产品级视觉，算法轮播移到首页下方独立子区。首屏上下留白撑满。 | `index.html:272-596` |
| H2 | `.hero-visual` 放大留作 Hero 主视觉（用户拍板）：占约 40% 视高，替代夹缝小流程图（现 560×160）。SVG 流程图在无真实检测结果时充当产品主角。 | `index.html:279-297` + `apple-redesign.css:111-160` + `js/hero-visual.js`/`animations.js` |

## 5. page-3 推理 + 结果展示（I1–I5，已确认左图右信息分栏）

目标结构（桌面）：
```
┌─ result-dashboard (max-width 1200) ─────────────────────┐
│  ┌──────────────────────┐  ┌────────────────────────┐  │
│  │                      │  │ ● 异常                 │  │
│  │  对比滑块（热力图）    │  │ PatchCore · bottle     │  │
│  │     ▍handle          │  │                        │  │
│  │  [图例 overlay]       │  │ 0.9234  异常得分        │  │
│  │                      │  │ 92.4%   置信度          │  │
│  │                      │  │ 0.821   阈值 τ          │  │
│  │                      │  │                        │  │
│  │                      │  │ 重新选择 →              │  │
│  └──────────────────────┘  └────────────────────────┘  │
│          图 ~60% (1.5fr)        信息 ~40% (1fr)        │
└────────────────────────────────────────────────────────┘
```

| 编号 | 改动 | 位置 |
|---|---|---|
| I1 | `.inference-toolbar { max-width: 1000→1200px }`，与结果卡对齐，消除宽度跳变。 | `apple-redesign.css:393` |
| I2 | `.result-dashboard` 统一 `max-width: 1200px`；删除 redesign 中 1400 覆盖与 `app.css:1679` 的 780 窄卡，二者统一到 1200。 | `apple-redesign.css:484-492` + `app.css:1679` |
| I3 | `.result-dashboard` 内部改 `display:grid; grid-template-columns: 1.5fr 1fr; gap:32px`。左列 `.result-dashboard-compare`（对比滑块，占满左列高，图例 overlay 留左下）；右列 `.result-dashboard-aside` 纵向堆叠：顶部 `inference-verdict` 大判定 → 三指标（得分/置信度/阈值，纵向大字）→ 底部「重新选择 →」。移动端 ≤768px 退回单列纵向堆叠。 | `index.html:974-1050` + `apple-redesign.css` |
| I4 | 去重复 badge：删标题栏右侧 `.result-badge`，只留信息栏顶部 `inference-verdict`。标题栏保留「模型·数据集」meta。 | `index.html:981-984` |
| I5 | 标题栏简化：`.result-dashboard-label` 并入右信息栏顶部（陈述式），标题栏仅留 meta 一行。 | `index.html:976-980` |

## 6. page-4 四模型对比（C1–C2）

| 编号 | 改动 | 位置 |
|---|---|---|
| C1 | `.compare-shared-image { max-width: 360→560px }`，居中成产品主图，下接对比墙。 | `apple-redesign.css:739-742` |
| C2 | 对比墙维持 4 列（原图上移到顶部主视觉后纵向空间更松）。若实测仍挤，回退 2×2 大格。 | `apple-redesign.css:693-699` |

## 受影响文件

- `modules/ui/static/index.html` — 结构（Hero 文案/重排、结果分栏 DOM、标题栏简化、共享原图容器）
- `modules/ui/static/css/app.css` — 排印 `:root` 基准、`.navbar-inner`、`.snap-page`、`.snap-page-inner`、`.result-dashboard`、`.compare-shared-image`、`.section-kicker`
- `modules/ui/static/css/apple-redesign.css` — Hero、玻璃移除、按钮、结果分栏、工具栏/结果卡宽度
- `modules/ui/static/js/hero-visual.js` / `animations.js` — heroVisual 放大后的尺寸/动画适配
- `modules/ui/static/js/app.js` — 验证 snap 放开后 `currentSection`/进度环仍正确（预计无需改）
- `tests/test_ui_static.py` — HTML 结构变动需同步断言

## 验收标准

1. 实测 `.navbar-inner`、`.snap-page-inner`、`.algo-carousel`、`.inference-toolbar`、`.result-dashboard` 左边界 x 一致（误差 ≤1px）。
2. 实测 `.hero-title` 字重 600、字距 ≈-0.01em、字号达 96px（宽屏）；无 `text-shadow`。
3. 推理结果卡为 `1.5fr 1fr` 双列；左列对比滑块为视觉主体；标题栏无重复 badge。
4. 非 Hero 页 `min-height` 为 `auto`，首页保留 `100dvh` + snap。
5. 内容卡（算法卡/工具栏/下拉/结果卡/对比槽/来源选择器）不再使用 `backdrop-filter`。导航栏 `.navbar` 保留其现有磨砂玻璃效果（固定悬浮元素，属功能性而非装饰性，本规范不动）。
6. `python -m pytest tests/ -v` 通过（结构断言已同步）。
7. 浏览器实测 1440px 与 1920px 两个宽度下布局无错位。

## 风险

- **G2 放开 snap** 后页面总高变长，`currentSection`/进度环依赖 IntersectionObserver，需验证滚动到底仍正确。
- **G3 内容宽改 1200** 涉及多处，需全局搜 `max-width` 逐一核对，避免漏改导致仍错位。
- **B7 去玻璃** 后暗色卡面靠 `--bg-system: rgba(255,255,255,0.055)` 维持，需确认去模糊后对比度仍够。
- **H2 heroVisual 放大** 需同步 `js/hero-visual.js`/`animations.js` 的尺寸与动画假设（当前基于 560×160）。
- **测试同步**：`tests/test_ui_static.py` 断言 DOM 结构，结构改动必须同步更新，否则测试失败。

## 不在本规范内（明确排除）

- 默认主题翻 light、`--accent` 数值、亮/暗色板、章节背景交替——颜色系统不动。
- SF Pro 字体自托管——单独一轮评估。
- Training Studio 内部布局——本次不动（其布局已较成熟）。
