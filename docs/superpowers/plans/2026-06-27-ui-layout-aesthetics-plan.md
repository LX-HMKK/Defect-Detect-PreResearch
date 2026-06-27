# UI 排印 + 布局对齐 AirPods Pro 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将本地工业异常检测 UI 的排印、装饰与布局对齐 Apple AirPods Pro 3 官网体感（颜色系统不动）。

**Architecture:** 纯前端改造（CSS + HTML + 极少 JS 验证），无后端/数据流变更。分 12 个可独立提交的任务，按风险递增排序：先 CSS 数值（低风险）→ 结构重排（高风险）。每任务含 DevTools 实测验证 + pytest 回归。服务器已在 `127.0.0.1:8000` 运行（FastAPI 静态资源 `no-cache`，浏览器硬刷新即生效）。

**Tech Stack:** FastAPI + Alpine.js SPA、CSS 自定义属性、Chrome DevTools MCP（实测验证）、pytest（静态结构断言）。

**设计规范：** `docs/superpowers/specs/2026-06-27-ui-layout-aesthetics-design.md`

**关键约定：**
- 服务器启动（若未运行）：`python scripts/run_ui.py --no-browser` → `http://127.0.0.1:8000`
- 每次验证：先 `navigate_page reload`，再 `evaluate_script` 取计算样式，必要时 `resize_page` + `take_screenshot`
- pytest：`python -m pytest tests/ -v`（须 40/40 或新增后全过）
- 提交规范：Angular 中文，禁止 `Co-authored-by`，多行用 `git commit -F .git-msg`

---

## 文件结构

| 文件 | 责任 | 本计划改动 |
|---|---|---|
| `modules/ui/static/css/app.css` | 主样式表 `:root` 变量、布局基础、组件 | 排印基准、`.snap-page`、`.snap-page-inner`、hero-title/subtitle/kicker 基础、按钮、`max-width` 统一、result-dashboard |
| `modules/ui/static/css/apple-redesign.css` | AirPods 风格覆盖层（加载在 app.css 之后，同特异性胜出） | hero-title/subtitle 覆盖、玻璃移除、按钮几何、result-dashboard 分栏、宽度统一 |
| `modules/ui/static/index.html` | Alpine SPA 入口、DOM 结构 | hero 文案/重排、result-dashboard 分栏 DOM、标题栏简化 |
| `modules/ui/static/js/hero-visual.js` | hero SVG 动画（CSS 驱动） | 无需改（已确认 viewBox 单位，缩放安全） |
| `tests/test_ui_static.py` | 静态结构断言 | 新增 result-dashboard 分栏结构断言 |

> **覆盖关系提醒：** `apple-redesign.css` 在 `index.html` 中晚于 `app.css` 加载（`:12-14`），同特异性规则 redesign 胜出。故 hero-title/subtitle 的"有效值"在 redesign 文件；但 app.css 基础值若不被 redesign 显式覆盖仍会生效（如 text-shadow）。**修改这类双定义属性时，两文件都要改**，否则基础值会"漏"出来。

---

## Task 1: 全局内容宽度统一到 1200px（G3 + G1）

**Files:**
- Modify: `modules/ui/static/css/app.css:200-205`（`.snap-page-inner` max-width 1400→1200）
- Modify: `modules/ui/static/css/app.css:442`（`.navbar-inner` 确认已是 1200）
- Modify: `modules/ui/static/css/app.css:1679`（`.result-dashboard` max-width 780→1200）
- Modify: `modules/ui/static/css/apple-redesign.css:485`（`.result-dashboard` max-width 1400→1200）
- Modify: `modules/ui/static/css/apple-redesign.css:697`（`.compare-wall` max-width 1400→1200）

- [ ] **Step 1: 改 `.snap-page-inner` max-width**

`app.css:202`：
```css
    max-width: 1200px;   /* 原 1400px — 统一内容基准 */
```

- [ ] **Step 2: 改 `.result-dashboard` 两处 max-width**

`app.css:1679`：
```css
    max-width: 1200px;   /* 原 780px */
```
`apple-redesign.css:485`：
```css
    max-width: 1200px;   /* 原 1400px */
```

- [ ] **Step 3: 改 `.compare-wall` max-width**

`apple-redesign.css:697`：
```css
    max-width: 1200px;   /* 原 1400px */
```

- [ ] **Step 4: 确认 navbar-inner 已是 1200**

Read `app.css:442`，确认 `max-width: 1200px`（实测已是）。若不是则改为 1200。无需改则跳过。

- [ ] **Step 5: DevTools 验证左边界对齐**

`navigate_page` reload → `evaluate_script`：
```js
() => {
  const x = (sel) => { const e=document.querySelector(sel); return e?Math.round(e.getBoundingClientRect().x):null; };
  return {
    navbar: x('.navbar-inner'),
    content: x('.snap-page-inner'),
    toolbar: x('.inference-toolbar'),
    dashboard: x('.result-dashboard'),
    compareWall: x('.compare-wall')
  };
}
```
期望：五个 x 值接近一致（误差 ≤2px，因各容器 padding 不同可能有 1-2px 差）。

- [ ] **Step 6: pytest 回归**

Run: `python -m pytest tests/test_ui_static.py -v`
Expected: PASS（本任务不改 HTML 结构，全过）。

- [ ] **Step 7: Commit**

```bash
git add modules/ui/static/css/app.css modules/ui/static/css/apple-redesign.css
git commit -F .git-msg
```
.git-msg:
```
style(ui): 统一内容宽度至 1200px 对齐导航

snap-page-inner/result-dashboard/compare-wall 统一 max-width:1200px，
消除导航(1200)与内容(1400)左边界错位。
```
完成后 `rm .git-msg`。

---

## Task 2: 正文基准字号 17px（A3）

**Files:**
- Modify: `modules/ui/static/css/app.css:221`

- [ ] **Step 1: 改 html font-size**

`app.css:221`：
```css
    font-size: 17px;   /* 原 15px — 对齐 Apple 正文 17px */
```

- [ ] **Step 2: DevTools 验证**

reload → `evaluate_script`：
```js
() => ({ bodyFs: getComputedStyle(document.body).fontSize })
```
期望：`17px`。

- [ ] **Step 3: pytest + Commit**

Run: `python -m pytest tests/test_ui_static.py -v` → PASS。
```bash
git add modules/ui/static/css/app.css
git commit -m "style(ui): 正文基准字号 15px→17px 对齐 Apple"
```

---

## Task 3: Hero 标题排印重标定 + 去辉光（A1 + B8）

**Files:**
- Modify: `modules/ui/static/css/apple-redesign.css:75-82`（hero-title 覆盖，有效值）
- Modify: `modules/ui/static/css/app.css:1012-1023`（hero-title 基础，去残留辉光）

- [ ] **Step 1: 改 redesign hero-title（有效值）**

`apple-redesign.css:75-82`，整块替换为：
```css
.hero-title {
    font-size: clamp(56px, 8vw, 96px);
    font-weight: 600;
    letter-spacing: -0.01em;
    line-height: 1.05;
    margin-bottom: 12px;
}
```
（删 `text-shadow` 行；字重 700→600；字距 -0.04em→-0.01em；clamp 上限 72→96、下限 48→56。）

- [ ] **Step 2: 删 app.css 基础 hero-title 残留辉光**

`app.css:1012-1023`，将 `text-shadow` 行删除：
```css
.hero-title {
    font-family: var(--font-display);
    font-size: 80px;
    font-weight: 600;          /* 原 700，与 redesign 一致 */
    letter-spacing: -0.01em;   /* 原 -0.04em */
    line-height: 1.05;
    color: var(--text);
    /* text-shadow 已移除（对齐 Apple 干净标题） */
    transition: opacity 0.4s var(--ease-out),
                transform 0.5s var(--ease-out-expo);
}
```
（基础值与 redesign 对齐，防 redesign 缺失时漏出旧值；text-shadow 删除避免暗色漏辉光。）

- [ ] **Step 3: DevTools 验证计算样式**

reload → `evaluate_script`：
```js
() => {
  const s = getComputedStyle(document.querySelector('.hero-title'));
  return { fw: s.fontWeight, ls: s.letterSpacing, fs: s.fontSize, ts: s.textShadow };
}
```
期望（宽屏 ≥1200px）：`fw:"600"`、`ls` 约 `"-0.96px"`（96px×-0.01）、`fs:"96px"`、`ts:"none"`。

- [ ] **Step 4: pytest + Commit**

Run: `python -m pytest tests/test_ui_static.py -v` → PASS。
```bash
git add modules/ui/static/css/app.css modules/ui/static/css/apple-redesign.css
git commit -m "style(ui): Hero 标题 96px/600/-0.01em 并去除辉光"
```

---

## Task 4: Hero 文案多行断句（A2）

**Files:**
- Modify: `modules/ui/static/index.html:275`

- [ ] **Step 1: 改 hero-title 文案为多行**

`index.html:275`，整行替换：
```html
                    <h1 class="hero-title scroll-reveal">工业级<br>无监督<br>缺陷检测</h1>
```

- [ ] **Step 2: DevTools 验证多行渲染**

reload → `evaluate_script`：
```js
() => {
  const e = document.querySelector('.hero-title');
  return { html: e.innerHTML, h: Math.round(e.getBoundingClientRect().height), lh: getComputedStyle(e).lineHeight };
}
```
期望：`html` 含两个 `<br>`；`h` 显著增大（三行，约 300px+）；`lh` 约 `100.8px`。

- [ ] **Step 3: pytest + Commit**

Run: `python -m pytest tests/test_ui_static.py -v` → PASS（断言未涉及此文本）。
```bash
git add modules/ui/static/index.html
git commit -m "style(ui): Hero 标题改多行断句制造呼吸感"
```

---

## Task 5: Hero 副标题去胶囊 + kicker 重做（A5 + A4）

**Files:**
- Modify: `modules/ui/static/css/apple-redesign.css:84-96`（hero-subtitle 覆盖，改纯文字）
- Modify: `modules/ui/static/css/app.css:1040-1053`（hero-subtitle 基础，去玻璃）
- Modify: `modules/ui/static/css/app.css:992-1002`（section-kicker 重做）

- [ ] **Step 1: 改 redesign hero-subtitle 为纯文字段落**

`apple-redesign.css:84-96`，整块替换为：
```css
.hero-subtitle {
    display: block;
    margin-top: 16px;
    font-size: 21px;
    font-weight: 400;
    line-height: 1.5;
    color: var(--text-secondary);
}
```
（删 background/border/border-radius/backdrop-filter；改 block 段落。）

- [ ] **Step 2: 改 app.css hero-subtitle 基础去玻璃**

`app.css:1040-1053`，将玻璃相关行删除：
```css
.hero-subtitle {
    font-family: var(--font-body);
    font-size: 21px;
    color: var(--text-secondary);
    margin-top: 16px;
    line-height: 1.5;
}
```
（删 `display:inline-block`、`padding`、`background`、`backdrop-filter` 两行、`border`、`border-radius`。）

- [ ] **Step 3: 重做 section-kicker**

`app.css:992-1002`，整块替换为：
```css
.section-kicker {
    display: block;
    text-align: center;
    font-size: 22px;
    font-weight: 600;
    letter-spacing: 0.1px;
    color: var(--text-secondary);
    margin-bottom: 12px;
}
```
（12→22px；去 `text-transform:uppercase`；0.12em→0.1px；`--accent`→`--text-secondary`；opacity 行删。）

- [ ] **Step 4: DevTools 验证**

reload → `evaluate_script`：
```js
() => {
  const sub = getComputedStyle(document.querySelector('.hero-subtitle'));
  const kck = getComputedStyle(document.querySelector('.section-kicker'));
  return {
    sub: { bg: sub.backgroundColor, bd: sub.backdropFilter, fs: sub.fontSize, fw: sub.fontWeight, disp: sub.display },
    kck: { fs: kck.fontSize, fw: kck.fontWeight, tt: kck.textTransform, ls: kck.letterSpacing, color: kck.color }
  };
}
```
期望：`sub.bg` 透明（`rgba(0,0,0,0)`）、`sub.bd` 为 `none`、`sub.disp:"block"`；`kck.fs:"22px"`、`kck.tt:"none"`、`kck.color` 为 text-secondary（暗色下 `rgba(255,255,255,0.6)`）。

- [ ] **Step 5: pytest + Commit**

Run: `python -m pytest tests/test_ui_static.py -v` → PASS。
```bash
git add modules/ui/static/css/app.css modules/ui/static/css/apple-redesign.css
git commit -m "style(ui): Hero 副标题去玻璃改纯文字、kicker 重做为 22px 非大写"
```

---

## Task 6: 按钮几何对齐 Apple CTA（A6）

**Files:**
- Modify: `modules/ui/static/css/app.css:780-804`（按钮共享规则）
- Modify: `modules/ui/static/css/apple-redesign.css:468-471`（inference-toolbar btn-primary 内距覆盖）

- [ ] **Step 1: 改按钮共享规则**

`app.css:790-797`，改 padding/font-weight 并加 letter-spacing：
```css
    padding: 11px 21px;       /* 原 10px 28px — Apple CTA 11px 21px */
    border-radius: 999px;
    background: var(--text);
    color: var(--bg-root);
    font-family: var(--font-body);
    font-size: 17px;           /* 原 15px */
    font-weight: 500;          /* 原 600 */
    letter-spacing: -0.37px;   /* 新增 */
    border: 1px solid transparent;
```
（在 `:790` 的 `padding` 行改值；`:795` font-size 改 17；`:796` font-weight 改 500；`:797` 后插入 letter-spacing 行。）

- [ ] **Step 2: 对齐 inference-toolbar btn-primary 覆盖**

`apple-redesign.css:468-471`，整块替换为：
```css
.inference-toolbar .btn-primary {
    padding: 11px 21px;
    font-size: 17px;
}
```
（原 `12px 28px`/`15px`，与共享规则统一。）

- [ ] **Step 3: DevTools 验证**

reload → 滚到 page-2（训练页，有 `.training-start-btn`）→ `evaluate_script`：
```js
() => {
  const b = document.querySelector('.training-start-btn');
  if (!b) return null;
  const s = getComputedStyle(b);
  return { pad: s.padding, fs: s.fontSize, fw: s.fontWeight, ls: s.letterSpacing };
}
```
期望：`pad:"11px 21px"`（或等价的 `11px 21px 11px 21px`）、`fs:"17px"`、`fw:"500"`、`ls` 约 `"-0.37px"`。

- [ ] **Step 4: pytest + Commit**

Run: `python -m pytest tests/test_ui_static.py -v` → PASS。
```bash
git add modules/ui/static/css/app.css modules/ui/static/css/apple-redesign.css
git commit -m "style(ui): 按钮几何对齐 Apple CTA 17px/500/11px21px"
```

---

## Task 7: 去全局玻璃 — 内容卡（B7）

**Files:**
- Modify: `modules/ui/static/css/app.css:532-533`（`.custom-select-trigger`）
- Modify: `modules/ui/static/css/app.css:607-608`（`.custom-select-menu`）
- Modify: `modules/ui/static/css/app.css:1050-1051`（`.hero-subtitle` 基础，Task 5 已删，确认无残留）
- Modify: `modules/ui/static/css/app.css:2563-2564`（`.compare-summary-row`）
- Modify: `modules/ui/static/css/app.css:2942-2943`（`.training-*-card`）
- Modify: `modules/ui/static/css/apple-redesign.css:212-213`（`.algo-card--slide`）
- Modify: `modules/ui/static/css/apple-redesign.css:443-444`（`.inference-toolbar .custom-select-trigger`）
- Modify: `modules/ui/static/css/apple-redesign.css:643-644`（`.pipeline.workbench-row .pipeline-step`）
- Modify: `modules/ui/static/css/apple-redesign.css:678-679`（`.result-dashboard`）
- Modify: `modules/ui/static/css/apple-redesign.css:705-706`（`.compare-wall .compare-slot`）
- Modify: `modules/ui/static/css/apple-redesign.css:1180-1181`（`.source-selector--glass`）

> **保留**（功能性悬浮/固定，不动）：`.navbar`(415)、`.heatmap-legend--overlay`(1763)、`.hm-tooltip`(2122)、`.compare-shared-label`(2489)、`.toast-pill`(2692)、移动端 `.snap-dots`(2823)。

- [ ] **Step 1: 删 app.css 内容卡 backdrop-filter（4 处）**

逐一删除 `backdrop-filter` 与 `-webkit-backdrop-filter` 两行（保留该规则其余属性）：
- `app.css:532-533`（`.custom-select-trigger`）
- `app.css:607-608`（`.custom-select-menu`）
- `app.css:2563-2564`（`.compare-summary-row`）
- `app.css:2942-2943`（`.training-config-card, .training-gallery-card, .training-monitor-card`）

> `app.css:1050-1051`（hero-subtitle）已在 Task 5 删除，确认无残留即可。

- [ ] **Step 2: 删 apple-redesign.css 内容卡 backdrop-filter（6 处）**

逐一删除两行：
- `:212-213`（`.algo-card--slide`）
- `:443-444`（`.inference-toolbar .custom-select-trigger`）
- `:643-644`（`.pipeline.workbench-row .pipeline-step`）
- `:678-679`（`.result-dashboard`）
- `:705-706`（`.compare-wall .compare-slot`）
- `:1180-1181`（`.source-selector--glass`）
- `:92-93`（`.hero-subtitle`）—— Task 5 已删，确认无残留。

- [ ] **Step 3: DevTools 验证内容卡无 blur**

reload → `evaluate_script`：
```js
() => {
  const sels = ['.algo-card--slide','.inference-toolbar','.custom-select-trigger','.result-dashboard','.compare-slot','.source-selector--glass','.compare-summary-row','.training-config-card'];
  return sels.map(sel => {
    const e = document.querySelector(sel);
    return { sel, bd: e ? getComputedStyle(e).backdropFilter : 'not-found' };
  });
}
```
期望：每个 `bd` 为 `"none"`。

- [ ] **Step 4: 验证保留项仍有效（navbar）**

```js
() => ({ navbar: getComputedStyle(document.querySelector('.navbar')).backdropFilter })
```
期望：`navbar` 仍含 `blur`（如 `blur(12px) saturate(1.4)`）。

- [ ] **Step 5: 暗色对比度检查**

切暗色（默认即暗色）→ 滚到训练页 → `take_screenshot`，肉眼确认 training 卡片文字在 `--bg-system: rgba(255,255,255,0.055)` 卡面上仍可读。若不可读，给卡补 `background: var(--bg-secondary)`（不恢复 blur）。

- [ ] **Step 6: pytest + Commit**

Run: `python -m pytest tests/test_ui_static.py -v` → PASS。
```bash
git add modules/ui/static/css/app.css modules/ui/static/css/apple-redesign.css
git commit -m "style(ui): 移除内容卡玻璃模糊，改纯白卡+细线（navbar 等悬浮元素保留）"
```

---

## Task 8: 放开非首页 snap 吸附（G2）

**Files:**
- Modify: `modules/ui/static/css/app.css:185-198`（`.snap-page` min-height + `.snap-page--home` 保留）

- [ ] **Step 1: 改 .snap-page min-height 为 auto + padding**

`app.css:185-192`，整块替换为：
```css
.snap-page {
    min-height: auto;
    display: flex;
    flex-direction: column;
    justify-content: center;
    position: relative;
    padding: max(120px, 12vh) 32px;
}
```
（`min-height: 100dvh` → `auto`；加上下 padding。）

- [ ] **Step 2: 确认 .snap-page--home 仍全屏吸附**

Read `app.css:194-198`，确认 `.snap-page--home` 仍保留 `scroll-snap-align: start; scroll-snap-stop: always;`。**额外**给首页强制 100dvh：在 `.snap-page--home` 规则内补 `min-height: 100dvh;`：
```css
/* 仅首页保留吸附 */
.snap-page--home {
    scroll-snap-align: start;
    scroll-snap-stop: always;
    min-height: 100dvh;
    padding: max(120px, 12vh) 32px;
}
```

- [ ] **Step 3: DevTools 验证页面高度与首页吸附**

reload → `evaluate_script`：
```js
() => {
  const pages = [...document.querySelectorAll('.snap-page')];
  return pages.map((p,i) => ({
    i, cls: p.className.slice(0,30), mh: getComputedStyle(p).minHeight, h: Math.round(p.getBoundingClientRect().height)
  }));
}
```
期望：首页（i=0）`mh:"100dvh"`；其余 `mh:"auto"` 且 `h` < 100dvh（内容自适应，非强制等高）。

- [ ] **Step 4: 验证导航/进度环仍正确**

滚到 page-2（训练）→ `evaluate_script`：
```js
() => ({ label: document.querySelector('.snap-dot-label')?.textContent })
```
期望：显示 `2 / 4`（IntersectionObserver 仍以 `.snap-container` 为 root，逻辑未改）。
> 若进度环错位：检查 `app.js` 的 `IntersectionObserver` root 是否为 `.snap-container`（CLAUDE.md 记载的陷阱），本任务不改 JS。

- [ ] **Step 5: pytest + Commit**

Run: `python -m pytest tests/test_ui_static.py -v` → PASS。
```bash
git add modules/ui/static/css/app.css
git commit -m "style(ui): 非 Hero 页放开 scroll-snap 为自然滚动，仅首页保留全屏吸附"
```

---

## Task 9: Hero 重排 — heroVisual 放大 + 首屏聚焦（H1 + H2）

**Files:**
- Modify: `modules/ui/static/css/apple-redesign.css:69-117`（hero 区 + hero-visual 放大）
- Modify: `modules/ui/static/index.html:272-298`（hero 加全屏包裹类）
- Verify: `modules/ui/static/js/hero-visual.js`（无需改，已确认 viewBox 单位安全）

- [ ] **Step 1: 给 hero 加全屏聚焦类**

`index.html:274`，`.hero` 加修饰类：
```html
                <div class="hero hero--fullscreen">
```

- [ ] **Step 2: 加 .hero--fullscreen CSS（撑满首屏，轮播自然下移到折叠下方）**

`apple-redesign.css`，在 `.hero { ... }`（:69-73）后新增：
```css
.hero--fullscreen {
    min-height: calc(100dvh - 48px);   /* 减去 navbar 高度 */
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
}
```

- [ ] **Step 3: 放大 hero-visual**

`apple-redesign.css:112-117`，改 `.hero-visual`：
```css
.hero-visual {
    width: 100%;
    max-width: 920px;        /* 原 560px — 放大为主视觉 */
    margin: 32px auto 0;
    aspect-ratio: 3 / 1;     /* 原 3.5/1 — 对齐 SVG viewBox 720×240 */
}
```

- [ ] **Step 4: DevTools 验证首屏聚焦 + 视觉放大**

reload → `evaluate_script`：
```js
() => {
  const hero = document.querySelector('.hero--fullscreen');
  const vis = document.querySelector('.hero-visual');
  return {
    heroH: hero ? Math.round(hero.getBoundingClientRect().height) : null,
    heroMinH: hero ? getComputedStyle(hero).minHeight : null,
    visW: vis ? Math.round(vis.getBoundingClientRect().width) : null,
    visAr: vis ? getComputedStyle(vis).aspectRatio : null,
    // 算法轮播是否在首屏下方（y > viewport h）
    carouselY: (() => { const c=document.querySelector('.algo-carousel'); return c?Math.round(c.getBoundingClientRect().top):null; })()
  };
}
```
期望：`heroH` ≈ viewport h − 48；`visW` ≈ 920（或 viewport 限宽）；`visAr:"3 / 1"`；`carouselY` > 0（轮播在首屏之下，需滚动可见）。

- [ ] **Step 5: 截图验证（1440 + 1920）**

`resize_page` 1440×932 → `take_screenshot`；`resize_page` 1920×1080 → `take_screenshot`。确认 hero 居中、视觉放大不溢出、轮播在折叠下方。

- [ ] **Step 6: pytest + Commit**

Run: `python -m pytest tests/test_ui_static.py -v` → PASS（`algo-carousel`/`bento-grid`/`hero-visual` 类名仍在）。
```bash
git add modules/ui/static/index.html modules/ui/static/css/apple-redesign.css
git commit -m "feat(ui): Hero 首屏聚焦 + heroVisual 放大为主视觉，轮播下移"
```

---

## Task 10: page-3 推理工具栏宽度统一（I1）

**Files:**
- Modify: `modules/ui/static/css/apple-redesign.css:393`（`.inference-toolbar` max-width 1000→1200）

- [ ] **Step 1: 改 inference-toolbar max-width**

`apple-redesign.css:393`：
```css
    max-width: 1200px;   /* 原 1000px — 与结果卡对齐，消除宽度跳变 */
```

- [ ] **Step 2: DevTools 验证工具栏与结果卡同宽**

reload → 滚到 page-3（单模型推理）→ `evaluate_script`：
```js
() => {
  const tb = document.querySelector('.inference-toolbar');
  const rd = document.querySelector('.result-dashboard');
  return {
    tbW: tb?Math.round(tb.getBoundingClientRect().width):null,
    tbMax: tb?getComputedStyle(tb).maxWidth:null,
    rdMax: rd?getComputedStyle(rd).maxWidth:null
  };
}
```
期望：`tbMax` 与 `rdMax` 均为 `1200px`。（`rd` 在 idle 态可能不存在，需先点"开始推理"或忽略 rd 项。）

- [ ] **Step 3: pytest + Commit**

Run: `python -m pytest tests/test_ui_static.py -v` → PASS。
```bash
git add modules/ui/static/css/apple-redesign.css
git commit -m "style(ui): 推理工具栏宽度 1000→1200 对齐结果卡"
```

---

## Task 11: page-3 结果展示改左图右信息分栏（I3 + I4 + I5）

**Files:**
- Modify: `tests/test_ui_static.py`（新增分栏结构断言）
- Modify: `modules/ui/static/index.html:973-1050`（result-dashboard DOM 重构）
- Modify: `modules/ui/static/css/apple-redesign.css`（新增分栏样式）

- [ ] **Step 1: 写失败测试（TDD）**

在 `tests/test_ui_static.py` 末尾追加：
```python
def test_result_dashboard_uses_split_layout():
    """推理结果卡应为左图右信息分栏（I3），移除旧标题栏与重复 badge（I4/I5）。"""
    text = _html_text()
    assert "result-dashboard-grid" in text
    assert "result-dashboard-aside" in text
    assert "inference-metrics--stack" in text
    # 旧的标题栏与重复 badge 已移除
    assert "result-dashboard-header" not in text
    assert "result-badge" not in text
    assert "inference-verdict-wrap" not in text
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_ui_static.py::test_result_dashboard_uses_split_layout -v`
Expected: FAIL（`result-dashboard-grid` not in text）。

- [ ] **Step 3: 重构 result-dashboard DOM**

`index.html:973-1050`，将 `<div class="result-dashboard ...">` 到对应 `</template>` 之间整块替换为：
```html
                    <template x-if="inferenceState === 'done' && resultData">
                    <div class="result-dashboard scroll-reveal" x-transition>
                        <!-- 左图右信息分栏 -->
                        <div class="result-dashboard-grid">
                            <!-- 左列：对比滑块（主视觉） -->
                            <div class="result-dashboard-compare" x-data="imageCompare">
                                <div class="compare-container">
                                    <img :src="resultData.image_b64" class="compare-image compare-original" alt="原图">
                                    <img :src="resultData.heatmap_b64" class="compare-image compare-heatmap"
                                         :style="{ clipPath: 'inset(0 ' + (100 - sliderPos) + '% 0 0)' }" alt="热力图">
                                    <div class="compare-handle" :style="{ left: sliderPos + '%' }"
                                         @pointerdown="startDrag">
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

                            <!-- 右列：信息栏 -->
                            <div class="result-dashboard-aside">
                                <div class="result-dashboard-aside-top">
                                    <span class="inference-verdict"
                                          :class="resultData.is_anomaly ? 'inference-verdict--anomaly' : 'inference-verdict--normal'"
                                          x-text="resultData.is_anomaly ? '● 异常' : '● 正常'"></span>
                                    <span class="result-dashboard-meta" x-text="resultData.model_name + ' · ' + selectedDataset"></span>
                                </div>
                                <div class="inference-metrics inference-metrics--stack">
                                    <div class="inference-metric">
                                        <div class="inference-metric-value result-score-value" x-text="resultData.score.toFixed(4)"></div>
                                        <div class="inference-metric-label">异常得分</div>
                                    </div>
                                    <div class="inference-metric">
                                        <div class="inference-metric-value result-confidence-value"
                                             x-text="(resultData.confidence * 100).toFixed(1) + '%'"></div>
                                        <div class="inference-metric-label">置信度</div>
                                    </div>
                                    <div class="inference-metric">
                                        <div class="inference-metric-value" x-text="resultData.threshold.toFixed(3)"></div>
                                        <div class="inference-metric-label">阈值 τ</div>
                                    </div>
                                </div>
                                <button class="btn-text result-reselect-btn" @click="resetInference()">重新选择 →</button>
                            </div>
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
（移除 `.result-dashboard-header`/`.result-dashboard-title`/`.result-dashboard-label`/`.result-badge`/`.inference-verdict-wrap`；新增 `.result-dashboard-grid`/`.result-dashboard-aside`/`.result-dashboard-aside-top`/`.inference-metrics--stack`/`.result-reselect-btn`。）

- [ ] **Step 4: 新增分栏样式**

`apple-redesign.css`，在 `.result-dashboard-header` 相关旧规则后（或文件 `.result-dashboard` 块后）新增：
```css
/* ── 结果卡：左图右信息分栏（AirPods Pro 工作台）── */
.result-dashboard-grid {
    display: grid;
    grid-template-columns: 1.5fr 1fr;
    gap: 32px;
    align-items: stretch;
}
.result-dashboard-compare {
    min-width: 0;
}
.result-dashboard-aside {
    display: flex;
    flex-direction: column;
    gap: 24px;
    justify-content: center;
    min-width: 0;
}
.result-dashboard-aside-top {
    display: flex;
    flex-direction: column;
    gap: 8px;
}
.result-dashboard-aside-top .result-dashboard-meta {
    font-size: 15px;
    font-weight: 500;
    color: var(--text-secondary);
}
/* 右栏纵向指标（替代原三列横排） */
.inference-metrics--stack {
    display: flex;
    flex-direction: column;
    gap: 20px;
    max-width: none;
    margin: 0;
    text-align: left;
}
.inference-metrics--stack .inference-metric {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 16px;
    padding-bottom: 16px;
    border-bottom: 1px solid var(--sep-subtle);
}
.inference-metrics--stack .inference-metric:last-child {
    border-bottom: none;
    padding-bottom: 0;
}
.result-reselect-btn {
    align-self: flex-start;
    margin-top: 8px;
}
@media (max-width: 768px) {
    .result-dashboard-grid { grid-template-columns: 1fr; gap: 24px; }
    .inference-metrics--stack .inference-metric { padding-bottom: 12px; }
}
```

- [ ] **Step 5: 运行测试确认通过**

Run: `python -m pytest tests/test_ui_static.py::test_result_dashboard_uses_split_layout -v`
Expected: PASS。

- [ ] **Step 6: DevTools 验证分栏布局**

reload → page-3 → 选图 + 点"开始推理" → 等结果 → `evaluate_script`：
```js
() => {
  const grid = document.querySelector('.result-dashboard-grid');
  if (!grid) return { err: 'grid not found (可能未触发推理完成态)' };
  const s = getComputedStyle(grid);
  const cmp = document.querySelector('.result-dashboard-compare');
  const asd = document.querySelector('.result-dashboard-aside');
  return {
    gridCols: s.gridTemplateColumns,
    cmpW: Math.round(cmp.getBoundingClientRect().width),
    asdW: Math.round(asd.getBoundingClientRect().width),
    ratio: (cmp.getBoundingClientRect().width / asd.getBoundingClientRect().width).toFixed(2),
    hasDupBadge: !!document.querySelector('.result-badge'),
    hasOldHeader: !!document.querySelector('.result-dashboard-header')
  };
}
```
期望：`gridCols` 含两列（`1.5fr 1fr` 或解析后值）；`ratio` ≈ 1.5；`hasDupBadge`/`hasOldHeader` 为 `false`。

- [ ] **Step 7: 截图验证（亮/暗双模式）**

暗色（默认）+ 切亮色各 `take_screenshot`，确认左图为主视觉、右信息栏纵向、无重复 badge。

- [ ] **Step 8: pytest 全量 + Commit**

Run: `python -m pytest tests/ -v` → 全 PASS。
```bash
git add tests/test_ui_static.py modules/ui/static/index.html modules/ui/static/css/apple-redesign.css
git commit -m "feat(ui): 推理结果改左图右信息分栏，移除重复 badge 与旧标题栏"
```

---

## Task 12: page-4 共享原图放大（C1 + C2）

**Files:**
- Modify: `modules/ui/static/css/apple-redesign.css:741`（`.compare-shared-image` max-width 360→560）

- [ ] **Step 1: 改 compare-shared-image max-width**

`apple-redesign.css:741`：
```css
    max-width: 560px;   /* 原 360px — 放大为产品主图 */
```

- [ ] **Step 2: DevTools 验证**

reload → page-4 → `evaluate_script`：
```js
() => {
  const e = document.querySelector('.compare-shared-image');
  const img = document.querySelector('.compare-shared-img');
  return {
    maxW: e ? getComputedStyle(e).maxWidth : null,
    imgW: img ? Math.round(img.getBoundingClientRect().width) : null
  };
}
```
期望：`maxW:"560px"`；`imgW` 接近 560（受原图比例与 max-height 限制）。

- [ ] **Step 3: 验证对比墙 4 列不挤**

`evaluate_script`：
```js
() => {
  const slots = [...document.querySelectorAll('.compare-slot')];
  return { count: slots.length, widths: slots.map(s=>Math.round(s.getBoundingClientRect().width)) };
}
```
期望：`count:4`，各列宽 ≈ (1200−3×20)/4 ≈ 285px。若实测热力图+双指标严重拥挤，则按 spec C2 回退方案：改 `.compare-wall { grid-template-columns: repeat(2, 1fr) }`（2×2 大格）。

- [ ] **Step 4: pytest + Commit**

Run: `python -m pytest tests/test_ui_static.py -v` → PASS。
```bash
git add modules/ui/static/css/apple-redesign.css
git commit -m "style(ui): 四模型对比共享原图放大为产品主图"
```

---

## Task 13: 全量验收

**Files:** 无（仅验证）

- [ ] **Step 1: pytest 全量**

Run: `python -m pytest tests/ -v`
Expected: 全 PASS（含新增 `test_result_dashboard_uses_split_layout`）。

- [ ] **Step 2: 双宽度截图回归**

`resize_page` 1440×932 → 逐页 `take_screenshot`（首页/训练/推理结果/对比）。
`resize_page` 1920×1080 → 同上。
确认：无错位、无溢出、左边界对齐、暗亮均可读。

- [ ] **Step 3: 关键计算样式终检**

`evaluate_script` 一次性核对：
```js
() => {
  const g = (sel,ps) => { const e=document.querySelector(sel); if(!e) return null; const s=getComputedStyle(e); const o={}; ps.forEach(p=>o[p]=s.getPropertyValue(p)); return o; };
  return {
    heroTitle: g('.hero-title',['font-weight','letter-spacing','font-size','text-shadow']),
    heroSub: g('.hero-subtitle',['backdrop-filter','background-color']),
    kicker: g('.section-kicker',['font-size','text-transform','color']),
    body: g('body',['font-size']),
    widths: {
      navbar: getComputedStyle(document.querySelector('.navbar-inner')).maxWidth,
      content: getComputedStyle(document.querySelector('.snap-page-inner')).maxWidth,
      toolbar: getComputedStyle(document.querySelector('.inference-toolbar')).maxWidth
    }
  };
}
```
期望核对：heroTitle font-weight 600 / text-shadow none；heroSub backdrop-filter none；kicker font-size 22px / text-transform none；body font-size 17px；三个 maxWidth 均 1200px。

- [ ] **Step 4: 提交记忆（可选）**

若验收中发现需记录的非显然约定，写入 `memory/`。

---

## 自审

**1. Spec 覆盖：** A1(T3)✓ A2(T4)✓ A3(T2)✓ A4(T5)✓ A5(T5)✓ A6(T6)✓ B7(T7)✓ B8(T3)✓ G1(T1 Step4)✓ G2(T8)✓ G3(T1)✓ H1(T9)✓ H2(T9)✓ I1(T10)✓ I2(T1+T11)✓ I3(T11)✓ I4(T11)✓ I5(T11)✓ C1(T12)✓ C2(T12)✓。排除项（颜色/字体/Training Studio 内部）均未涉及。**无遗漏。**

**2. 占位符扫描：** 无 TBD/TODO；每步含确切代码与命令。

**3. 类型一致性：** 新增类名 `.result-dashboard-grid`/`.result-dashboard-aside`/`.result-dashboard-aside-top`/`.inference-metrics--stack`/`.result-reselect-btn`/`.hero--fullscreen` 在 HTML 与 CSS 中一一对应；测试断言的类名与 DOM 一致。保留项（navbar/legend/tooltip/toast/snap-dots）在 T7 明确列出。

**4. 双定义陷阱已处理：** hero-title（app.css:1012 + redesign:75）、hero-subtitle（app.css:1040 + redesign:84）在 T3/T5 同时改两文件，防基础值漏出。
