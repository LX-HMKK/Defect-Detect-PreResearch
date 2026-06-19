# Phase 2 前端重构设计规范 — 工业缺陷检测系统

> 版本 1.0 · 2026-06-19 · Apple 官网级交互 · FastAPI + Alpine.js SPA
>
> 承接 [Phase 1 Apple UI 设计规范](2026-06-18-apple-ui-design-spec.md)，
> 保留 CSS 变量体系、色板、字体栈，新增五层动效系统和三区块自适应滚动布局。

---

## 1. 设计目标

1. **摆脱 Gradio 约束**：Gradio 的 Svelte 渲染层限制 JS/CSS 自由度（`<script>` 不执行、CSS 作用域处理、DOM 结构不可控）。自建前端壳获得完整控制权。
2. **Apple 官网级交互**：鼠标光晕跟随、滚动驱动动画、数字跳动计数、原图↔热力图对比滑块、胶囊主题开关、SVG 算法流程图。
3. **渐进式保留**：Gradio `demo.py` 完整保留作为 fallback，新 `server.py` 作为主入口。

---

## 2. 架构

```
┌─ 浏览器 ─────────────────────────────────────────┐
│  index.html (Alpine.js SPA)                       │
│  ├─ app.css    (~1500行, 全部视觉/动效)            │
│  ├─ app.js     (Alpine 全局状态 + 初始化)          │
│  ├─ inference.js  (推理逻辑 + SSE 流消费)          │
│  ├─ compare.js    (四模型对比逻辑)                 │
│  ├─ animations.js (WAAPI 动画库)                  │
│  ├─ flowchart.js  (SVG stroke 绘制)               │
│  └─ cursor-glow.js (鼠标光晕跟随)                 │
│                                                    │
│  fetch('/api/predict')  ← SSE 流式推理进度         │
│  fetch('/api/compare')  ← SSE 四模型对比           │
│  fetch('/api/models')   ← 可用模型/数据集列表       │
└────────────────────┬───────────────────────────────┘
                     │ HTTP + SSE
┌─ Python ──────────────────────────────────────────┐
│  modules/ui/server.py  (FastAPI)                  │
│  ├─ GET  /                  → index.html           │
│  ├─ GET  /api/models         → 模型/数据集列表     │
│  ├─ POST /api/predict        → SSE stream          │
│  └─ POST /api/compare        → SSE stream (4模型)  │
│                                                    │
│  复用: AnomalyDetector + MODEL_CONFIGS             │
│        (modules/ui/demo.py 核心逻辑不变)           │
└────────────────────────────────────────────────────┘

不动模块:
  modules/ui/demo.py             完整保留 (Gradio fallback)
  modules/ui/styles.css          完整保留 (Gradio fallback)
  modules/ui/theme.py            保留, 新增变量生成函数
  modules/ui/static/theme.js     保留 (Gradio fallback)
  modules/ui/static/inference-interact.js 保留
  scripts/run_ui.py              入口改为启动 FastAPI server
```

### 新增文件清单

| 文件 | 用途 |
|------|------|
| `modules/ui/server.py` | FastAPI 主服务器 (路由 + SSE + 静态文件) |
| `modules/ui/static/index.html` | SPA 入口 (Alpine.js CDN) |
| `modules/ui/static/css/app.css` | 完整样式 (~1500行) |
| `modules/ui/static/css/flowchart.css` | 流程图 SVG 动画 |
| `modules/ui/static/js/app.js` | Alpine 全局状态 + 初始化 |
| `modules/ui/static/js/inference.js` | 推理逻辑 + SSE 流 |
| `modules/ui/static/js/compare.js` | 四模型对比 + 并行 SSE |
| `modules/ui/static/js/animations.js` | WAAPI 动画库 |
| `modules/ui/static/js/flowchart.js` | SVG stroke-dashoffset 绘制 |
| `modules/ui/static/js/cursor-glow.js` | 鼠标光晕跟随 |
| `modules/ui/static/svg/logo.svg` | Logo 源文件 |
| `modules/ui/static/svg/patchcore-flow.svg` | PatchCore 流程图 |
| `modules/ui/static/svg/padim-flow.svg` | PaDiM 流程图 |
| `modules/ui/static/svg/fre-flow.svg` | FRE 流程图 |
| `modules/ui/static/svg/draem-flow.svg` | DRAEM 流程图 |

---

## 3. 布局：三区块自适应连续滚动

放弃 `scroll-snap` 强制 100vh 翻页（会产生割裂感），改为连续滚动 + 内容驱动自适应高度。

```
┌─ 导航栏 (sticky, 随滚动加深磨砂) ─────────────────────┐
│  [Logo]  缺陷检测    数据集▼  [胶囊主题开关]  ●○○     │
└───────────────────────────────────────────────────────┘

        ┌── Hero ──────────────────────────┐
        │  工业异常检测                      │
        │  四算法 · 无监督 · 实时推理         │  约 70vh
        └──────────────────────────────────┘
                ↓ 自然滚动 (无 snap)
        ┌── 算法介绍 ───────────────────────┐
        │  2×2 四宫格:                      │
        │  PatchCore 流程图  PaDiM 流程图    │  自适应高度
        │  FRE 流程图       DRAEM 流程图     │  (~200vh+)
        └──────────────────────────────────┘
                ↓
        ┌── 过渡带 ────────────────────────┐
        │  细线 + "选择算法开始推理" 提示     │  约 15vh
        └──────────────────────────────────┘
                ↓
        ┌── 单模型推理 ────────────────────┐
        │  上传 → 推理 → 结果卡片           │  自适应高度
        │  原图↔热力图 对比滑块 + 图例      │
        └──────────────────────────────────┘
                ↓
        ┌── 四模型对比 ────────────────────┐
        │  四列并排: PC | PaDiM | FRE | DRAEM│ 自适应高度
        │  一键运行, 逐个完成, 汇总对比      │
        └──────────────────────────────────┘
                ↓
        ┌── Footer ────────────────────────┐
        │  使用说明 · 版本信息               │
        └──────────────────────────────────┘
```

### 消除割裂感的机制

| 机制 | 实现方式 |
|------|---------|
| **滚动驱动动画** | Intersection Observer + WAAPI，元素滚入视口时触发，非页面切换时 |
| **视差过渡带** | 区块间 15-20vh 留白 + 细线装饰 |
| **导航点自适应** | 不是"切页"而是"当前视口中心落在哪个区块"，滞后跟随 spring 过渡 |
| **sticky 上下文标题** | 区块标题可短暂 sticky 在导航栏下方，暗示进入新区块 |
| **底部淡出** | 上一区块底部元素随滚动逐渐 opacity 降低 |

### 导航点

```
固定在页面右侧中间：
    ●  ── 算法介绍  (当前区块, 蓝色高亮)
    ○  ── 单模型推理
    ○  ── 四模型对比

- 当前区块判断: Intersection Observer，哪个区块占据视口中心 >50%
- 切换动画: spring 缩放 0.9→1.1→1.0 + 颜色渐变, 200ms
- 点击导航点 → 平滑滚动到对应区块 (scroll-behavior: smooth)
```

---

## 4. 动效体系（五层）

| 层级 | 名称 | 用途 | 技术 |
|------|------|------|------|
| **L0** | Ambient | 环境光呼吸、Logo 脉冲 | CSS animation + 错相 |
| **L1** | Cursor-Aware | 鼠标光晕跟随、磁吸悬浮 | mousemove → CSS vars |
| **L2** | Scroll-Triggered | 滚动到位触发动画 | Intersection Observer + WAAPI |
| **L3** | Micro-Interaction | 按钮涟漪/弹簧、开关切换、数字跳动 | WAAPI + CSS transition |
| **L4** | Page Transition | 视图切换 | View Transitions API (渐进增强) |

---

## 5. 组件详细设计

### 5.1 胶囊主题开关（替代原双按钮）

```
┌────────────────────┐
│  ☀️    ●    🌙    │  36×72px 玻璃质感胶囊
└────────────────────┘

交互：
- 点击 → 白色圆球 spring 滑到对侧 + 全局 400ms 颜色过渡
- 亮→暗: 冷蓝色微光闪一下 (box-shadow 瞬间增强再消退)
- 暗→亮: 暖金色微光闪一下
- hover: translateY(-1px) + shadow-lg 增强
- 键盘: Tab 可达, Enter/Space 切换

实现:
- Alpine x-data + @click 切换
- CSS transition 处理圆球滑动
- WAAPI KeyframeEffect 处理光晕闪烁
```

### 5.2 Logo 呼吸脉冲

```
菱形 Logo 的三个嵌套 <rect> 做缩放呼吸:
- 最外层 (22.63px):   scale 1.0↔1.06, 15s cycle, ease-in-out
- 中层 (16.97px):     scale 1.0↔1.05, 12s cycle, ease-in-out (错相)
- 内层 (12.02px):     scale 1.0↔1.04, 9s cycle,  ease-in-out (错相)

hover 时: 所有层加速到 2s cycle + 中心白点略微放大

实现: CSS @keyframes, animation-delay 错相
```

### 5.3 算法流程图（四张 SVG）

每张流程图结构一致，4-6 个节点 + 箭头连线。在卡片滚入视口时触发：

```
动画序列 (staggered):
1. 容器淡入 (opacity 0→1, 400ms)
2. 节点依次出现: stroke-dashoffset 绘制边框 (800ms/node, 150ms stagger)
3. 节点内文字淡入 (200ms, 延迟于节点)
4. 箭头连线: stroke-dashoffset 从左到右绘制 (600ms/条, staggered)

技术: SVG path.getTotalLength() + WAAPI animate
      Intersection Observer 触发
      threshold: 0.3 (卡片露出 30% 时开始)
```

### 5.4 图片上传区

```
空态:
┌─────────────────────────────┐
│          ☁️                 │  ← CSS 浮动动画 (translateY 6px, 3s)
│   拖拽图片到此处             │
│   或点击选择文件             │
│   支持 PNG / JPG / BMP      │
└─────────────────────────────┘

上传后:
- 虚线框→实线框 过渡 (border-style 变化, 300ms)
- 图片淡入
- 上传区缩小或收起

技术: dragenter/dragover/drop 事件 + FileReader
      >20MB 图片前端 canvas 缩放到 max 1024px
```

### 5.5 推理按钮

```
状态机:
[空闲]     → "开始推理", accent 背景, 微呼吸光晕
[上传完成]  → 光晕增强, 文字变亮 (吸引点击)
[加载模型]  → 文字 fadeOut → spinner 淡入, 不可点击
[推理中]    → spinner + "推理中…", 进度条 shimmer
[完成]      → ✓ "推理完成" 绿色, 2s 后恢复为空闲
[错误]      → ⚠ "重试" 红色边框脉冲

交互:
- 点击: WAAPI ripple 效果 (从点击位置扩散圆形光晕)
- hover: scale(1.03) + shadow-glow
- active: scale(0.97) spring
```

### 5.6 推理进度 SSE 事件

```
event: progress
data: {"stage":"loading_model","message":"正在加载 PatchCore...","pct":10}

event: progress
data: {"stage":"inference","message":"正在推理...","pct":60}

event: result
data: {"score":0.9234,"label":1,"heatmap_b64":"data:image/png;base64,...",
       "anomaly_map_b64":"data:image/png;base64,...","bboxes":[[x,y,w,h,score],...],
       "threshold":0.85,"confidence":0.9234,"image_b64":"data:image/png;base64,..."}

event: done
data: {}
```

### 5.7 结果面板

```
┌── 原图↔热力图 对比滑块 ──────────────┐
│  拖拽中线的 slider overlay             │
│  hover 热力图 → tooltip 显示异常得分    │
│  bbox overlay (红框, hover→蓝框+光晕)  │
│  右侧颜色比例尺 (滚动到位渐入)          │
└──────────────────────────────────────┘

┌── 指标卡片 ──┐  ┌── 判决卡片 ──┐
│ 异常得分       │  │ ● 异常        │
│ 0.9234 ↗     │  │ 0.9234 > 0.85 │
│ [进度条█████░]│  │ → 异常        │
│ 置信度        │  └──────────────┘
│ 98.7%        │
└──────────────┘

### 数字跳动

数字从 0 滚动到最终值, duration 600ms, ease-out-expo:
- 实现: 创建 10 个数字格 (0-9 纵向排列), 每个 digit 独立滚动到目标数字
- hover 数字时微微放大 (scale 1.05)
- 置信度百分比在整数位加空格分隔 (如 98.7%)
```

### 5.8 原图↔热力图对比滑块

```
┌─────────────────────────────┐
│ 原图        │  热力图        │
│             │               │
│        ◄──●──►              │  ← 拖拽中线
│             │               │
└─────────────────────────────┘

实现:
- 两层 <img> 叠加: 下层原图, 上层热力图 + CSS clip-path (inset)
- 中线 <input type="range"> 控制 clip-path 位置
- 拖拽手柄: 圆形 + 垂直线, 蓝色光晕
- 移动端支持 touch drag
```

### 5.9 四模型对比

```
┌────────────┬────────────┬────────────┬────────────┐
│ PatchCore  │ PaDiM      │ FRE        │ DRAEM      │
│ 🟢 正常    │ 🟢 正常    │ 🔴 异常    │ 🟢 正常    │
│ [热力图]   │ [热力图]   │ [热力图]   │ [热力图]   │
│ 得分 0.12  │ 得分 0.18  │ 得分 0.92  │ 得分 0.25  │
└────────────┴────────────┴────────────┴────────────┘

流程:
1. 初始: 四列显示 "等待推理…" 浅灰色
2. 点击对比: 四列同时变骨架屏
3. 逐个完成: patchcore → padim → fre → draem
   - 完成的列: 边框绿色闪一下 + 内容淡入
   - 未完成的列: 骨架屏 spinner
4. 全部完成: 顶部汇总条出现 "最佳: PatchCore (得分最低/最正常)"
```

---

## 6. 状态覆盖

| 状态 | 按钮 | 结果区 | 图片区 |
|------|------|--------|--------|
| **空闲** | "开始推理", 脉动微光 | 空, "上传图片开始" | 虚线框 + 浮动图标 |
| **上传完成** | 发光增强 | 空 | 实图, 边框实线过渡 |
| **加载模型** | spinner, 不可点击 | 骨架屏 shimmer | 不变 |
| **推理中** | spinner + "推理中…" | 骨架屏 + 进度条 | 不变 |
| **完成** | ✓ 2s 恢复 | 结果卡片动画序列 | 热力图注入 |
| **错误** | ⚠ "重试" 红色 | 错误卡片 (红色边框) | 不变 |

### 边界情况

| 场景 | 处理 |
|------|------|
| 快速切换算法 | AbortController 取消上次 SSE, 以最新为准 |
| 推理中切换页面 | 流继续, 结果缓存, 切回时直接展示 |
| 大图上传 (>20MB) | 前端 canvas 缩放到 max(1024px) |
| 后端未就绪 | 导航栏指示 "● 服务离线", 3s 自动重试 |
| 首次加载慢 | 全页遮罩 + Logo 脉冲, 与现有 page-loader 兼容 |
| 移动端触摸 | tooltip→tap, 对比滑块→touch drag |

---

## 7. 全局交互约束

| 规则 | 说明 |
|------|------|
| `user-select: none` | 所有交互控件（按钮、下拉、导航、开关）禁止文字选中 |
| 正文可选 | 描述文字、算法介绍、使用说明等正文保持 `user-select: text` |
| 下拉滚动 | 自定义滚动条 + `scroll-behavior: smooth` + Mac 惯性 |
| 键盘可达 | 所有交互元素支持 Tab/Enter/Escape |
| `prefers-reduced-motion` | 动效降级：减少 duration、禁用视差、保留功能 |
| `prefers-color-scheme` | 主题跟随系统，手动选择 (localStorage) 优先 |

---

## 8. 技术约束

| 项目 | 选型 |
|------|------|
| 前端框架 | Alpine.js (CDN, 无构建) |
| CSS | 手写, CSS 自定义属性与 Phase 1 变量体系兼容 |
| 动画 | WAAPI (Web Animations API) + CSS transition/keyframes |
| 后端 | FastAPI + sse-starlette |
| 流式 | Server-Sent Events (SSE) |
| 模型逻辑 | 复用 `modules/ui/demo.py` 的 `AnomalyDetector` 类 |
| 浏览器最低 | Chrome 100+, Edge 100+, Safari 16+, Firefox 120+ |

---

## 9. 实现阶段

| 阶段 | 内容 | 优先级 |
|------|------|--------|
| **P0** | `server.py` + `index.html` 骨架 + Alpine 全局状态 + 胶囊主题开关 + Logo 脉冲 | 🔴 核心 |
| **P1** | 单模型推理 (上传→SSE→结果→数字跳动→热力图→对比滑块) | 🔴 核心 |
| **P2** | 算法介绍页 + 四张 SVG 流程图 + stroke-dashoffset 绘制动画 | 🟡 高 |
| **P3** | 四模型对比 + 并行 SSE + 汇总条 | 🟡 高 |
| **P4** | 鼠标光晕跟随 + 滚动驱动动画 + 环境光呼吸 + 导航点 | 🟢 中 |
| **P5** | 过渡带设计 + sticky 上下文标题 + 全局文字选中控制 | 🟢 中 |
| **P6** | 移动端适配 + 边界情况打磨（AbortController/大图/重连） | 🔵 低 |
