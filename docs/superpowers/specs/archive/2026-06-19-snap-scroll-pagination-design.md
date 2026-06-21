# 全屏 Snap 滚动 + Apple 风格切页动画 — 设计规范

**日期**: 2026-06-19
**范围**: `modules/ui/` — FastAPI + Alpine.js SPA
**目标**: 将三段连续滚动改为全屏 Snap 吸附滚动，增加 Apple 产品页级别的切页动画和视觉连续性

---

## 1. 架构变更

### 当前结构

```
<body> → 普通自由滚动
  nav.navbar (sticky)
  section#section-0 (算法介绍)
  div.transition-zone
  section#section-1 (单模型推理)
  div.transition-zone
  section#section-2 (四模型对比)
  footer
  nav-dots (fixed, 右侧)
```

### 目标结构

```
<body>
  div.snap-container      ← scroll-snap-type: y mandatory; scroll-snap-stop: always
    section#s0.snap-page  ← 100dvh, scroll-snap-align: start
    section#s1.snap-page
    section#s2.snap-page  ← footer 移入此 section 底部
  nav.navbar              ← position: fixed (不再 sticky)
  div.nav-dots            ← 重构为进度环 pill
```

### 关键变化

| 项 | 之前 | 之后 |
|---|------|------|
| 滚动模型 | 自由滚动 | CSS scroll-snap 强制吸附到整页 |
| Section 高度 | 内容自适应 | 100dvh（全视口） |
| 过渡区 | 独立 div.transition-zone | 移除——吸附切换本身即是过渡 |
| 导航栏 | position: sticky | position: fixed，始终在最上层 |
| Footer | 独立在 section 之外 | 嵌入 S2 底部 |
| 右侧导航点 | 圆点 active/inactive | 进度环 pill，显示当前页/总页数 |

---

## 2. 各 Section 设计

### 2.1 Section 0 — 算法介绍

**标题**: "工业缺陷检测"（从"工业异常"改为"缺陷"，更贴合领域）
**副标题**: "四算法 · 无监督 · 像素级定位"

**2×2 算法卡片**（保留网格，重新设计内部）：
- 每张卡片左侧彩色竖线（算法标识色）
- 图标替代纯文字标签：PatchCore→🔍, PaDiM→📊, FRE→🔄, DRAEM→🎯
- 底部短标签：参数规模 + 一句话小结
- Hover：整卡上浮 2px + 彩色竖线发光

**算法标识色**：
| 算法 | 色值 | 含义 |
|------|------|------|
| PatchCore | `#2997ff` (蓝) | 首选 |
| PaDiM | `#30d158` (绿) | 轻量 |
| FRE | `#ff9f0a` (橙) | 备选 |
| DRAEM | `#bf5af2` (紫) | 备选 |

**底部提示**: 首次访问显示向下箭头动画 → 3 秒后淡出

### 2.2 Section 1 — 单模型推理

**三列流水线布局**（从左到右，水平排列）：
```
[ 上传图片 ] → [ 选择模型 ] → [ 开始推理 ]
```
- 视觉暗示"流程"：步骤 1→2→3
- 已上传后第一列变为缩略图预览
- 推理中：推理按钮内圈旋转（进度环替代进度条）
- 错误状态：对应节点变红，错误信息下方滑入

**结果区**（完成后显示）：
- 左右分栏：左侧图片对比滑块，右侧指标卡
- 指标卡紧凑排列：得分 / 置信度 / 阈值 → 判决

### 2.3 Section 2 — 四模型对比

**原图去重**：
- 四列网格之上放一张共享原图（因为四个模型输入相同）
- 每列只显示热力图（这是对比的核心）

**每列结构**：
```
[算法名 + 色标竖线]
[热力图]
[得分 + 置信度 + 阈值]
[○正常 / ●异常 徽章]
```

**摘要栏**：横向展开，`#1 PaDiM → #2 PatchCore → #3 FRE → #4 DRAEM`

**响应式**：
- 平板 (≤768px): 2×2 网格
- 手机 (≤480px): 单列

---

## 3. 切页动画 & 视觉连续性

### 技术方案

**主方案（Chrome 115+）**:
- CSS `scroll-snap-type: y mandatory` 原生吸附
- `ViewTimeline` (CSS `animation-timeline: view()`) 驱动进出动画
- Web Animation API 编排跨 section 接力过渡

**降级方案（Safari / Firefox）**:
- CSS scroll-snap 仍然工作（Safari 15.4+, Firefox 99+）
- `IntersectionObserver` + CSS class 切换替代 ViewTimeline
- 功能相同，时机粒度从连续降为 5 档阈值

### 阶段化动画

| 阶段 | 触发条件 | 动画 |
|------|---------|------|
| 离开 S0 | S0 在视口中 < 90% | Hero 标题 opacity→0 + scale→0.95 |
| 进入 S1 | S1 进入视口 > 20% | 标题淡入，三列流水线逐列弹出 (delay: 0→80→160ms) |
| S0↔S1 中间 | 滚动在分界附近 | 两页内容交叉淡入淡出 |
| 离开 S1 | S1 离开视口 | 结果面板淡出 |
| 进入 S2 | S2 进入视口 > 20% | 标题淡入，四列卡片逐列弹出 (delay: 0→100→200→300ms) |

### 接力过渡元素

1. **页标题接力**: SX 标题淡出 ↔ SX+1 标题淡入，交叉淡入淡出
2. **导航栏标题**: 离开 S0 时，导航栏 Logo 旁出现当前 section 的小标题
3. **导航点 → 进度环**: 右侧导航点从简单圆点变为带填充进度的环，指示在两 section 之间的位置

### 动效参数

| 属性 | 值 |
|------|-----|
| 吸附过渡 | `scroll-behavior: smooth`, ~600ms 吸附 |
| 内容入场 | `opacity 0→1`, `translateY(24px→0)`, `duration: 500ms`, `ease: cubic-bezier(0.16, 1, 0.3, 1)` |
| 内容出场 | `opacity 1→0`, `translateY(0→-16px)`, `duration: 350ms`, `ease: cubic-bezier(0, 0, 0.2, 1)` |
| 标题接力 | `opacity` 交叉淡出，`duration: 400ms` |
| 卡片逐级延迟 | 每级 +80ms (S1), +100ms (S2) |

---

## 4. 细节打磨 — 排布优化

### 4.1 Section 0
- Hero 标题从 56px → 64px（全屏空间更多）
- 卡片 grid-gap 从 24px → 32px（更多呼吸感）
- 卡片内 padding 从 32px → 36px
- 算法 SVG 流程图从固定 `viewBox="0 0 420 270"` 缩放适配

### 4.2 Section 1
- 三列等宽 `grid-template-columns: 1fr 1fr 1fr`，gap: 20px
- 上传区 min-height 从 240px → 200px（三列水平后不需那么高）
- 结果区左右 6:4 分栏
- 指标字体从 28px → 32px

### 4.3 Section 2
- 四列 gap 从 16px → 20px
- 共享原图 max-height: 240px
- 热力图 max-height: 180px
- 指标字体从 18px → 16px（四列空间紧，缩小微调）

### 4.4 全局
- Section 内 padding: `max(48px, 6vh) 32px`，自适应视口高度
- 导航栏高度: 52px → 48px（fixed 后更轻薄）
- 字体基准从 15px 保持不变
- `scrollbar-width: none` 在 snap 容器上隐藏滚动条（吸附页面不需要）

---

## 5. 技术约束 & 兼容性

### 必须保持的
- FastAPI 后端不变（`server.py` 无需改动）
- Alpine.js 3.14.9 继续使用，不做框架替换
- 所有现有 API 端点不变
- 亮/暗双模式 CSS 变量体系不变
- 现有 JS 模块（inference.js / compare.js / animations.js / cursor-glow.js / flowchart.js）接口不变

### 需要修改的文件
| 文件 | 改动 |
|------|------|
| `index.html` | 结构重写：snap-container + snap-page + 每 section 内容新布局 |
| `app.css` | 大幅改写：增加 snap 容器样式、重写 section 布局、新增进出动画 |
| `app.js` | 中度修改：导航逻辑从 IntersectionObserver 改为 snap 事件监听 + 进度环更新 |
| `animations.js` | 小幅修改：增加 ViewTimeline 驱动动画（主方案）+ Observer 降级 |
| `flowchart.css` | 小幅修改：SVG 自适应缩放 |
| `theme.py` | 无需改动 |
| `server.py` | 无需改动 |

### 不修改的文件
- `inference.js` / `compare.js` — 功能不变，只改 HTML 模板中的 Alpine 绑定
- `cursor-glow.js` — 保持
- `flowchart.js` — 保持

### 浏览器兼容
| 特性 | Chrome | Safari | Firefox |
|------|--------|--------|---------|
| scroll-snap | ✅ 69+ | ✅ 15.4+ | ✅ 99+ |
| ViewTimeline | ✅ 115+ | ❌ | ❌ |
| IntersectionObserver | ✅ 51+ | ✅ 12.1+ | ✅ 55+ |
| WAAPI | ✅ 67+ | ✅ 13.1+ | ✅ 63+ |

---

## 6. 设计审阅

### 自检

- [x] 无 TODO/TBD/placeholder
- [x] 内部一致：各 section 设计语言统一，色标体系贯穿全文
- [x] 范围聚焦：仅涉及前端 UI 层，不碰后端 API 和推理逻辑
- [x] 无歧义：所有尺寸、颜色、动画参数明确
- [x] 降级方案完整：Safari/Firefox 功能等价
