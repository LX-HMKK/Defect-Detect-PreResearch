# UI 布局精修设计规范 — S1/S2/S3 三页改进

**日期**: 2026-06-19  
**状态**: 设计完成，待用户审核  
**范围**: `modules/ui/static/` — index.html, app.css, app.js, animations.js

---

## 1. 问题回顾

| # | 页面 | 问题 | 根因 |
|---|------|------|------|
| 1 | S0↔S1↔S2 进出 | 衔接不流畅 | 离开动画向上(-20px)、进入动画从下方(+24px)，方向冲突；CSS/JS 两套动画竞争 |
| 2 | S1 推理结果 | 布局零散 | 6:4 左右分栏，流水线与结果互斥切换，元素间缺乏统一容器 |
| 3 | S3 多模型对比 | 共享原图太小、四列比例失调 | 内联样式 `height:64px` 覆盖 CSS `max-height:240px`；槽位 min-height:320px 空置 |

---

## 2. 方案 1 — S1 进出衔接：统一滚动方向

### 2.1 设计思路

将离开和进入动画统一为**向下位移**（模拟滚动推送的物理感）：

- **离开页（S_old）**：内容向下推出 30px + 透明度 1→0 + 微缩放 1→0.97，持续 350ms
- **进入页（S_new）**：内容从下方 30px 推入 → 原位 + 透明度 0→1，子元素 stagger 80ms，持续 500ms

### 2.2 动画参数

| 属性 | 离开动画 | 进入动画 |
|------|---------|---------|
| 位移 | translateY(0 → +30px) | translateY(+30px → 0) |
| 透明度 | 1 → 0 | 0 → 1 |
| 缩放 | 1 → 0.97 | 1（不变） |
| 时长 | 350ms | 500ms |
| 缓动 | cubic-bezier(0, 0, 0.2, 1) | cubic-bezier(0.16, 1, 0.3, 1) |
| 子元素延迟 | 0s / 0.04s / 0.08s / 0.12s | stagger 80ms |

### 2.3 实现变更

**animations.js**:
- `snapPageExit()`: 方向改为 `translateY(+30px)`（向下推出）
- `snapPageEnter()`: 保持 `translateY(24px → 0)`（从下方进入），stagger 从 100ms 改为 80ms
- 移除 Chrome 115+ `@supports (animation-timeline: view())` 块中的冲突动画（或加 `/* 暂禁用 */`）

**app.css**:
- **删除** `.snap-page--exiting .snap-page-inner > *` 的 CSS animation 规则（第 759-765 行）—— 退出动画完全由 JS `snapPageExit()` 驱动，消除 CSS/JS 双动画竞争
- **删除** `@keyframes pageContentExit`（第 767-770 行）—— 不再被引用
- `.snap-page--exiting .hero-title`: 保持独立处理（Hero 大标题场景特殊），方向同步改为 `translateY(20px)`
- `@supports (animation-timeline: view())` 块（第 2522-2533 行）：添加 `/* 暂禁用：与 JS scroll-snap 编排冲突 */` 注释，不删除以便后续 Chrome 115+ 原生方案成熟后迁移

**app.js**:
- IntersectionObserver 回调中：先触发 JS `snapPageExit()` 再切换 section（当前是同时触发，exit 动画可能被打断）
- 进入新 section 时确保先移除 `snap-page--exiting` class
- 回滚到之前的 section 时，重新触发 `snapPageEnter()`

### 2.4 `snap-page--exiting` class 生命周期

当前问题：class 被 `IntersectionObserver` 回调添加后，只在 `snapPageEnter()` 中移除。如果用户快速来回滚动，class 可能残留。

新规则：
1. **添加时机**：`IntersectionObserver` 检测到 `currentSection` 即将切换时，给旧 section 加 class
2. **移除时机**：class 对应的 CSS 退出动画完成后（`animationend` 事件），或新 section 的进入动画开始时（`snapPageEnter` 首行），取先到者
3. **防止残留**：`snapPageEnter()` 首行调用 `section.classList.remove('snap-page--exiting')`，同时清除该 section 下所有子元素的内联动画样式（`getAnimations().forEach(a => a.cancel())`）

### 2.5 关键约束

- 快滚（连续滚两页）时：新动画启动前必须 `cancel()` 中间 section 的旧动画
- `prefers-reduced-motion` 时：跳过所有动画，直接显示最终状态
- 键盘 ↑↓ 导航：使用相同的 enter/exit 逻辑（`scrollToSection` → 自然触发 snap → Observer 回调）

---

## 3. 方案 2 — S2 推理结果：一体化仪表盘卡片

### 3.1 设计思路

将当前「左滑块 + 右指标」的松散布局改为**单张一体化卡片**，从上到下：

```
┌─────────────────────────────────────────┐
│ 推理结果 · PatchCore · bottle    [异常] │  ← 标题栏
├─────────────────────────────────────────┤
│                                         │
│    原图 / 热力图 对比滑块 (全宽)         │  ← 对比区
│    [━━━━━━━●━━━━━━━━]                   │
│                                         │
├─────────────────────────────────────────┤
│  异常得分    │  置信度    │  阈值 τ      │  ← 三列指标行
│  0.8921      │  94.2%    │  0.423       │
│  ████████░░  │  ████████░ │  Youden's J  │
├─────────────────────────────────────────┤
│ ● 得分 0.8921 > τ 0.423 → 异常  [重新上传] │  ← 判决 + 操作
└─────────────────────────────────────────┘
```

### 3.2 布局规范

| 区域 | 宽度 | 说明 |
|------|------|------|
| 标题栏 | 100% | 模型名 + 数据集 + 异常/正常徽章，flex space-between |
| 对比滑块 | 100% | 原图/热力图叠加，保持现有效果，object-fit: contain |
| 指标行 | 3 列等宽 grid | 得分(带进度条) / 置信度(带进度条) / 阈值 |
| 判决条 | 100% | 判决文本 + 重置按钮，flex space-between |
| 卡片整体 | max-width: 780px 居中 | 圆角 16px，border + 微阴影 |

### 3.3 流水线→结果过渡

- 推理完成后，流水线**不隐藏**（移除 `x-show` 切换），改为收缩为一行紧凑的「步骤摘要」：
  ```html
  <div class="pipeline-summary" x-show="inferenceState === 'done'">
    <span class="pipeline-summary-step pipeline-summary-step--done">● 已上传</span>
    <span class="pipeline-summary-arrow">→</span>
    <span class="pipeline-summary-step pipeline-summary-step--done">● PatchCore</span>
    <span class="pipeline-summary-arrow">→</span>
    <span class="pipeline-summary-step pipeline-summary-step--done">● 推理完成 ✓</span>
  </div>
  ```
  - `.pipeline-summary`: `display: flex; align-items: center; justify-content: center; gap: 10px; padding: 12px 0;`
  - 每个步骤为圆形小点 + 文字，完成态为 `var(--ok)` 绿色
  - 与原 `.pipeline` 三列 grid 互斥显示：idle/uploaded/error → 显示 pipeline，done → 显示 pipeline-summary
- 结果卡片（`.result-dashboard`）从摘要行下方以 `scroll-reveal` + `fadeInUp` 升入

### 3.4 实现变更

**index.html**:
- `.result-layout` (grid 6:4) → 替换为 `.result-dashboard` (单列 flex)
- 指标从 `.result-right` 独立卡片 → 移入 `.result-dashboard` 内的三列 grid
- 判决条从 `.result-verdict` → 移入仪表盘底部
- 对比滑块从左列 → 移入仪表盘中间全宽
- 上传区/模型选择保持现有 `.pipeline` 三列不变
- **新增**：流水线收缩摘要行 `.pipeline-summary`

**app.css**:
- 新增 `.result-dashboard` 样式（单卡片容器，max-width 780px，居中）
- 新增 `.result-metrics-row`（三列等宽 grid）
- 修改 `.result-verdict` 为卡片底部内联条
- 新增 `.pipeline-summary`（收缩后的步骤摘要）
- 保留现有 `.pipeline` / `.pipeline-step` 不变

### 3.5 关键约束

- 移动端 (<768px)：指标行从 3 列变为堆叠，滑块高度减小
- `resultData` Alpine 数据结构不变，仅改 DOM 模板
- 图例从滑块下方移入滑块 overlay 左下角（半透明背景）

---

## 4. 方案 3 — S3 多模型对比：中型共享图 + 紧凑四列

### 4.1 设计思路

修复 64px 缩略图 bug，重调四列比例，让热力图成为每列的视觉重心。

### 4.2 共享原图区域

```
┌──────────────────────────────────────┐
│                                      │
│         [输入图像 · 280~512px]        │
│         共享原图 · 四种算法同一输入     │
│                                      │
└──────────────────────────────────────┘
```

| 属性 | 当前值 | 新值 |
|------|--------|------|
| 图片高度 | `height: 64px` (内联) | `max-height: 200px` (CSS) |
| 图片宽度 | `width: auto` | `width: 100%`, `object-fit: contain` |
| 容器最大宽度 | 640px | 480px（更紧凑） |
| 容器布局 | flex row + gap | flex column 居中 |
| 容器边距 | `margin-bottom: 20px` | `margin-bottom: 20px` |

### 4.3 四列槽位

```
┌────────────┬────────────┬────────────┬────────────┐
│ PatchCore  │   PaDiM    │    FRE     │   DRAEM    │
│ ▔▔▔ 蓝色   │ ▔▔▔ 绿色   │ ▔▔▔ 橙色   │ ▔▔▔ 紫色   │
│            │            │            │            │
│ [热力图]   │ [热力图]   │ [热力图]   │ [热力图]   │
│  180px     │  180px     │  180px     │  180px     │
│            │            │            │            │
│ 得分 .8921 │ 得分 .8234 │ 得分 .7891 │ 得分 .6543 │
│ 置信度 94% │ 置信度 89% │ 置信度 82% │ 置信度 71% │
└────────────┴────────────┴────────────┴────────────┘
```

| 属性 | 当前值 | 新值 |
|------|--------|------|
| 色标位置 | 左侧竖线 3px (absolute) | **顶部横线 3px**（更醒目） |
| 槽位 min-height | 320px | **auto**（内容撑高） |
| 热力图 max-height | 180px | 180px（保持不变） |
| 槽位 padding | 20px | 16px |
| 槽位 hover | translateY(-4px) scale(1.02) | translateY(-2px)（减弱，避免四列同时跳动） |
| 指标行数 | 3 行（得分/置信度/阈值） | 2 行（得分 + 置信度，阈值移入 header badge 或省略） |

### 4.4 摘要栏

```
🏆 最佳：PatchCore 0.8921  |  #2 PaDiM 0.8234  |  #3 FRE 0.7891  |  #4 DRAEM 0.6543
```

- 单行 flex row，居中对齐
- 共享原图下方、四列网格上方
- `background: var(--bg-secondary)`, `border-radius: 10px`, `padding: 10px 20px`

### 4.5 实现变更

**index.html**:
- 移除内联 `<style>` 中 `.compare-shared-image img { height: 64px; }` 规则
- 移除 `.compare-shared-image` 的 flex row 内联样式（改为 CSS 中的 column 布局）
- `.compare-slot-accent` 从 `position: absolute; left: 0; width: 3px; height: 100%` → `width: 100%; height: 3px; position: static`
- 每个槽位指标从 3 行 → 2 行（得分 + 置信度）

**app.css**:
- `.compare-shared-image`: 重写为 column 布局，图片 `max-height: 200px`
- `.compare-shared-img`: 修复 `width: 100%; max-height: 200px; object-fit: contain`
- `.compare-slot-accent`: 从左侧竖线改为顶部横线
- `.compare-slot`: `min-height: auto`，`padding: 16px`
- `.compare-slot:hover`: `transform: translateY(-2px)`（无 scale）
- `.compare-slot .compare-heatmap`: `max-height: 180px`（保持）
- 新增 `.compare-summary-row`（压缩摘要栏）

---

## 5. 文件变更清单

| 文件 | 变更类型 | 涉及内容 |
|------|---------|---------|
| `index.html` | 修改 | S1: 移除竞争动画触发；S2: 重构结果面板 DOM；S3: 修复 64px + 色标方向 + 摘要栏 |
| `app.css` | 修改 | 进出动画参数、`.result-dashboard` 新组件、`.compare-shared-image` 重写、`.compare-slot` 重调 |
| `animations.js` | 修改 | snapPageExit 方向统一、snapPageEnter stagger 调整、移除 view-timeline 冲突 |
| `app.js` | 微调 | IntersectionObserver 回调中 exit/enter 时序 |

---

## 6. 不变项

- Alpine.js 数据模型（`resultData`、`compareSlots`、`inferenceState` 等）结构不变
- SSE 推理/对比流程不变
- 主题系统（亮/暗双模式）不变
- 导航栏、进度环、页脚不变
- 响应式断点（768px / 480px）策略不变
- 流水线三列 grid 结构不变
- **`.compare-heatmap` 双用法不变**：S2 单模型滑块中保持 `position: absolute`，S3 四列槽位中保持 `position: relative`，两处选择器隔离规则不变

---

## 7. 验收标准

1. **S1 进出**：上下滚动时，页面内容沿滚动方向平滑推移，无跳跃、无闪烁
2. **S1 回滚**：向上滚动回到旧页面时，内容正确重新入场（不残留 opacity:0）
3. **S2 结果**：推理完成后，结果以单张卡片呈现，各元素视觉关联清晰
4. **S2 过渡**：流水线收缩 + 结果卡片升入，整体流畅
5. **S3 原图**：共享原图以合适尺寸 (160-200px) 居中显示，无大量留白
6. **S3 四列**：热力图在所有列中视觉占比最大，色标醒目，hover 反馈克制
7. **亮/暗模式**：所有新样式在双模式下视觉正确
8. **移动端**：平板 (≤768px) 和手机 (≤480px) 下布局不崩
9. **reduced-motion**：`prefers-reduced-motion: reduce` 时动画跳过
