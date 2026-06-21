# Phase 2 UI Apple 化重设计 — Cinematic Pro

- **日期**: 2026-06-20
- **状态**: 已确认，待实施
- **设计方向**: Cinematic Pro（电影感暗色产品页）

## 背景

当前 Phase 2 UI 已具备 FastAPI + Alpine.js SPA、scroll-snap 三页结构、亮/暗双主题、SSE 推理与四模型对比功能。用户希望进一步把界面优化为 **Apple 官网产品介绍页** 的质感：更大的字体、更精致的玻璃拟态、更有节奏感的动效，同时保留全部现有功能。

## 目标

在不改动后端 API 与 SSE 推理逻辑的前提下，对前端进行 Apple 风格的视觉与交互升级，使答辩/演示时更具产品感与专业度。

## 关键设计决策

| 决策 | 选择 | 原因 |
|------|------|------|
| 页面结构 | 保留三页 scroll-snap | 用户明确选择 A；改动最小，功能完整 |
| 默认主题 | 暗色（#0a0a0b），保留亮/暗切换 | 用户选择；更像 Apple Pro / WWDC 页面 |
| Hero 视觉 | 抽象检测管线 SVG + Bento 算法卡片 | 用户认为热力图不适合 Hero，流程图更好；方案 C 的混合式 |
| 功能页改造 | 平衡式：结构微调 + 视觉 Apple 化 | 用户选择 C；效果最好/风险比 |
| 字体 | 继续使用系统字体栈 | 符合 CLAUDE.md 约束，零外部字体依赖 |
| 动效引擎 | 继续由 JS WAAPI 驱动，配合 CSS 变量 | 避免与 scroll-snap 冲突，支持 reduced-motion |

## 范围

**修改文件**:
- `modules/ui/static/index.html`：调整 DOM 结构，新增 HeroVisual、Bento 卡片、玻璃仪表盘等
- `modules/ui/static/css/app.css`：重写视觉样式，统一玻璃拟态、字体层级、动效缓动
- `modules/ui/static/js/animations.js`：新增 Hero 管线动画、结果仪表盘揭示、对比卡片入场等
- `modules/ui/static/js/app.js`：少量适配（动画触发时机、DOM 选择器）
- `modules/ui/static/js/inference.js`、`compare.js`：仅做必要的 DOM 初始化调整

**不改动**:
- 后端 `modules/ui/server.py`
- SSE 推理协议与 API 路径
- Alpine.js 全局状态结构与字段名
- 系统字体栈
- 亮/暗主题切换机制

## 页面结构

### Section 0 — 首页 / 算法介绍

1. **电影感 Hero**
   - 居中 80px+ 超大标题「工业缺陷检测」，letter-spacing -0.04em
   - 副标题为磨砂玻璃胶囊：「无监督 · 像素级 · 实时推理」
   - Hero 视觉：抽象「检测管线」SVG 装置，循环播放（输入 → 特征 → 异常得分 → 热力图）
   - 底部滚动提示箭头
2. **Bento 算法卡片网格**
   - 2×2 玻璃卡片：PatchCore / PaDiM / FRE / DRAEM
   - 每张卡片左侧有色标竖线，hover 时色标发光
   - 卡片内嵌 mini 流程图 SVG，与当前流程图一致但更精致

### Section 1 — 单模型推理

1. 页面标题「开始检测」居中
2. **玻璃工作台**
   - 大 Drop Zone：虚线边框，拖拽时发光变实线
   - 模型选择：Apple 风格分段控件或精致自定义下拉
   - 居中大推理按钮，加载中显示进度条与微光扫过
3. **结果仪表盘卡片**（推理完成后出现）
   - 全宽玻璃卡片，顶部标题栏 + 状态徽章（正常/异常）
   - 全宽对比滑块（原图 vs 热力图）
   - 三列指标：异常得分 / 置信度 / 阈值 τ
   - 底部判决条 + 重新上传按钮

### Section 2 — 四模型对比

1. 页面标题「四模型对比」居中
2. 共享原图居中展示
3. 一键「四模型同时对比」按钮
4. 2×2 大幅热力图卡片墙
   - 每张顶部色标（PatchCore 蓝 / PaDiM 绿 / FRE 橙 / DRAEM 紫）
   - 热力图下方显示得分、置信度
5. 全部完成后出现排名摘要栏：最佳模型 + 1-4 名

## 视觉方向

- **默认暗色**:
  - 背景 `#0a0a0b`
  - 卡片背景 `rgba(255,255,255,0.06)` + `backdrop-filter: blur(20px)`
  - 边框 `rgba(255,255,255,0.10)`
  - 强调色 `#2997ff`，状态色 `#30d158` / `#ff453a` / `#ff9f0a`
- **亮色模式**: 通过 `html[data-theme="light"]` 覆盖变量，保持相同组件结构
- **字体**:
  - 标题：`-apple-system, 'SF Pro Display', 'PingFang SC', 'Microsoft YaHei', sans-serif`
  - 正文：`-apple-system, 'SF Pro Text', 'PingFang SC', 'Microsoft YaHei', sans-serif`
  - 数字：`'SF Mono', 'JetBrains Mono', 'Consolas', monospace`
- **背景装饰**:
  - 保留极淡的点阵纹理（opacity 0.12→0.08）
  - 环境呼吸光调弱 30%，避免干扰内容
  - 鼠标光晕保留但降低强度

## 动效设计

| 元素 | 动效 |
|------|------|
| Snap 页面切换 | 继续由 `Anim.snapPageEnter` / `Anim.snapPageExit` 驱动，统一向下淡入/淡出 |
| Hero 标题 | 进入时透明度 + 轻微上移，副标题延迟 120ms |
| Hero 管线 SVG | 路径 `stroke-dashoffset` 描边动画 + 节点脉冲，循环播放 |
| Bento 卡片 | stagger 入场（每张延迟 100ms），hover 上浮 4px + 色标发光 |
| Drop Zone | 拖拽时边框由虚线变实线 + 蓝色发光脉冲 |
| 进度条 | 微光扫过 + 宽度过渡 |
| 结果仪表盘 | 卡片淡入，指标条从 0 填充，得分数字滚动 |
| 四模型对比 | 卡片逐个翻转/淡入，最佳模型完成后高亮呼吸 |

**无障碍**: 所有动效在 `prefers-reduced-motion: reduce` 下瞬间完成，不触发眩晕。

## 组件清单

- **HeroVisual**：SVG 检测管线装置，纯 CSS/JS 动画，无需新依赖
- **BentoCard**：玻璃卡片，内含 mini flowchart，色标可配置
- **DropZone**：升级上传区，支持拖拽高亮与脉冲提示
- **SegmentedControl / CustomSelect**：模型选择器 Apple 化
- **ResultDashboard**：全宽玻璃结果卡片
- **CompareWall**：2×2 热力图卡片墙

## 数据流

1. 页面加载后 `Alpine.data('app')` 初始化主题、获取模型/数据集列表、启动健康检查
2. 用户上传图片 → `uploadedFile` / `uploadPreviewUrl` 更新 → 状态变为 `uploaded`
3. 用户选择模型并点击推理 → `InferenceRunner.run()` 通过 SSE 与 `/api/predict` 通信
4. SSE `onResult` 回调更新 `resultData` → Alpine 渲染 `ResultDashboard` → 触发 `Anim.resultReveal()`
5. 四模型对比通过 `Alpine.data('compare')` 调用 `/api/compare` SSE → 更新 `compareSlots` / `summary` → 触发 `Anim.compareReveal()`
6. 主题切换继续通过 `localStorage` + `matchMedia` 管理

## 错误处理与兼容性

- 保留现有 `error-card` 与 `toast` 通知系统
- 动画函数在目标元素不存在时静默返回，不阻塞业务逻辑
- 尊重 `prefers-reduced-motion`
- 移动端响应式保留：
  - Hero 标题缩小
  - Bento 卡片单列
  - 对比网格 2×2 / 单列
  - 进度环导航点移至底部
- 继续关注 `.compare-heatmap` position 泄漏问题，确保 `.compare-slot .compare-heatmap` 为 `relative`

## 测试计划

1. 启动 FastAPI UI：`python scripts/run_ui.py`
2. 手动验证：
   - 三页 scroll-snap 滚动与导航点
   - 亮/暗主题切换
   - 上传图片 → 单模型推理 → 结果仪表盘渲染
   - 四模型对比 → 2×2 热力图墙 → 排名摘要
3. 浏览器：Chrome / Edge 桌面端 + 移动端模拟器
4. 回归测试：`python -m pytest tests/ -v`

## 风险与注意事项

- **CSS 陷阱**: `.pipeline` 仍为精确三列 grid，禁止添加额外子元素；连接线使用 `::after`
- **CSS 陷阱**: `.compare-heatmap` 选择器同时用于单模型滑块（absolute）与四模型槽位（relative），修改时必须验证两种场景
- **性能**: 大量 backdrop-filter 与阴影在低端设备可能掉帧，已在移动端简化
- **时间**: 主要为前端工作，预计对现有功能无回归风险

## 下一步

由 `writing-plans` skill 制定详细实施计划（文件修改顺序、动画实现细节、验证步骤）。
