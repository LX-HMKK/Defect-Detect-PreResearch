# AirPods Pro 风格 UI 重设计方向

## 当前状态诊断

当前 UI 已实现：
- FastAPI + Alpine.js SPA 四页 scroll-snap 全屏吸附
- 玻璃质感卡片、药丸按钮、进度环导航
- 亮/暗双模式、系统字体栈、细微动效
- 单模型推理仪表盘、Training Studio、四模型对比

与 https://www.apple.com/airpods-pro/ 的差距主要在于：
1. **默认暗色** vs. AirPods Pro 的**明亮自信主调**
2. **全屏强制 snap** vs. AirPods Pro 的**模块化堆叠、高度自由**
3. **控件密度高**（流水线、下拉框、进度环）vs. AirPods Pro 的**大面积留白、产品本身成为视觉主体**
4. **装饰性动画**（光标光晕、背景噪点）vs. AirPods Pro 的**叙事性滚动动效**（startframe/endframe、波浪脉冲）

---

## 设计概念：Lab Precision / Cinematic Clarity

把「缺陷热力图」当作 AirPods Pro 里的产品本身来呈现：
- 工业检测 = 精密、可信、果断
- AirPods Pro = 自信、简洁、产品叙事
- 结合为：用产品摄影般的自信，展示检测结果的精密感

---

## 核心修改方向

### 1. 色彩系统：从「暗色玻璃」转向「明亮自信」

AirPods Pro 页面以 `#ffffff` / `#f5f5f7` 为主，偶尔插入深色章节制造节奏。

修改建议：
- 将默认主题改为 **light**，暗色作为训练/对比章节的有意强调
- 主背景：`--bg-root: #f5f5f7`
- 卡片背景：纯白 `#ffffff` + 极淡阴影，替代玻璃模糊
- 分隔：用 1px `rgba(0,0,0,0.08)` 细线，替代半透明边框
- 强调色保留 `#2997ff`（Apple 蓝），但减少发光强度，更克制
- 新增「章节节奏色」：
  - Hero / 推理：白色
  - Training Studio：深色沉浸（类比 AirPods Pro 的 Intelligent noise control 黑场）
  - 四模型对比：浅灰画廊

涉及文件：
- `modules/ui/static/css/app.css` `:root` 与 `html[data-theme="light"]`
- `modules/ui/static/css/apple-redesign.css` 覆盖层
- `modules/ui/static/theme.js` 默认主题初始化

### 2. 字体排印：更大胆、更有叙事感

AirPods Pro Hero 标题特点：
- 极大字号（桌面约 80-120px）
- 行高紧（0.9-0.95）
- 字母间距紧（-0.04em 到 -0.06em）
- 多行断句制造呼吸感

修改建议：
- Hero 标题改为多行断句：
  ```
  工业级
  无监督
  缺陷检测
  ```
- 字号从 88px 提升到 **96-112px**，行高 0.92
- 副标题用 21px，字重 400，颜色 `--text-secondary`
- Section 标题改为陈述式短句，例如：
  - "一次推理，定位缺陷。"
  - "四款算法，同台竞技。"
- 中文 tracking 不宜过紧，使用 `-0.02em`

涉及文件：
- `modules/ui/static/css/app.css` 字体相关类
- `modules/ui/static/css/apple-redesign.css` Hero 覆盖
- `modules/ui/static/index.html` 标题文案

### 3. 布局：从「四页 snap」转向「模块化堆叠」

AirPods Pro 不是每屏等高的 snap 页面，而是：
- 首屏全高 Hero
- 后续章节高度自由（可 70vh、100vh、auto）
- 模块之间用大量留白和细线分隔

修改建议：
- **保留** Hero 全屏吸附（最具冲击力）
- **放松** 后续章节的 scroll-snap，允许自然滚动
- 单模型推理章节：改为「工作台」式布局，而非三列流水线
- Training Studio：深色沉浸全屏，左右分栏更舒展
- 四模型对比：画廊式网格，减少边框，用间距组织

涉及文件：
- `modules/ui/static/css/app.css` `.snap-container`、`.snap-page`
- `modules/ui/static/js/app.js` snap 滚动逻辑（降低强制吸附）
- `modules/ui/static/index.html` 各 section 结构

### 4. Hero：把「热力图」当作产品主角

AirPods Pro Hero 是产品居中、文字环绕、背景纯净。

修改建议：
- Hero 中央放置一张**放大的缺陷检测热力图/产品图**（可预置 demo 图或动态 SVG）
- 图像下方有柔和投影 + 轻微悬浮动画（translateY 循环 8px）
- 背景使用极淡的径向渐变光晕（白色中心 → 浅灰边缘）
- 标题从上方淡入，产品图从下方淡入，形成「文字 / 产品」的垂直叙事
- 移除当前 Hero 中的 SVG 流程图，移到第二屏「How it works」轻量展示

涉及文件：
- `modules/ui/static/index.html` Hero 区域
- `modules/ui/static/css/apple-redesign.css` Hero 样式
- `modules/ui/static/js/animations.js` Hero 入场动画

### 5. 组件形态：更少 chrome，更多留白

AirPods Pro 的控件特点：
- 按钮是**黑色填充药丸**或**纯文字链接**
- 卡片几乎无边框，靠阴影/背景区分
- 下拉框、步骤数字等 UI chrome 高度克制

修改建议：
- **按钮**：
  - 主 CTA：黑色填充 `#1d1d1f` + 白色文字，圆角 100px
  - 次按钮：透明 + 黑色文字 + 细边框
  - 当前蓝色渐变按钮改为仅在深色章节使用
- **卡片**：
  - 移除玻璃模糊，改为纯白/浅灰 + 1px 细线 + 微阴影
  - hover 时轻微上浮（translateY -4px）+ 阴影加深
- **下拉框**：
  - 简化为更轻量的触发器，减少装饰
- **进度指示**：
  - 进度条变细（2px），颜色用黑色或强调色
  - 考虑移除右侧进度环，改为底部极简页码或滚动指示

涉及文件：
- `modules/ui/static/css/app.css` 按钮、卡片、下拉框
- `modules/ui/static/css/apple-redesign.css` 覆盖
- `modules/ui/static/index.html` 按钮类名

### 6. 动效：从「环境装饰」转向「叙事滚动」

AirPods Pro 的动效服务于叙事：
- 标题逐行 stagger 出现
- 产品图从模糊/缩小到清晰
- 滚动触发 startframe → endframe 变化
- 章节切换时文字从底部滑入

修改建议：
- **Hero 入场**：
  - 标题三行依次出现，delay 0.1s/0.2s/0.3s
  - 产品图 scale 0.92 → 1.0 + opacity 0 → 1，duration 1.2s
- **章节进入**：
  - 标题从 `translateY(40px)` 滑入
  - 内容从 `translateY(60px)` 滑入，stagger 0.08s
- **滚动触发**：
  - 使用 IntersectionObserver + CSS 自定义属性（或 WAAPI）
  - 为推理结果图添加「扫描线」脉冲动画（类比 AirPods 的声波可视化）
- **减少/移除**：
  - 鼠标光晕（与 AirPods Pro 的克制风格冲突）
  - 背景噪点纹理（过于装饰）
  - 进度环（可替换为底部细线）

涉及文件：
- `modules/ui/static/js/animations.js`
- `modules/ui/static/js/app.js` IntersectionObserver
- `modules/ui/static/js/cursor-glow.js`（考虑移除或弱化）
- `modules/ui/static/css/app.css` 动效相关

### 7. 背景与氛围：干净 + 有深度的光

AirPods Pro 背景几乎是纯色，但通过产品阴影和极少的光晕营造深度。

修改建议：
- 主背景：干净的 `#f5f5f7` 或 `#ffffff`
- Hero：径向渐变光晕（中心亮、边缘暗）+ 产品下方柔和漫射阴影
- 深色章节（Training Studio）：极深灰 `#0d0d0f` + 顶部微弱环境光
- 移除全局噪点纹理
- 可添加极淡的网格点背景（1-2px 点，rgba 0.04）作为「实验室」暗示

涉及文件：
- `modules/ui/static/css/app.css` body / `.snap-container` 背景
- `modules/ui/static/css/apple-redesign.css`

### 8. 单模型推理：从「三列流水线」到「工作台」

当前三列流水线信息密度高。AirPods Pro 式的处理：
- 顶部一行：算法选择、数据来源、测试图片（轻量 pill 选择器）
- 中央大面积：结果图（产品主角）
- 底部：极简指标行

修改建议：
- 将三个步骤压缩到顶部一条工具栏
- 使用图标 + 药丸标签表示当前选择
- 结果图占据 60-70% 视觉权重
- 指标用大字 + 小标签，横向排列，去除进度条装饰
- 「异常/正常」判定做成大色块 badge，类似 AirPods 的 feature highlight

涉及文件：
- `modules/ui/static/index.html` 单模型推理 section
- `modules/ui/static/css/apple-redesign.css` 工作台样式
- `modules/ui/static/js/inference.js`（交互逻辑可保留，DOM 结构微调）

### 9. Training Studio：深色沉浸章节

AirPods Pro 会在关键特性处切到深色背景，制造节奏。

修改建议：
- Training Studio 整体使用深色（`#0a0a0b`）
- 左侧样本画廊、右侧监控面板用深色玻璃卡片
- Loss 曲线使用发光效果（`#2997ff` 带光晕）
- 训练状态用脉冲圆点（类似 AirPods 充电指示灯）
- 参数输入框用深色背景 + 细边框，focus 时边框高亮

涉及文件：
- `modules/ui/static/index.html` Training Studio section
- `modules/ui/static/css/apple-redesign.css`
- `modules/ui/static/js/training.js`（逻辑基本不变）

### 10. 四模型对比：画廊式并排

AirPods Pro 的对比展示通常是干净的多列布局。

修改建议：
- 共享原图放在顶部中央，像产品主图
- 四张结果图以等宽画廊形式排列
- 每个槽位仅保留：模型名、得分、小缩略热力图
- 去除多余边框，用间距区分
- hover 时该槽位轻微放大（1.02）并提升阴影

涉及文件：
- `modules/ui/static/index.html` 四模型对比 section
- `modules/ui/static/css/app.css` `.compare-grid`
- `modules/ui/static/js/compare.js`

---

## 实施优先级

### P0（立即改变整体气质）
1. 默认主题改为 light，调整 `:root` 色板
2. Hero 重构：大标题断句 + 产品图主角 + 简化背景
3. 按钮/卡片改为 Apple 式药丸 + 白底微阴影
4. 移除或弱化鼠标光晕、背景噪点

### P1（提升节奏与叙事）
5. 放松 scroll-snap，Training Studio 设为深色沉浸章节
6. 单模型推理工作台化（顶部工具栏 + 大图主角）
7. Hero 入场动画 stagger + 产品图悬浮

### P2（精致化）
8. 四模型对比画廊化
9. 进度环替换为底部滚动指示
10. 全站微交互统一（hover、focus、loading）

---

## 风险与注意事项

- **scroll-snap 放松后**，需同步更新 `app.js` 中的 `currentSection` 计算逻辑，确保导航和页码仍准确
- **默认 light 主题** 会影响 favicon、logo SVG 的可见性，需准备适配版本
- **移除玻璃模糊** 后，深色章节（Training Studio）需单独覆盖样式，否则文字对比度不足
- **Hero 产品图** 如使用真实检测结果，需确保在 Windows / 无 GPU 环境也有 demo 图可展示
- 当前测试依赖 UI 静态结构，修改 HTML 时需同步更新 `tests/test_ui_static.py`

---

## 参考锚点

- Apple AirPods Pro: https://www.apple.com/airpods-pro/
- 当前设计规范：docs/superpowers/specs/2026-06-19-apple-ui-phase2-design.md
- 当前布局精修规范：docs/superpowers/specs/2026-06-19-ui-layout-polish-design.md
- Training Studio 规范：docs/superpowers/specs/2026-06-20-training-studio-design.md
