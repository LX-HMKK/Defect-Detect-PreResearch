# 变更日志

本文件记录项目的所有重要变更。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## 2026-06-19 — Phase 2 UI Bug 集中修复

- **修复:** 四模型对比热力图不显示 — `.compare-heatmap { position: absolute }` 全局选择器泄漏到对比槽位，热力图脱离文档流导致父容器 `.compare-heatmap-wrap` 高度塌陷至 0px，配合 `overflow: hidden` 将热力图完全裁剪。修复：全局规则收缩为 `.compare-container .compare-heatmap`（仅影响单模型滑块），`.compare-slot .compare-heatmap` 添加 `position: relative`
- **修复:** 对比滑块拖拽失效 — `.compare-handle` z-index 被 `.compare-heatmap` 遮挡，将手柄 z-index 提升至 30
- **修复:** bbox overlay 错位 — `setupBboxOverlays` 未计入 `object-fit: contain` 内容居中偏移量 (`contentOffsetX/Y`)，现正确计算内容区域缩放比和偏移
- **修复:** 四模型对比缺少原图展示 — 对比槽位新增 `.compare-mini-img-wrap` + `.compare-mini-img` 展示原始输入图
- **修复:** Cache-Control 浏览器缓存过期 — `max-age=60` 改为 `no-cache`，每次使用前强制 ETag 验证
- **修复:** SSE 消息断流 — `asyncio.to_thread` 将同步推理调用移出事件循环线程，避免阻塞 SSE 推送
- **修复:** Alpine.js `x-show` null 属性崩溃 — 敏感路径改用 `<template x-if>` 条件渲染

## 2026-06-12 — 第 2-3 阶段：安全网 + 结构改进

- **重构:** 将 monkey-patch 兼容层提取到 `modules/algorithm/_anomalib_compat.py`，trainer.py 代码量减少 160+ 行 (H1)
- **重构:** 将 `_configure_runtime_temp()` 提取到 `modules/_runtime.py` 共享模块，消除 5 个脚本中的重复代码 (M5)
- **重构:** tools/ 中 3 个脚本的 `SUPPORTED_MODELS` 改为从 `modules.algorithm` 导入，消除重复定义 (H4)
- **文档:** 修正 trainer.py docstring 中 efficientad → fre/patchcore/draem/padim (M2)
- **文档:** run_ablation.py 添加 patchcore/padim 1-epoch 原因的注释说明 (M3)
- **文档:** CLAUDE.md GPU 表述统一为 RTX 4060 Laptop GPU, 8GB VRAM (L4)
- **文档:** README 中 AGENTS.md → CLAUDE.md (L3)
- **修复:** run_confusion_matrix.py 添加英文 Score 标签的备选匹配模式，降低静默跳过风险 (M4)

## 2026-06-12 — 第 1 阶段：止血

- **测试:** 新增测试套件 — config 管理器单元测试、metrics 单元测试、trainer 烟雾测试 (C1)
- **功能:** 新增数据验证脚本 `tools/validate_data.py`，检测 BMP 伪装 PNG、目录结构、类别分布 (C2)
- **文档:** 新增 `data/DATASET_REGISTRY.md` 数据集注册表，记录 region4 缺失及各数据集已知问题 (C3)
- **重构:** 为 `modules/algorithm`、`modules/evaluation`、`modules/data_processing` 添加公共 API 导出 (H3)
- **重构:** 更新消费者导入路径为模块级 API (H3)
- **构建:** 新增 `requirements.txt` 固定依赖版本 (M1)
- **文档:** README 修正 anomalib 版本 `>=2.0.0` → `==2.3.0` (M1)

## 2026-06-18 — Phase 1-2：前端增强

### Phase 1 — 主题系统 + 手动切换 + 图标
- **新增:** `modules/ui/theme.py` — 主题管理器（DARK/LIGHT 色板字典、`build_css_variables()` CSS 编译、`get_light_css()`/`get_theme_switch_html()`/`get_theme_js()`/`get_favicon_html()` 注入函数）
- **新增:** `modules/ui/static/theme.js` — 主题切换逻辑（localStorage 持久化、`matchMedia` 系统检测、点击切换、双击清除偏好、`.dark` 类同步、favicon 联动）
- **新增:** 手动亮/暗切换按钮 — 标题栏右侧太阳/月亮 SVG 按钮，`html[data-theme="light"]` 选择器控制覆盖
- **新增:** Favicon — SVG 内联菱形图标（深色圆底 + 蓝色菱形），`theme.js` 按主题动态切换
- **修改:** `modules/ui/demo.py` — 集成 theme 模块（favicon/亮色 CSS/JS 注入、标题栏 switch 按钮嵌入）
- **修改:** `modules/ui/styles.css` — 亮色变量来源注释

### Phase 2 — 推理结果交互增强
- **新增:** `modules/ui/static/inference-interact.js` — 热力图 hover tooltip（离屏 canvas 读灰度值 → 像素级异常分数）+ NMS bbox overlay（hover 高亮 `var(--accent)` 边框）
- **新增:** 结果卡片逐层入场动画 — `.reveal-child-1/2/3` 类名，500ms `ease-out-expo`，100ms 间隔级联
- **新增:** `.heatmap-tooltip` / `.bbox-overlay` CSS — Apple 风格浮动 tooltip（磨砂玻璃 + mono 字体）+ bbox 叠加层 hover 效果
- **修改:** `modules/ui/demo.py:_generate_heatmap()` — 多返回原始灰度图 numpy（供 JS 读取像素值）
- **修改:** `modules/ui/demo.py:predict()` — base64 编码灰度图 + bbox JSON，传递至 `_format_result()` 嵌入结果 HTML
- **修改:** `modules/ui/demo.py:_format_result()` — 新参数 `anomaly_map_b64`/`bboxes_json`，隐藏数据元素 + 逐层动画类名

### 修复
- **修复:** 亮色模式下 Gradio 组件（`.form`/`.block`）背景仍为暗色 — 根因 `body.dark` 类未移除，`theme.js` 改为 `querySelectorAll('.dark')` 全局清除
- **修复:** LIGHT 色板补充 Gradio 中性色阶（`neutral_700~950`）和组件级变量（`block_background_fill` 等 12 个）
- **修复:** Favicon `<link>` 的 SVG 双引号与 HTML `href="..."` 冲突导致文本泄露 — SVG 属性改用单引号
- **修复:** 标题栏 `gr.Markdown` 净化器剥离 `<button>`/`<svg>` — 改为 `gr.HTML`
- **修复:** Gradio Svelte 不执行 `innerHTML` 注入的 `<script>` — 改用 `<img onerror>` bootstrapper 动态创建 `<script>` 注入 `<head>`

## 2026-06-18 — Phase 3：细节打磨 + 全量前端特征完成

### Phase 3 — 细节打磨
- **新增:** 主题切换平滑过渡 — 400ms `background-color`/`border-color`/`color`/`box-shadow` 过渡覆盖 Gradio 组件
- **新增:** 骨架屏（Skeleton Screen）— `SKELETON_HTML` + `@keyframes shimmer`，模型加载/推理时显示
- **新增:** 热力图图例延迟淡入 — `.reveal-child-4` 级联动画
- **新增:** `prefers-reduced-motion` 无障碍适配 — 骨架屏 shimmer 动画降级为静态

### 追加特征（A1/V2/V3/V4/C1/C2）
- **新增:** A1 字体优化 — 字体栈增加 `Inter Variable`/`Noto Sans SC`/`JetBrains Mono`，`text-rendering: optimizeLegibility`，`font-synthesis: none`
- **新增:** V2 全页加载遮罩 — `.page-loader` 首次访问 1.2s 后自动淡出，双环旋转动画
- **新增:** V3 Logo — 标题栏左侧 36×36 渐变菱形 SVG（与 Favicon 同源）
- **新增:** V4 页脚入场动画 — `.footer-item` 交错 `footerRise` 动画 (100/250/400ms)
- **新增:** C1 对比模式流式渐进渲染 — `on_compare_click` 改为 generator `yield`，每完成一个模型立即更新结果，其余显示加载态
- **新增:** C2 不确定进度条 — `.progress-bar` shimmer 滑动动画，加载态时在状态面板中显示

### Bug 修复
- **修复:** 下拉框 INPUT 左右不对称 — `.block:has(.wrap-inner)` 和 `.wrap-inner` 同时清零 `padding-left` + `padding-right`（之前仅清零左侧）
- **修复:** 滚轮滚动时下拉弹出框偏移 — 移除 `.reveal`/`.reveal-child-*` 的 `animation-fill-mode: forwards`，默认态设 `transform: none; filter: none;`（CSS Transform/Filter Effects spec §Containing Block）
- **修复:** 导入顺序违规 — `base64`/`io`/`json` 移至标准库组，`cv2` 保持第三方首位
- **修复:** `_format_result` 空指针防护 — `self.current_model=None` 时返回友好错误
- **修复:** `theme.js` 双重 `bindEvents()` 调用 — 删除外部重复调用
- **修复:** FOUC（亮色用户首帧暗色闪烁）— `gr.Blocks(head=...)` 注入阻塞式反闪烁脚本 + CSS 兜底
- **修复:** `inference-interact.js` 内存泄漏 — `_bboxCleanups[]` 追踪 resize/load 监听器，`cleanupOverlays()` 统一释放
- **修复:** MutationObserver 递归风险 — 回调中先 `disconnect()` → 操作 DOM → `setTimeout` 后重新 `observe()`

## [未发布] - 2026-06-11

### 新增
- 添加 PaDiM 算法支持（patch 高斯分布建模 + 马氏距离），包括模型配置 `configs/padim.yaml`、训练/评估/UI 全流程集成
- 添加独立阈值计算脚本 (`scripts/run_threshold.py`)，支持按模型/类别计算最优阈值并导出 JSON
- 为 DRAEM/FRE 训练添加早停机制，通过 YAML 配置 `early_stopping` 参数
- 添加小样本鲁棒性分析工具 (`tools/run_small_sample.py`)，支持 30/60/100/150 张正常样本的梯度实验
- 添加推理性能基准测试工具 (`tools/run_benchmark.py`)，测量各算法的推理速度、显存占用、参数量
- 添加数据集统计分析工具 (`tools/run_data_stats.py`)，自动生成样本量、缺陷类型分布报告
- 添加综合实验报告生成工具 (`tools/run_report.py`)，聚合所有实验结果 JSON 生成对比报告
- 在 CLAUDE.md 中新增文档语言规则：所有文档、注释、提交信息必须使用中文

### 变更
- 重构脚本目录：将汇报/分析类脚本移至 `tools/` 目录，scripts 保留核心工作流
- 更新 CLAUDE.md 架构描述：补充 `assets/`、`docs/`、`pre_trained/`、`temp/` 目录说明，修正配置列表包含 `padim.yaml`
- 更新 CLAUDE.md Windows 特定章节：添加 `_configure_runtime_temp()` 和 UTF-8 stdout 包装模式文档
- 修复 CLAUDE.md 中基准测试命令缺少 `-d` 标志的问题
- 升级工业暗色模式 UI：优化色彩调色板、紧凑按钮、抛光图像区域、动画进度条、改进排版

### 修复
- 修复训练评估流程中的回调兼容性问题（anomalib 2.3.0 与 PyTorch Lightning 1.9.5）
- 修复阈值搜索范围从受限分数区间扩展至完整 [0,1] 范围，添加分数分布诊断
- 修复张量安全取值和 NMS bbox 热力图阈值问题

## [1.0.0] - 2026-03-28

### 新增
- 初始化异常检测项目，包含算法模块（PatchCore、DRAEM、FRE）
- 将 YAML 配置集成到训练流程和评估工作流中

### 变更
- 升级至 anomalib 2.x，支持三种算法
- 用 FRE 重构方法替换 Ganomaly
- 美化 UI 界面并优化启动速度
- 改进检测结果显示和交互体验
- 优化算法参数：启用 PatchCore 预训练，提升 DRAEM 效果
- 优化 DRAEM 训练配置参数
- 配置预训练权重缓存目录

### 修复
- 修复 PatchCore AUROC 卡在 0.5 的问题，在 MVTec AD bottle 数据集上验证
- 将 emoji 替换为 ASCII 编码，修复 Windows GBK 显示问题
- 修复模型权重加载与 anomalib 2.x 目录结构的兼容性问题
