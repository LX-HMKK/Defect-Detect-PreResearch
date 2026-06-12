# 变更日志

本文件记录项目的所有重要变更。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

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
