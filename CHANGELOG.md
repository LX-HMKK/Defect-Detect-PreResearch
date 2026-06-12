# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 2026-06-12 — Phase 1: Stop the Bleeding

- **test:** Added test suite — config manager unit tests, metrics unit tests, trainer smoke tests (C1)
- **feat:** Added data validation script `tools/validate_data.py` — detects BMP-as-PNG, directory structure, class distribution (C2)
- **docs:** Added `data/DATASET_REGISTRY.md` dataset registry, documenting region4 absence and per-dataset known issues (C3)
- **refactor:** Added public API exports for `modules/algorithm`, `modules/evaluation`, `modules/data_processing` (H3)
- **refactor:** Updated consumer import paths to module-level API (H3)
- **build:** Added `requirements.txt` with pinned dependency versions (M1)
- **docs:** README fixes — anomalib version `>=2.0.0` → `==2.3.0` (M1)

## [Unreleased] - 2026-06-11

### Added
- 添加 PaDiM 算法支持（patch 高斯分布建模 + 马氏距离），包括模型配置 `configs/padim.yaml`、训练/评估/UI 全流程集成
- 添加独立阈值计算脚本 (`scripts/run_threshold.py`)，支持按模型/类别计算最优阈值并导出 JSON
- 为 DRAEM/FRE 训练添加早停机制，通过 YAML 配置 `early_stopping` 参数
- 添加小样本鲁棒性分析工具 (`tools/run_small_sample.py`)，支持 30/60/100/150 张正常样本的梯度实验
- 添加推理性能基准测试工具 (`tools/run_benchmark.py`)，测量各算法的推理速度、显存占用、参数量
- 添加数据集统计分析工具 (`tools/run_data_stats.py`)，自动生成样本量、缺陷类型分布报告
- 添加综合实验报告生成工具 (`tools/run_report.py`)，聚合所有实验结果 JSON 生成对比报告
- 在 CLAUDE.md 中新增文档语言规则：所有文档、注释、提交信息必须使用中文

### Changed
- 重构脚本目录：将汇报/分析类脚本 (`run_report`, `run_data_stats`, `run_benchmark`, `run_small_sample`) 移至 `tools/` 目录，脚本保留核心工作流
- 更新 CLAUDE.md 架构描述：补充 `assets/`、`docs/`、`pre_trained/`、`temp/` 目录说明，修正配置列表包含 `padim.yaml`
- 更新 CLAUDE.md Windows 特定章节：添加 `_configure_runtime_temp()` 和 UTF-8 stdout 包装模式文档
- 修复 CLAUDE.md 中基准测试命令缺少 `-d` 标志的问题
- 升级工业暗色模式 UI：优化色彩调色板、紧凑按钮、抛光图像区域、动画进度条、改进排版

### Fixed
- 修复训练评估流程中的回调兼容性问题（anomalib 2.3.0 与 PyTorch Lightning 1.9.5）
- 修复阈值搜索范围从受限分数区间扩展至完整 [0,1] 范围，添加分数分布诊断
- 修复张量安全取值和 NMS bbox 热力图阈值问题

## [1.0.0] - 2026-03-28

### Added
- Initialize anomaly detection project with algorithm modules (PatchCore, DRAEM, FRE)
- Integrate YAML configuration into training pipeline and evaluation workflows

### Changed
- Upgrade to anomalib 2.x with three-algorithm support
- Replace Ganomaly with FRE reconstruction method
- Beautify UI and optimize startup speed
- Improve detection result display and interaction experience
- Optimize algorithm parameters: enable pretraining for PatchCore, improve DRAEM effectiveness
- Optimize DRAEM training configuration parameters
- Configure pretrained weight cache directory

### Fixed
- Fix PatchCore AUROC stuck at 0.5 and validate on MVTec AD bottle dataset
- Replace emoji with ASCII encoding to fix Windows GBK display issues
- Fix model weight loading compatibility with anomalib 2.x directory structure
