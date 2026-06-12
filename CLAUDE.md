# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在本仓库中工作时提供指导。

## 项目概述

基于 anomalib 2.3.0 的工业异常检测系统。仅使用正常（无缺陷）样本进行无监督缺陷检测。四种算法：PatchCore（特征记忆库 + 最近邻搜索）、PaDiM（patch 高斯分布建模 + 马氏距离）、FRE（特征重构误差）、DRAEM（合成异常 + 判别网络）。

**环境**: Miniforge conda 环境 `anomalib` | Python 3.10 | PyTorch 2.x CUDA 11.8

## 命令

```bash
# 激活环境
mamba activate anomalib

# 数据处理：将原始图像转换为 MVTec AD 格式
python scripts/run_data_processing.py -i ./data/raw -o ./data/processed/bottle --max_train 150

# 训练单个模型/单个数据集
python scripts/run_training.py -m patchcore -c bottle -d ./data

# 训练所有模型/所有数据集
python scripts/run_training.py -m all -c all -d ./data

# 小样本鲁棒性分析（30/60/100/150 张正常样本）
python tools/run_small_sample.py -m all -c all -d ./data

# 从已训练 checkpoint 计算最优阈值
python scripts/run_threshold.py -m patchcore -c bottle
python scripts/run_threshold.py -m all -c all --save

# 评估（加载已保存的 JSON 结果）
python scripts/run_evaluation.py -m patchcore -c bottle
python scripts/run_evaluation.py -m all -c all

# 生成综合实验报告
python tools/run_report.py

# 数据集统计分析
python tools/run_data_stats.py -d ./data

# 推理性能基准测试
python tools/run_benchmark.py -m all -c bottle -d ./data

# 启动 Gradio UI
python scripts/run_ui.py
# → http://127.0.0.1:7860

# 直接指定 Python 路径 (Windows)
"C:\Users\lx_hm\.conda\envs\anomalib\python.exe" scripts/run_training.py -m patchcore -c bottle

# 快速冒烟测试
python -c "from anomalib.data import MVTec; from anomalib.engine import Engine; print('OK')"
python -c "from modules.algorithm.trainer import AnomalyDetectionTrainer; print('OK')"
```

CLI 参数：`-m` 模型 (`patchcore|padim|fre|draem|all`)，`-c` 类别 (`bottle|carpet|region1|region2|region3|region5|all`)，`-d` 数据根目录。

## 环境配置

```bash
# 创建环境（推荐 mamba）
mamba create -n anomalib python=3.10 -y
mamba activate anomalib

# 安装依赖
mamba install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install anomalib>=2.0.0
pip install opencv-python==4.8.1.78 timm
```

## 架构

```
scripts/run_*.py              # 入口脚本（CLI 轻量封装）
modules/
  algorithm/trainer.py         # 核心：AnomalyDetectionTrainer + 模型/数据模块工厂函数
  config/manager.py            # ConfigManager 单例，管理 configs/config.yaml
  data_processing/dataset_formatter.py  # MVTecFormatter：原始数据 → MVTec AD 结构
  evaluation/metrics.py        # MetricsEvaluator（从零实现 AUROC/AUPR/PRO）
  ui/demo.py                   # Gradio UI：AnomalyDetector + create_interface
configs/
  config.yaml                        # 主配置（路径、训练参数、阈值）
  {patchcore,padim,fre,draem}.yaml   # 各模型 anomalib CLI 格式配置
results/                             # 训练结果
assets/                              # 静态资源 (requirements.txt, pyrightconfig)
docs/                                # 演示材料 (讲稿.html/md)
pre_trained/                         # torch_hub / huggingface 权重缓存
temp/                                # 临时文件 (pycache, logs)
```

### 关键类及其职责

- **`AnomalyDetectionTrainer`** (`modules/algorithm/trainer.py`) — 核心调度器。`setup()` 创建 datamodule + model，`train()` 运行 anomalib `Engine.fit()`，`evaluate()` 运行 `Engine.test()` 并通过 Youden's J 统计量网格搜索计算最优阈值，`_save_results()` 将 JSON 写入 `results/comparison/`。静态方法 `compare_models()` 生成对比 CSV/Markdown。

- **`get_model_from_config(model_name, config)`** — 始终附加 `anomalib.metrics.Evaluator`，包含 6 个指标（图像级 AUROC/AUPR/F1Score + 像素级 AUROC/PRO/F1Score）。参数严格从配置读取——缺失键抛出 `ValueError`。

- **`get_datamodule_from_config(data_path, category, model_name, config)`** — 自动检测 MVTec AD 与通用 Folder 格式。Folder 格式始终设置 `task='segmentation'`。

- **`ConfigManager`** (`modules/config/manager.py`) — 基于 `configs/config.yaml` 的单例。支持点号分隔取值 `get('data.image_size')`。`get_threshold(model, dataset)` 从已保存的 JSON 结果读取。模块级便捷函数：`get()`、`get_threshold()`、`get_model_config()`、`get_data_config()`。

- **`MetricsEvaluator`** (`modules/evaluation/metrics.py`) — 独立指标计算（由 `run_evaluation.py` 使用）。训练器改为委托 anomalib 内置评估器。

- **`MVTecFormatter`** (`modules/data_processing/dataset_formatter.py`) — 将企业原始图像转换为 MVTec AD 目录布局，使用 letterbox 缩放。训练样本上限由 `max_train_samples` 控制。

- **`AnomalyDetector`** (`modules/ui/demo.py`) — 加载 checkpoint、执行推理、生成热力图叠加层，并在独立阈值 (`NMS_BBOX_THRESHOLD = 0.3`) 下生成 NMS 边界框，与分类阈值无关。

### 数据流

1. 原始图像 → `MVTecFormatter` → MVTec AD 格式 (`train/good/`, `test/<defect>/`, `ground_truth/<defect>/`)
2. `AnomalyDetectionTrainer.setup()` → `get_datamodule_from_config()` + `get_model_from_config()`
3. `train()` → anomalib `Engine.fit()`（PatchCore：仅 coreset 子采样，1 个 epoch）
4. `evaluate()` → `Engine.test()` → 提取 4 个指标 → `_compute_optimal_threshold()` (Youden's J) → `_save_results()` → JSON 输出至 `results/comparison/{model}_{category}_results.json`
5. UI 通过 `find_latest_checkpoint()` 加载 checkpoint → `Engine.predict()` → 热力图 + NMS 边界框

### 早停机制

DRAEM/FRE 监控 `val_image_AUROC`；PatchCore 监控 `image_AUROC`。在各模型 `{model}.yaml` 的 `early_stopping` 下配置 `patience`、`min_delta`、`monitor_metric`。

## 编码规范

### 文档语言

**所有文档、注释、提交信息必须使用中文。** 包括但不限于：README、CHANGELOG、CLAUDE.md、docstring、行内注释、Git 提交主题行。

### 导入顺序（必须遵守）

```python
# 1. 标准库（按字母顺序）
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# 2. 第三方库 — cv2 必须在 anomalib 之前导入
import cv2  # ← 始终是第三方导入中的第一个
import numpy as np
import pandas as pd
from tqdm import tqdm

# 3. 框架 (anomalib)
from anomalib.data import MVTec
from anomalib.engine import Engine

# 4. 本地导入（绝对导入）
from modules.config import get_threshold
from modules.evaluation.metrics import MetricsEvaluator
```

`import cv2` **必须**在任何 anomalib 导入之前。否则 Windows 上 DLL 加载失败。

### 命名规范

| 类型 | 规范 | 示例 |
|------|------|------|
| 类名 | PascalCase | `AnomalyDetectionTrainer`, `MetricsEvaluator` |
| 函数/变量 | snake_case | `compute_image_auroc`, `good_scores` |
| 常量 | UPPER_SNAKE_CASE | `SUPPORTED_MODELS`, `NMS_BBOX_THRESHOLD` |
| 私有方法 | `_` 前缀 | `_load_config`, `_compute_optimal_threshold` |
| 模块级私有 | `_` 前缀 | `_lightning_callback_class` |

### 类型注解

```python
# 使用类型注解提高可读性
def compute_threshold(scores: List[float], labels: List[bool]) -> float:
    threshold: float = 0.5
    return threshold

# Union 类型使用 `|`
def process_data(data: str | Path) -> Dict[str, Any]:
    ...
```

### 错误处理

```python
# 必须捕获具体异常
try:
    from anomalib.data import MVTec
except ImportError as e:
    print(f"错误：请运行 pip install anomalib>=2.0.0")
    raise

# 禁止空 except — 始终指定异常类型
try:
    result = risky_operation()
except ValueError as e:
    print(f"[WARN] 值错误: {e}")
    raise  # 或返回默认值
```

### 文档字符串

```python
def train_and_evaluate(self, max_epochs: Optional[int] = None) -> Dict[str, Any]:
    """
    完整流程：训练 + 评估。

    Args:
        max_epochs: 最大训练轮次。

    Returns:
        Dict: 评估结果（4 个核心指标）。

    Raises:
        ValueError: 必需配置键缺失时抛出。
    """
```

## Git 提交规范

Angular 协议：`<类型>(<范围>): <主题>`。

| 类型 | 说明 | 示例 |
|------|------|------|
| feat | 新功能 | `feat(ui): 添加算法切换功能` |
| fix | 修复 bug | `fix(trainer): 修复阈值搜索范围` |
| docs | 文档更新 | `docs: 更新 README` |
| style | 代码格式 | `style: 格式化代码` |
| refactor | 重构 | `refactor: 重构模型配置结构` |
| perf | 性能优化 | `perf(patchcore): 启用预训练权重` |

规则：
- 主题行不超过 72 字符
- 使用命令式语气（add, fix, update）
- **禁止添加 `Co-authored-by`** 到提交信息

## 算法推荐

| 算法 | 原理 | image_AUROC | 推荐度 |
|------|------|-------------|--------|
| **PatchCore** | 特征记忆库 + 最近邻搜索 | 100% | 首选 |
| **PaDiM** | patch 高斯分布 + 马氏距离 | - | 特征建模对照 |
| **FRE** | 特征重构误差 | 95% | 备选 |
| **DRAEM** | 合成异常 + 判别网络 | 99% | 备选 |

**核心约束**：只有正常样本可用，无监督设定。

## 关键注意事项

### 训练并发限制

**训练任务必须串行执行，一次只能跑一个。** 本机 GPU (RTX 4060 Laptop GPU, 8GB VRAM) 显存有限，多个训练进程并发会导致 GPU 内存溢出、系统卡死。必须等当前训练完全结束后才能启动下一个。

### 数据集路径

`--data_path` / `data_path` 始终指向**类别父目录** (`./data`)，而非 `./data/bottle`。

### Windows 特定

- 配置中 `num_workers: 0` 以避免多进程问题
- 安全张量取值：使用 `.cpu().max().item()` 而非 `.max().item()`
- 新脚本应重定向 `sys.pycache_prefix` 到 `./temp/pycache`（参见现有脚本中的 `_configure_runtime_temp()`）
- 新脚本应将 stdout/stderr 包装为 UTF-8：`sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')`

### 配置加载优先级

当各模型 YAML 配置路径传入 `AnomalyDetectionTrainer` 时，优先使用该配置。否则回退到 `ConfigManager.get()` 从 `configs/config.yaml` 读取。

### Trainer 猴子补丁

`modules/algorithm/trainer.py` 第 64-218 行包含针对 anomalib 2.3.0 `TimerCallback` 与 PyTorch Lightning 1.9.5 `Trainer` 的兼容性补丁（回调签名不匹配）。在未验证 Lightning/anomalib 版本组合之前，不要移除这些补丁。

## 相关文件

- [README.md](README.md) — 完整项目文档，含算法对比表
- [CHANGELOG.md](CHANGELOG.md) — 发布历史
