# AGENTS.md - Agent 开发指南

## 项目本质

**这是一个工业图像异常检测系统的开发指南，基于 anomalib 2.x 库。**

核心问题：如何在仅有正常样本的情况下，自动检测工业产品缺陷？

## 核心模块

```
Defect-Detect-PreResearch/
├── modules/
│   ├── data_processing/    # 1. 数据转换：将原始图片转为 MVTec AD 格式
│   ├── algorithm/           # 2. 模型训练：PatchCore / DRAEM / Ganomaly
│   ├── evaluation/          # 3. 指标计算：AUROC / AUPR / PRO
│   └── ui/                 # 4. 交互演示：Gradio Web 界面
├── configs/                 # YAML 配置文件
├── run_*.py               # 入口脚本
└── results/               # 训练结果
```

## 算法选择

**工业异常检测的核心约束：只有正常样本，没有缺陷标注。**

| 算法 | 原理 | 核心优势 | MVTec AD 效果 | 推荐度 |
|:---|:---|:---|:---:|:---:|
| PatchCore | 特征记忆库 + 最近邻搜索 | 简单、高效、工业最优 | 100% AUROC | ⭐⭐⭐ |
| DRAEM | 合成异常 + 判别网络 | 像素级定位好 | 99% AUROC | ⭐⭐ |
| Ganomaly | GAN 重构 | 概念直观 | 49% AUROC | ⭐ |

**结论：首选 PatchCore。**

## 命令速查

```bash
# 1. 数据处理（必须）
python run_data_processing.py -i ./data/raw -o ./data/processed/my_product --max_train 150

# 2. 训练（推荐 PatchCore）
python run_training.py -m patchcore -c bottle -d ./data

# 3. 评估
python run_evaluation.py -m all -c bottle

# 4. UI 演示
python run_ui.py
```

## 代码规范

### 导入顺序
```python
# 1. 标准库
import os, sys, json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# 2. 第三方库
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# 3. 本地导入（绝对导入）
from modules.data_processing.dataset_formatter import MVTecFormatter
```

### 命名规范
| 类型 | 规范 | 示例 |
|------|---------|---------|
| 类名 | PascalCase | `AnomalyDetectionTrainer` |
| 函数/变量 | snake_case | `compute_image_auroc` |
| 常量 | UPPER_SNAKE_CASE | `SUPPORTED_MODELS` |
| 私有方法 | 前缀 `_` | `_load_config` |

### 错误处理
```python
try:
    from anomalib.data import MVTec
except ImportError as e:
    print(f"❌ 错误: 请运行 pip install anomalib>=2.0.0")
    raise
```

## 关键注意事项

1. **数据路径**：`--data_path` 指向类别父目录（如 `./data` 不是 `./data/bottle`）
2. **cv2 导入**：必须在 anomalib 之前导入
3. **Windows 多进程**：`num_workers: 0`
4. **Ganomaly 参数**：`lr=0.0002`（不是 0.002）

## Git 提交规范（Angular 协议）

### 格式
```
<类型>(<范围>): <主题>

[可选正文]
```

### 类型说明

| 类型 | 说明 | 示例 |
|------|------|------|
| feat | 新功能 | `feat(ui): 添加算法切换功能` |
| fix | 修复 bug | `fix(trainer): 修复显存溢出问题` |
| docs | 文档更新 | `docs: 更新 README` |
| style | 代码格式 | `style: 格式化代码` |
| refactor | 重构 | `refactor: 重构模型配置结构` |
| perf | 性能优化 | `perf(patchcore): 启用预训练权重` |

### 规则
- 主题行不超过 72 字符
- 使用命令式语气（add, fix, update）
- 禁止添加 Co-authored-by

## 环境配置

本项目在 **Miniforge 虚拟环境** 中开发，环境名称：`anomalib`

> Miniforge 是 conda 的轻量级替代，使用 mamba 作为包管理器，安装更快速。

### 创建环境

```bash
# 方式1：使用 mamba（推荐，更快）
mamba create -n anomalib python=3.10 -y
mamba activate anomalib

# 方式2：使用 conda
conda create -n anomalib python=3.10 -y
conda activate anomalib

# 安装依赖（建议用 mamba/conda 安装 torch，再 pip 安装其他）
mamba install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install anomalib>=2.0.0
pip install opencv-python==4.8.1.78 timm
```

### 运行命令

```bash
# 方式1：激活环境后运行
mamba activate anomalib
python run_training.py -m patchcore -c bottle

# 方式2：直接指定 Python 路径
"C:\Users\lx_hm\.conda\envs\anomalib\python.exe" run_training.py -m patchcore -c bottle
```

### 环境测试

```bash
"C:\Users\lx_hm\.conda\envs\anomalib\python.exe" -c "
from anomalib.data import MVTec
from anomalib.engine import Engine
from anomalib.models import Patchcore, Draem, Ganomaly
print('OK')
"
```
