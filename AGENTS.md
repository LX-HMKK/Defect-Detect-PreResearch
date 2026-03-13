# AGENTS.md - Agent 开发指南

## 项目概述

这是一个基于 **anomalib** 库的工业图像异常检测系统，实现了 3 种算法：
- **AutoEncoder**（基于重构）
- **PatchCore**（基于特征建模 - 工业应用最佳）
- **DRAEM**（基于自监督学习）

## 项目结构

```
Defect-Detect-PreResearch/
├── modules/                      # 核心代码模块
│   ├── data_processing/         # 数据格式转换 (MVTec AD)
│   ├── algorithm/               # 模型训练与推理
│   ├── evaluation/              # 指标计算
│   └── ui/                      # Gradio 演示界面
├── configs/                      # 算法 YAML 配置
│   ├── autoencoder.yaml
│   ├── patchcore.yaml
│   └── draem.yaml
├── data/                         # 数据目录
│   ├── raw/                     # 原始企业数据
│   └── processed/               # MVTec AD 格式处理后数据
├── results/                      # 训练结果
├── weights/                      # 模型权重
├── run_*.py                      # 入口脚本（根目录）
└── requirements.txt
```

## 命令

### 环境配置
```bash
# 创建虚拟环境
python -m venv venv

# 激活环境 (Windows)
venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 数据处理（模块 1）
```bash
python run_data_processing.py \
    --input_dir ./data/raw/enterprise_data \
    --output_dir ./data/processed/my_product \
    --max_train 150
```

### 训练（模块 2）
```bash
# 训练所有算法
python run_training.py --model all --category my_product --data_path ./data/processed/my_product

# 训练单个算法
python run_training.py --model patchcore --category my_product --data_path ./data/processed/my_product
```

### 评测（模块 3）
```bash
python run_evaluation.py --model all --category my_product --results_dir ./results
```

### UI 演示（模块 4）
```bash
python run_ui.py
# 访问 http://127.0.0.1:7860
```

### 类型检查
```bash
# 使用 pyright（配置在 pyrightconfig.json）
pyright modules/                    # 检查 modules 目录
pyright .                           # 检查整个项目

# 或使用 python
python -m pyright modules/
```

### 运行单个测试
本项目没有正式的测试套件。测试模块的方式：
```bash
# 测试数据处理
python -c "from modules.data_processing.dataset_formatter import MVTecFormatter; f = MVTecFormatter('input', 'output'); print('OK')"

# 测试评测指标
python -c "from modules.evaluation.metrics import MetricsEvaluator; e = MetricsEvaluator(); print('OK')"

# 测试训练器
python -c "from modules.algorithm.trainer import AnomalyDetectionTrainer; print('OK')"
```

## 代码风格指南

### 通用规则
- **语言**：Python + 类型注解（遵循 pyrightconfig.json：Python 3.8+，基础类型检查）
- **行长度**：目标 ≤120 字符，硬限制 150
- **编码**：UTF-8，CSV 导出使用 `utf-8-sig`

### 导入顺序
```python
# 标准库优先
import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# 第三方库
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# 本地导入
from modules.data_processing.dataset_formatter import MVTecFormatter

# 使用绝对导入，避免相对导入如：from ..algorithm import trainer
```

### 类型注解
- 所有函数参数和返回值使用类型提示
- 使用 `Optional[X]` 而非 `X | None`
- 使用 `typing` 中的 `Dict`、`List` 而非内置类型
```python
def process_data(
    input_dir: str,
    output_dir: str,
    max_samples: Optional[int] = 150
) -> Dict[str, int]:
    ...
```

### 命名规范
- **类名**：`PascalCase`（如 `AnomalyDetectionTrainer`）
- **函数/变量**：`snake_case`（如 `compute_image_auroc`）
- **常量**：`UPPER_SNAKE_CASE`（如 `SUPPORTED_MODELS`）
- **私有方法**：前缀 `_`（如 `_load_config`）
- **模块名**：`snake_case`（如 `dataset_formatter.py`）

### 错误处理
- 使用具体的异常类型
- 包含有意义的错误信息
- 先捕获具体异常
```python
try:
    from anomalib.config import get_configurable_parameters
except ImportError as e:
    print(f"❌ 错误: 未安装 anomalib。请运行: pip install anomalib")
    raise
```

### 文档字符串
使用多行文档字符串，中英文保持一致：
```python
def train(self) -> Dict[str, Any]:
    """
    训练模型。
    
    Returns:
        Dict: 训练结果，包含状态和轮数。
    """
```

### 类结构
```python
class AnomalyDetectionTrainer:
    """异常检测算法训练器"""
    
    def __init__(
        self,
        model_name: str,
        data_path: str,
        category: str,
        output_dir: str = './results',
        device: str = 'auto'
    ):
        """初始化训练器"""
        ...
    
    def train(self) -> Dict[str, Any]:
        """训练模型"""
        ...
    
    def _private_method(self) -> None:
        """内部方法"""
        ...
```

### 日志输出
- 用户可见输出使用带 emoji 的 print
- 使用中文显示用户消息
- 格式：`✅ 成功`、`❌ 错误`、`⚠️ 警告`、`📊 统计`、`🔄 处理中`
- 循环中使用 tqdm 显示进度条

### 文件组织
- **入口脚本**（`run_*.py`）：放在根目录，逻辑最小化，从 modules 导入
- **核心模块**（`modules/`）：所有业务逻辑
- **配置文件**（`configs/`）：YAML 配置文件
- **避免重复**：不要在根目录创建重复脚本

### 配置文件
- 使用 OmegaConf 管理 YAML 配置
- 模型配置放在 `configs/` 目录
- 示例：
```python
config = OmegaConf.load(self.config_path)
config.dataset.path = str(self.data_path)
config.dataset.category = self.category
```

### 数据处理规范
- 使用 `pathlib.Path` 处理所有路径
- 使用 `shutil.copy2` 复制文件
- 使用 `cv2.imread/imwrite` 处理图像
- 需要时用 `Path.resolve()` 规范化路径

### 性能考虑
- 循环中使用 tqdm 显示进度
- 尽可能使用 numpy 向量化操作
- 延迟导入重型库（如 anomalib）
- 设置随机种子保证可复现：`seed_everything(42)`

### 废弃与清理
- 合并时删除重复文件
- 对已知无害的警告使用 `warnings.filterwarnings('ignore')`
- 处理完后清理临时文件
