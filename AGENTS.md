# AGENTS.md - Agent 开发指南

**项目**: 工业图像异常检测系统 (基于 anomalib 2.x)
**环境**: Miniforge 虚拟环境 `anomalib` | Python 3.10

---

## 项目结构

```
Defect-Detect-PreResearch/
├── modules/
│   ├── algorithm/        # 模型训练 (PatchCore/FRE/DRAEM)
│   ├── config/           # 配置管理 (YAML)
│   ├── data_processing/  # 数据格式转换 (MVTec AD)
│   ├── evaluation/        # 指标计算 (AUROC/AUPR/PRO)
│   └── ui/               # Gradio Web 界面
├── configs/               # 算法 YAML 配置
├── config/config.yaml     # 主配置文件
├── run_*.py              # 入口脚本
└── results/              # 训练结果
```

---

## 命令速查

### 运行命令 (conda 环境)

```bash
# 激活环境
mamba activate anomalib

# 数据处理
python run_data_processing.py -i ./data/raw -o ./data/processed/bottle --max_train 150

# 模型训练
python run_training.py -m patchcore -c bottle -d ./data
python run_training.py -m all -c all -d ./data    # 所有模型+数据集

# 独立计算阈值
python run_threshold.py -m patchcore -c bottle     # 计算阈值
python run_threshold.py -m all -c all --save       # 计算并保存

# 评估
python run_evaluation.py -m patchcore -c bottle

# UI 启动
python run_ui.py
# 访问 http://127.0.0.1:7860

# 直接指定 Python (Windows)
"C:\Users\lx_hm\.conda\envs\anomalib\python.exe" run_training.py -m patchcore -c bottle
```

### 测试 (无正式测试套件)

```bash
# 环境验证
python -c "from anomalib.data import MVTec; from anomalib.engine import Engine; print('OK')"

# 快速验证导入
python -c "from modules.algorithm.trainer import AnomalyDetectionTrainer; print('OK')"
```

---

## 代码规范

### 导入顺序 (必须遵守)

```python
# 1. 标准库 (按字母顺序)
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# 2. 第三方库
import cv2  # 必须放在 anomalib 之前导入
import numpy as np
import pandas as pd
from tqdm import tqdm

# 3. 框架 (anomalib)
from anomalib.data import MVTec
from anomalib.engine import Engine

# 4. 本地导入 (绝对导入)
from modules.config import get_threshold
from modules.evaluation.metrics import MetricsEvaluator
```

### 命名规范

| 类型 | 规范 | 示例 |
|------|------|------|
| 类名 | PascalCase | `AnomalyDetectionTrainer`, `MetricsEvaluator` |
| 函数/变量 | snake_case | `compute_image_auroc`, `good_scores` |
| 常量 | UPPER_SNAKE_CASE | `SUPPORTED_MODELS`, `NMS_BBOX_THRESHOLD` |
| 私有方法 | 前缀 `_` | `_load_config`, `_compute_optimal_threshold` |
| 模块级私有 | 前缀 `_` | `_lightning_callback_class` |

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
    print(f"❌ 错误: 请运行 pip install anomalib>=2.0.0")
    raise

# 避免空 catch
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
    完整流程：训练 + 评估
    
    Args:
        max_epochs: 最大训练轮次
    
    Returns:
        Dict: 评估结果（4个硬性指标）
    
    Raises:
        ValueError: 配置缺失时抛出
    """
```

---

## Git 提交规范 (Angular 协议)

### 格式
```
<类型>(<范围>): <主题>

[可选正文]
```

### 类型

| 类型 | 说明 | 示例 |
|------|------|------|
| feat | 新功能 | `feat(ui): 添加算法切换功能` |
| fix | 修复 bug | `fix(trainer): 修复阈值搜索范围` |
| docs | 文档更新 | `docs: 更新 README` |
| style | 代码格式 | `style: 格式化代码` |
| refactor | 重构 | `refactor: 重构模型配置结构` |
| perf | 性能优化 | `perf(patchcore): 启用预训练权重` |

### 规则
- 主题行不超过 72 字符
- 使用命令式语气 (add, fix, update)
- 禁止添加 Co-authored-by

---

## 关键注意事项

| 规则 | 说明 |
|------|------|
| **cv2 优先导入** | `import cv2` 必须在 anomalib 之前，否则有 DLL 加载问题 |
| **数据路径** | `--data_path` 指向类别父目录 (`./data` 不是 `./data/bottle`) |
| **Windows 多进程** | `num_workers: 0` 避免多进程问题 |
| **张量取值** | 使用 `.cpu().max().item()` 安全获取标量 |

---

## 算法推荐

| 算法 | 原理 | image_AUROC | 推荐度 |
|------|------|-------------|--------|
| **PatchCore** | 特征记忆库 + 最近邻搜索 | 100% | ⭐⭐⭐ 首选 |
| **FRE** | 特征重构误差 | 95% | ⭐⭐ 备选 |
| **DRAEM** | 合成异常 + 判别网络 | 99% | ⭐⭐ 备选 |

**核心约束**: 只有正常样本，无监督设定。

---

## 环境配置

```bash
# 创建环境 (mamba 推荐)
mamba create -n anomalib python=3.10 -y
mamba activate anomalib

# 安装依赖
mamba install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install anomalib>=2.0.0
pip install opencv-python==4.8.1.78 timm
```
