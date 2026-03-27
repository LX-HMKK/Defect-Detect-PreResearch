# AGENTS.md - Agent 开发指南

这是一个基于 **anomalib 2.x** 库的工业图像异常检测系统，实现了 3 种算法：
- **Ganomaly**（基于重构 - GAN）
- **PatchCore**（基于特征建模 - 工业应用最佳）
- **DRAEM**（基于自监督学习）

## 项目结构

```
Defect-Detect-PreResearch/
├── asset/                  # 项目配置与环境脚本
│   ├── pyrightconfig.json # Pyright 类型检查配置
│   ├── requirements.txt   # Python 依赖清单
│   └── setup_miniforge.bat # Miniforge 环境配置脚本
├── modules/
│   ├── data_processing/    # MVTec AD 数据格式转换
│   ├── algorithm/          # 模型训练与推理
│   ├── evaluation/         # 指标计算
│   └── ui/                # Gradio 演示界面
├── configs/                # YAML 算法配置 (Anomalib 2.x 格式)
├── data/
│   ├── raw/               # 原始企业数据
│   └── processed/         # MVTec AD 格式处理后数据
├── results/               # 训练结果
├── run_*.py              # 入口脚本（根目录）
```

## Anomalib 2.x 升级说明

本项目已升级至 **anomalib 2.x**，主要 API 变化如下：

| 旧 API (0.7.x) | 新 API (2.x) | 说明 |
|:---|:---|:---|
| `from anomalib.config import get_configurable_parameters` | 直接使用 `OmegaConf` | 配置管理 |
| `from anomalib.data import get_datamodule` | `from anomalib.data import MVTec, Folder` | 数据模块 |
| `from anomalib.models import get_model` | `from anomalib.models import Patchcore, Draem, Ganomaly` | 模型 |
| `from pytorch_lightning import Trainer` | `from anomalib.engine import Engine` | 训练引擎 |
| `from anomalib.deploy import TorchInferencer` | `Engine().predict()` | 推理 |
| `trainer.fit(model, datamodule)` | `engine.fit(model=model, datamodule=datamodule)` | 训练 |
| `trainer.test(model, datamodule)` | `engine.test(model=model, datamodule=datamodule)` | 测试 |

### 配置文件格式变化

旧格式 (0.7.x):
```yaml
dataset:
  name: mvtec
  format: mvtec
  path: ./data
  category: bottle

model:
  name: patchcore
  backbone: wide_resnet50_2
```

新格式 (2.x):
```yaml
data:
  class_path: anomalib.data.MVTec
  init_args:
    root: ./data
    category: bottle

model:
  class_path: anomalib.models.Patchcore
  init_args:
    backbone: wide_resnet50_2
```

## 命令

### 环境配置（Miniforge）

**自动化脚本**：可使用 `asset/setup_miniforge.bat` 自动完成以下步骤。

```bash
# 创建虚拟环境
"C:\ProgramData\miniforge3\Scripts\conda.exe" create -n anomalib python=3.10 -y

# 通过 conda 安装 PyTorch CUDA 版本（RTX 4060 需要 CUDA 11.8）
"C:\ProgramData\miniforge3\Scripts\conda.exe" install -n anomalib -c pytorch -c nvidia pytorch torchvision pytorch-cuda=11.8 -y

# 安装其他依赖
"C:\ProgramData\miniforge3\Scripts\conda.exe" run -n anomalib conda install -c conda-forge scikit-learn scipy pandas pillow opencv tqdm pyyaml -y

# 通过 pip 安装 anomalib 2.x（推荐）
"C:\Users\lx_hm\.conda\envs\anomalib\python.exe" -m pip install "anomalib>=2.0.0" --upgrade

# 升级依赖（解决版本冲突）
"C:\Users\lx_hm\.conda\envs\anomalib\python.exe" -m pip install timm --upgrade
"C:\Users\lx_hm\.conda\envs\anomalib\python.exe" -m pip install opencv-python==4.8.1.78 --force-reinstall --no-deps
```

**注意**：首次运行模型训练时需要从 HuggingFace 下载预训练权重，请确保网络连接正常。

### 直接运行（推荐）

```bash
# 训练单个算法
python run_training.py --model patchcore --category bottle --data_path ./data --device cuda

# 训练所有算法
python run_training.py --model all --category bottle --data_path ./data --device cuda
```

### 数据处理
```bash
# 重要：任务书要求训练样本 ≤ 150 张，必须使用此脚本
python run_data_processing.py --input_dir ./data/raw --output_dir ./data/processed/my_product --max_train 150
```

### 评估
```bash
python run_evaluation.py --model all --category bottle --results_dir ./results
```

### UI 演示
```bash
python run_ui.py
# 访问 http://127.0.0.1:7860
```

### 类型检查
```bash
pyright modules/
pyright .
python -m pyright modules/
```

### 运行单个模块测试
```bash
# 测试数据处理
python -c "from modules.data_processing.dataset_formatter import MVTecFormatter; print('OK')"

# 测试评估指标
python -c "from modules.evaluation.metrics import MetricsEvaluator; print('OK')"

# 测试训练器
python -c "from modules.algorithm.trainer import AnomalyDetectionTrainer; print('OK')"
```

## 代码风格指南

### 通用规则
- **语言**：Python + 类型注解（asset/pyrightconfig.json: Python 3.8+，基础类型检查）
- **行长度**：目标 ≤120 字符，硬限制 150
- **编码**：UTF-8，CSV 导出使用 `utf-8-sig`

### 导入顺序（严格）
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

# 3. 本地导入（绝对导入，禁止相对导入）
from modules.data_processing.dataset_formatter import MVTecFormatter
```

### 类型注解
- 所有函数参数和返回值必须使用类型提示
- 使用 `Optional[X]` 而非 `X | None`
- 使用 `typing.Dict`、`typing.List` 而非内置类型
```python
def process_data(input_dir: str, max_samples: Optional[int] = 150) -> Dict[str, int]:
    ...
```

### 命名规范
| 类型 | 规范 | 示例 |
|------|-----------|---------|
| 类名 | PascalCase | `AnomalyDetectionTrainer` |
| 函数/变量 | snake_case | `compute_image_auroc` |
| 常量 | UPPER_SNAKE_CASE | `SUPPORTED_MODELS` |
| 私有方法 | 前缀 `_` | `_load_config` |
| 模块名 | snake_case | `dataset_formatter.py` |

### 错误处理
- 使用具体的异常类型，带有意义的错误信息
- 先捕获具体异常
```python
try:
    from anomalib.data import MVTec
except ImportError as e:
    print(f"❌ 错误: 未安装 anomalib。请运行: pip install anomalib>=2.0.0")
    raise
```

### 文档字符串
- 使用多行文档字符串，Google风格
- 包含 Args、Returns、Raises 部分
```python
def train(self) -> Dict[str, Any]:
    """
    训练模型。
    
    Returns:
        Dict: 训练结果，包含状态和轮次。
    """
```

### 类结构
```python
class AnomalyDetectionTrainer:
    """异常检测算法训练器"""
    
    def __init__(self, model_name: str, data_path: str):
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
- 使用带 emoji 的 print 输出用户可见消息
- 格式：`✅ 成功`、`❌ 错误`、`⚠️ 警告`、`📊 统计`、`🔄 处理中`
- 循环中使用 tqdm 显示进度条

### 文件组织
- 入口脚本（`run_*.py`）：根目录，逻辑最小化，从 modules 导入
- 核心模块（`modules/`）：所有业务逻辑
- 配置文件（`configs/`）：YAML 配置（Anomalib 2.x 格式）

### 数据处理
- 使用 `pathlib.Path` 处理所有路径
- 使用 `shutil.copy2` 复制文件
- 使用 `cv2.imread/imwrite` 处理图像
- 使用 `Path.resolve()` 规范化路径

### 性能考虑
- 循环中使用 tqdm 显示进度
- 尽可能使用 numpy 向量化操作
- 延迟导入重型库（如 anomalib）
- 设置随机种子

## Git 提交规范（Angular 协议）

### 格式
```
<类型>(<范围>): <主题>

[可选正文]

[可选脚注]
```

### 类型说明

| 类型 | 说明 | 示例 |
|------|------|------|
| `feat` | 新功能 | `feat(ui): 添加算法切换功能` |
| `fix` | 修复 bug | `fix(trainer): 修复显存溢出问题` |
| `docs` | 文档更新 | `docs: 更新 README` |
| `style` | 代码格式（不影响功能） | `style: 格式化代码` |
| `refactor` | 重构（不是修复也不是新功能） | `refactor: 重构模型配置结构` |
| `perf` | 性能优化 | `perf(patchcore): 启用预训练权重` |
| `test` | 测试相关 | `test: 添加单元测试` |
| `chore` | 构建/工具/依赖 | `chore: 更新依赖版本` |

### 示例（中文）

```bash
# 新功能
git commit -m "feat(ui): 添加算法切换下拉菜单

支持 PatchCore/Ganomaly/DRAEM 三种算法切换"

# 修复bug
git commit -m "fix(trainer): 修复 DRAEM 显存溢出问题

将 batch_size 从 32 降至 4"

# 重构
git commit -m "refactor(algorithm): 重构模型配置结构

提取 get_model_from_config 函数"

# 性能优化
git commit -m "perf(patchcore): 启用预训练权重

image_AUROC 从 85% 提升至 100%"
```

### 规则
- 主题行不超过 72 字符
- 使用命令式语气（add, fix, update, not added, fixed）
- 主题首字母小写，不以句号结尾
- 正文解释 **what** 和 **why**，不解释 **how**
- **禁止添加 Co-authored-by**：GitHub 会将其识别为共同贡献者，导致 contributors 列表混乱

## pyrightconfig.json

配置文件位于 `asset/pyrightconfig.json`，内容如下：

```json
{
  "pythonVersion": "3.8",
  "typeCheckingMode": "basic",
  "reportMissingImports": "warning",
  "reportGeneralTypeIssues": "warning"
}
```

## Agent 关键注意事项

1. **依赖版本**：本项目使用 anomalib **2.x**（已升级）
2. **数据路径**：`--data_path` 应指向类别文件夹的父目录（如 `./data` 而不是 `./data/region5`）
3. **YAML 配置路径**：`data.init_args.root` 应该是处理后的目录，`category` 单独设置
4. **NumPy 版本**：使用 numpy >= 1.24.0
5. **cv2 导入顺序**：必须在 anomalib 之前导入 cv2，避免 DLL 加载冲突
6. **Windows 多进程问题**：设置 `num_workers: 0` 避免 DataLoader 多进程崩溃
7. **Python 路径**：使用 `/c/Users/lx_hm/.conda/envs/anomalib/python.exe` 直接运行脚本

## 当前状态与已知问题

### 已完成 ✅
- PyTorch CUDA 配置（RTX 4060 + CUDA 11.8）
- **Anomalib 2.x 升级完成**
  - 更新 `trainer.py` 使用新 API
  - 更新 `demo.py` 使用 `Engine().predict()`
  - 更新 YAML 配置文件格式
  - 更新 `asset/requirements.txt`
  - 更新 `AGENTS.md` 文档
- 支持三种算法：Ganomaly, PatchCore, DRAEM
- **UI 权重加载修复**
  - 增强权重搜索逻辑，递归搜索 anomalib 2.x 嵌套目录结构
  - 为 Ganomaly 添加模型参数配置以匹配训练参数
  - 使用 `strict=False` 加载权重避免参数不匹配

### 升级后变化 ⚠️
- **API 完全变更**：从 0.7.x 的函数式 API 改为 2.x 的面向对象 API
- **配置格式变更**：使用 `class_path` 和 `init_args` 格式
- **训练引擎变更**：从 `pytorch_lightning.Trainer` 改为 `anomalib.engine.Engine`
- **推理方式变更**：从 `TorchInferencer` 改为 `Engine().predict()`
- **权重路径变更**：anomalib 2.x 保存路径为 `results/<model>/<ModelName>/MVTec/<category>/vN/weights/lightning/model.ckpt`

## 快速测试命令

```bash
# 测试环境是否正确配置
python -c "
import torch
import anomalib
import cv2
import gradio
print(f'PyTorch: {torch.__version__}')
print(f'Anomalib: {anomalib.__version__}')
print('All imports OK')
"

# 测试 Anomalib 2.x 导入
"C:\Users\lx_hm\.conda\envs\anomalib\python.exe" -c "
from anomalib.data import MVTec, Folder
from anomalib.engine import Engine
from anomalib.models import Patchcore, Draem, Ganomaly
print('Anomalib 2.x imports OK')
"
```

## Anomalib 2.x 参考文档

- [Anomalib 2.x 官方文档](https://anomalib.readthedocs.io/en/latest/)
- [迁移指南](https://anomalib.readthedocs.io/en/latest/markdown/get_started/migration.html)
- [15 分钟快速入门](https://anomalib.readthedocs.io/en/latest/markdown/get_started/anomalib.html)
