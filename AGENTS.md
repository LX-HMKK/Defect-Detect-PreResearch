# AGENTS.md - Agent 开发指南

这是一个基于 **anomalib** 库的工业图像异常检测系统，实现了 3 种算法：
- **AutoEncoder**（基于重构）
- **PatchCore**（基于特征建模 - 工业应用最佳）
- **DRAEM**（基于自监督学习）

## 项目结构

```
Defect-Detect-PreResearch/
├── modules/
│   ├── data_processing/    # MVTec AD 数据格式转换
│   ├── algorithm/          # 模型训练与推理
│   ├── evaluation/         # 指标计算
│   └── ui/                # Gradio 演示界面
├── configs/                # YAML 算法配置
├── data/
│   ├── raw/               # 原始企业数据
│   └── processed/         # MVTec AD 格式处理后数据
├── results/               # 训练结果
├── run_*.py              # 入口脚本（根目录）
└── requirements.txt
```

## 命令

### 环境配置（Miniforge）
```bash
# 创建虚拟环境
"C:\ProgramData\miniforge3\Scripts\conda.exe" create -n anomalib python=3.10 -y

# 通过 conda 安装 PyTorch CUDA 版本（RTX 4060 需要 CUDA 11.8）
"C:\ProgramData\miniforge3\Scripts\conda.exe" install -n anomalib -c pytorch -c conda-forge pytorch torchvision pytorch-lightning lightning scikit-learn scipy pandas pillow opencv tqdm pyyaml -y

# 通过 pip 安装 anomalib 0.7.0 和其他依赖（注意版本约束）
"C:\Users\lx_hm\.conda\envs\anomalib\python.exe" -m pip install anomalib==0.7.0 gradio omegaconf "numpy<2" "setuptools==68.0.0" packaging albumentations==1.3.1
```

### 直接运行（推荐）

**修改 `configs/patchcore.yaml` 中的 `run` 配置块，然后直接运行：**

```yaml
run:
  model: patchcore              # 算法：patchcore / autoencoder / draem
  data_path: ./data            # 数据目录（指向region5的父目录）
  category: region5            # 类别名称
  device: cuda                 # cuda / cpu
  output_dir: ./results        # 输出目录
```

```bash
# 直接运行（从 patchcore.yaml 读取配置）
python run_training.py
```

### 数据处理
```bash
# 重要：任务书要求训练样本 ≤ 150 张，必须使用此脚本
python run_data_processing.py --input_dir ./data/raw --output_dir ./data/processed/my_product --max_train 150
```

### 训练（命令行参数）
```bash
# 训练所有算法
python run_training.py --model all --category region5 --data_path ./data --device cuda

# 训练单个算法
python run_training.py --model patchcore --category region5 --data_path ./data --device cuda
```

### 评估
```bash
python run_evaluation.py --model all --category region5 --results_dir ./results
```

### UI 演示
```bash
python run_ui.py
# 访问 http://127.0.0.1:7860
```

### 评估
```bash
python run_evaluation.py --model all --category my_product --results_dir ./results
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
- **语言**：Python + 类型注解（pyrightconfig.json: Python 3.8+，基础类型检查）
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
    from anomalib.config import get_configurable_parameters
except ImportError as e:
    print(f"❌ 错误: 未安装 anomalib。请运行: pip install anomalib")
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
- 配置文件（`configs/`）：YAML 配置

### 数据处理
- 使用 `pathlib.Path` 处理所有路径
- 使用 `shutil.copy2` 复制文件
- 使用 `cv2.imread/imwrite` 处理图像
- 使用 `Path.resolve()` 规范化路径

### 性能考虑
- 循环中使用 tqdm 显示进度
- 尽可能使用 numpy 向量化操作
- 延迟导入重型库（如 anomalib）
- 设置随机种子：`seed_everything(42)`

## pyrightconfig.json

```json
{
  "pythonVersion": "3.8",
  "typeCheckingMode": "basic",
  "reportMissingImports": "warning",
  "reportGeneralTypeIssues": "warning"
}
```

## Agent 关键注意事项

1. **依赖版本**：本项目使用 anomalib 0.7.0（不是 1.x 或 2.x），因为 API 兼容性问题
2. **数据路径**：`--data_path` 应指向类别文件夹的父目录（如 `./data` 而不是 `./data/region5`）
3. **YAML 配置路径**：`dataset.path` 应该是处理后的目录，`category` 单独设置
4. **NumPy 版本**：必须使用 `numpy<2`，因为 torch 兼容性
5. **Setuptools 版本**：使用 `setuptools==68.0.0` 以保证 pkg_resources 兼容性
6. **cv2 导入顺序**：必须在 anomalib 之前导入 cv2，避免 DLL 加载冲突
7. **OpenVINO 警告**：已通过 stdout 重定向压制，不影响功能
8. **Windows 多进程问题**：在 `setup()` 中设置 `config.dataset.num_workers = 0` 避免 DataLoader 多进程崩溃
9. **Python 路径**：使用 `/c/Users/lx_hm/.conda/envs/anomalib/python.exe` 直接运行脚本

## 当前状态与已知问题

### 已完成 ✅
- PyTorch CUDA 配置（RTX 4060 + CUDA 11.8）
- YAML 配置整合（可直接 `python run_training.py` 运行）
- OpenVINO/wandb 警告压制
- PatchCore 内存库 bug 修复（anomalib 0.7.0 内部未实现）
- **PatchCore AUROC=0.5 问题已修复** (2026-03-27)
  - 问题原因: Lightning Trainer.test() 未正确使用 memory bank
  - 解决方案: 使用直接推理 + 随机采样代替慢速 coreset 选择
  - 验证结果: MVTec AD bottle 数据集 AUROC=99.76%, AUPR=99.92%

### 已知问题 ⚠️
- **企业数据 AUROC 仍为 0.5**: 
  - 原因分析: 企业数据（region5）图像尺寸（409x1421）与 MVTec AD 标准（256x256）不同
  - 状态: 需要进一步调查，可能是图像预处理问题
- **训练样本超限**: region5 有 200 张训练样本，违反任务书 ≤150 限制
- **建议**: 企业数据需通过 `run_data_processing.py` 处理并限制 150 张

## PatchCore 修复详情

### 问题
- 训练完成但评估 AUROC=0.5（随机基线）

### 根因
1. **内存库未构建**: `PatchcoreLightning.training_step()` 返回 None，导致 `training_epoch_end` 不被调用，memory bank 保持为空 `tensor([])`
2. **Lightning test() 问题**: 即使手动构建了 memory bank，通过 `trainer.test()` 调用时，Lightning 的 test_step 没有正确使用 memory bank

### 解决方案
1. 在 `trainer.train()` 末尾手动构建 memory bank:
   ```python
   # 收集训练集特征
   embeddings_list = []
   for batch in train_loader:
       img = batch["image"]
       features = model.model.feature_extractor(img)
       features = {layer: model.model.feature_pooler(f) for layer, f in features.items()}
       embedding = model.model.generate_embedding(features)
       embeddings_list.append(embedding)
   
   all_embeddings = torch.cat(embeddings_list, dim=0)
   all_embeddings = model.model.reshape_embedding(all_embeddings)
   
   # 随机采样代替慢速 coreset 选择
   indices = torch.randperm(all_embeddings.shape[0])[:1000]
   memory_bank = all_embeddings[indices].clone()
   model.model.memory_bank = memory_bank
   ```

2. 在 `evaluate()` 中对 PatchCore 使用直接推理:
   ```python
   # PatchCore: 使用直接推理而不是 Lightning Trainer.test()
   if self.model_name == 'patchcore':
       self.results = self._evaluate_patchcore_direct()
   ```

3. 添加 `_evaluate_patchcore_direct()` 方法直接调用模型:
   ```python
   output = model.model(img)  # 返回 (anomaly_map, image_scores) 元组
   ```

### 验证结果 (MVTec AD bottle)
- **image_AUROC**: 99.76%
- **image_AUPR**: 99.92%
- **pixel_AUROC**: 98.38%
- **pixel_PRO**: 98.38%

## 快速测试命令

```bash
# 测试环境是否正确配置
python -c "
import torch; import anomalib; import cv2; import gradio
print(f'PyTorch: {torch.__version__}')
print(f'Anomalib: {anomalib.__version__}')
print('All imports OK')
"
```
