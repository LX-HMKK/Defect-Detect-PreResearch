# 配置系统说明文档

## 概述

本项目已统一配置文件管理系统，将分散在代码中的硬编码参数集中到 `config.yaml` 文件中管理。

## 配置文件结构

### 主配置文件

**文件**: `./config.yaml`

包含以下配置段：

### 1. 模型架构配置 (`models`)

```yaml
models:
  patchcore:
    backbone: wide_resnet50_2
    layers: [layer2, layer3]
    pre_trained: true
    coreset_sampling_ratio: 0.1
    num_neighbors: 9
  
  fre:
    backbone: resnet50
    layer: layer3
    pre_trained: true
    pooling_kernel_size: 2
    input_dim: 65536
    latent_dim: 220
  
  draem:
    beta: [0.1, 1.0]
    enable_sspcab: false
```

### 2. 数据配置 (`data`)

```yaml
data:
  image_size: [256, 256]
  train_batch_size: 32
  eval_batch_size: 32
  num_workers: 0
  draem_batch_size: 4  # DRAEM 显存占用大
```

### 3. 阈值配置 (`threshold`)

```yaml
threshold:
  default: 0.5
  result_file_template: "./results/comparison/{model}_{dataset}_results.json"
  threshold_key: "optimal_threshold"
  dataset_defaults:
    bottle: 0.5
    carpet: 0.53
```

### 4. 训练配置 (`training`)

```yaml
training:
  epochs:
    patchcore: 1
    fre: 50
    draem: 200
  optimizer:
    lr: 0.0001
    weight_decay: 0.00001
  seed: 42
  accelerator: auto
```

### 5. 评估配置 (`evaluation`)

```yaml
evaluation:
  threshold_search:
    min: 0.0
    max: 1.0
    steps: 100
```

## 配置管理模块

### 使用方式

#### 1. 获取配置值

```python
from modules.config import get

# 获取图像尺寸
image_size = get('data.image_size')  # [256, 256]

# 获取默认 batch size
batch_size = get('data.train_batch_size', 32)  # 带默认值
```

#### 2. 获取阈值（智能选择）

```python
from modules.config import get_threshold

# 自动选择最优阈值
# 优先级: 训练结果文件 > 配置文件默认值 > 代码默认值
threshold = get_threshold('patchcore', 'carpet')  # 返回 0.53
```

#### 3. 获取模型配置

```python
from modules.config import get_model_config

# 获取 PatchCore 配置
config = get_model_config('patchcore')
# 返回: {'backbone': 'wide_resnet50_2', 'layers': ['layer2', 'layer3'], ...}
```

#### 4. 获取数据配置

```python
from modules.config import get_data_config

# 获取通用数据配置
data_cfg = get_data_config()

# 获取 DRAEM 专用配置（自动调整 batch size）
draem_cfg = get_data_config('draem')  # batch_size = 4
```

## 优先级规则

### 配置加载优先级（从高到低）

1. **运行时传入参数**（函数参数）
2. **anomalib YAML 配置文件**（如 `configs/patchcore.yaml`）
3. **主配置文件**（`config.yaml`）
4. **代码硬编码默认值**

### 阈值选择优先级

1. **训练结果文件**（`results/comparison/{model}_{dataset}_results.json` 中的 `optimal_threshold`）
2. **配置文件中的数据集默认值**（`config.yaml` 中 `threshold.dataset_defaults.{dataset}`）
3. **全局默认阈值**（`config.yaml` 中 `threshold.default`）
4. **代码默认值**（0.5）

## 已移除的硬编码

### 训练模块 (`modules/algorithm/trainer.py`)

- ✅ 模型架构参数（backbone、layers、batch size 等）
- ✅ 训练 epoch 默认值
- ✅ 阈值搜索范围和步数
- ✅ 默认阈值 0.5

### UI 模块 (`modules/ui/demo.py`)

- ✅ 硬编码的阈值配置（已改为动态读取）
- ✅ 默认阈值 0.5
- ✅ 模型配置默认值

### 评估模块 (`modules/evaluation/metrics.py`)

- ⚠️ 保留数学默认值（如 AUROC 无法计算时返回 0.5，这是随机猜测的基准值）

## 示例：添加新数据集

1. **在 `config.yaml` 中添加阈值默认值**：

```yaml
threshold:
  dataset_defaults:
    bottle: 0.5
    carpet: 0.53
    my_new_dataset: 0.45  # 新增
```

2. **运行训练**（自动计算最优阈值）：

```bash
python run_training.py -m patchcore -c my_new_dataset -d ./data
```

3. **阈值会自动保存**到结果文件，后续优先使用计算值

## 示例：修改模型参数

直接编辑 `config.yaml`：

```yaml
models:
  patchcore:
    backbone: resnet50  # 改为 resnet50
    num_neighbors: 5    # 改为 5
```

重新运行训练即可生效。

## 注意事项

1. **config.yaml 不存在时**：系统会使用内置默认配置
2. **配置文件格式错误**：系统会打印警告并使用默认配置
3. **阈值更新**：重新训练模型后，最优阈值会自动更新到结果文件
4. **多数据集支持**：每个数据集可以有独立的阈值配置

## 配置文件备份

建议定期备份 `config.yaml`，特别是在修改重要参数后。
