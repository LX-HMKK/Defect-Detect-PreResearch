# 第 1 阶段：止血 — 实施计划

> **面向执行代理：** 必须使用子技能 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 来逐任务实施此计划。步骤使用复选框 (`- [ ]`) 语法进行追踪。

**目标：** 解决审计中发现的 5 个最高优先级问题（C1 零测试、C2 BMP 伪装 PNG、C3 region4 缺失、H3 模块 API、M1 依赖漂移）

**架构：** 新增 7 个文件（测试套件、验证脚本、数据集注册表、固定依赖），修改 11 个现有文件（模块 API 导出、消费者导入路径、README、CHANGELOG）。不修改任何运行时逻辑。

**技术栈：** Python 3.10、pytest、numpy、Pillow、OpenCV

**预计总步骤数：** 约 70 步，分 10 个任务

---

## 文件结构

```
Defect-Detect-PreResearch/
├── tests/                              # 新增
│   ├── __init__.py                     # 包标记
│   ├── test_config.py                  # ConfigManager 单例、YAML、get()、get_model_config()
│   ├── test_metrics.py                 # AUROC、AUPR、pixel AUROC、PRO 合成数据测试
│   └── test_trainer_smoke.py           # 导入烟雾测试
├── tools/
│   └── validate_data.py                # 新增 — 独立数据验证脚本
├── data/
│   └── DATASET_REGISTRY.md             # 新增 — 数据集清单
├── requirements.txt                    # 新增 — 固定依赖
├── modules/
│   ├── algorithm/__init__.py           # 修改 — 公共 API 导出
│   ├── evaluation/__init__.py          # 修改 — 公共 API 导出
│   ├── data_processing/__init__.py     # 修改 — 公共 API 导出
│   └── ui/demo.py                      # 修改 — 更新导入路径
├── scripts/
│   ├── run_training.py                 # 修改 — 更新导入路径
│   ├── run_evaluation.py               # 修改 — 更新导入路径
│   ├── run_threshold.py                # 修改 — 更新导入路径
│   ├── run_data_processing.py          # 修改 — 更新导入路径
│   └── run_ui.py                       # 修改 — 更新导入路径
├── README.md                           # 修改 — anomalib 版本、数据集数量
└── CHANGELOG.md                        # 修改 — 添加条目
```

---

### 任务 1：创建 tests/ 包并编写 test_config.py

**文件：**
- 创建：`tests/__init__.py`
- 创建：`tests/test_config.py`

- [ ] **步骤 1：创建 tests 包标记文件**

```bash
mkdir -p D:\StudyWorks\3.2\Defect-Detect-PreResearch\tests
```

撰写 `tests/__init__.py`：

```python
# tests package
```

- [ ] **步骤 2：编写 test_config.py — 所有 6 个测试**

撰写 `tests/test_config.py`：

```python
"""Tests for modules/config/manager.py — ConfigManager singleton, YAML loading, get()."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from modules.config.manager import ConfigManager, get_config, reset_config, get


# —— 每个测试前重置单例 ——
# ConfigManager 使用模块级 _config_instance 作为单例。
# 若未显式重置，各测试间状态会相互影响，因此必须在每个测试函数开头调用一次。
def _reset():
    reset_config()


def test_singleton_identity():
    """两次调用 get_config() 返回同一个实例。"""
    _reset()
    a = get_config()
    b = get_config()
    assert a is b


def test_yaml_loads_data_section():
    """从 configs/config.yaml 加载后 get() 返回预期类型的值。"""
    _reset()
    batch = get('data.train_batch_size')
    assert isinstance(batch, int)
    assert batch > 0

    eval_batch = get('data.eval_batch_size')
    assert isinstance(eval_batch, int)
    assert eval_batch > 0

    workers = get('data.num_workers')
    assert isinstance(workers, int)
    assert workers >= 0


def test_get_nested_key():
    """嵌套键 get('data.train_batch_size') 应与 YAML 文件中的值一致。"""
    _reset()
    import yaml
    config_path = PROJECT_ROOT / 'configs' / 'config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f)

    expected = raw['data']['train_batch_size']
    assert get('data.train_batch_size') == expected


def test_get_model_config_returns_dict_patchcore():
    """get_model_config('patchcore') 返回包含 backbone/layers/coreset_sampling_ratio/num_neighbors/pre_trained 的字典。"""
    _reset()
    from modules.config import get_model_config
    cfg = get_model_config('patchcore')
    assert isinstance(cfg, dict)
    for key in ['backbone', 'layers', 'coreset_sampling_ratio', 'num_neighbors', 'pre_trained']:
        assert key in cfg, f"patchcore 配置缺少键: {key}"


def test_get_model_config_returns_dict_all():
    """fre/draem/padim 的 get_model_config() 均返回非空字典。"""
    _reset()
    from modules.config import get_model_config
    for model_name in ['fre', 'draem', 'padim']:
        cfg = get_model_config(model_name)
        assert isinstance(cfg, dict), f"{model_name} 配置不是字典"
        assert len(cfg) > 0, f"{model_name} 配置为空字典"


def test_threshold_default_fallback():
    """不存在的键返回提供的默认值或 None。"""
    _reset()
    # 未在 YAML 中显式设置时，threshold.default 使用代码中的默认值
    result = get('threshold.default', 0.5)
    assert result == 0.5

    result_none = get('completely.nonexistent.key.xyz', None)
    assert result_none is None


def test_missing_key_returns_none():
    """完全不存在且未提供默认值的键返回 None，不抛出异常。"""
    _reset()
    result = get('completely.nonexistent.key.xyz')
    assert result is None
```

- [ ] **步骤 3：运行 test_config.py 验证全部通过**

```bash
cd D:\StudyWorks\3.2\Defect-Detect-PreResearch && python -m pytest tests/test_config.py -v
```

预期：7 项全部 PASS。

> 注意：如果 pytest 未安装，先运行 `pip install pytest`。

- [ ] **步骤 4：提交**

```bash
git -C "D:\StudyWorks\3.2\Defect-Detect-PreResearch" add tests/__init__.py tests/test_config.py
git -C "D:\StudyWorks\3.2\Defect-Detect-PreResearch" commit -m "$(cat <<'EOF'
test: add config manager unit tests (C1)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### 任务 2：编写 test_metrics.py

**文件：**
- 创建：`tests/test_metrics.py`

- [ ] **步骤 1：编写 test_metrics.py — 全部 8 个测试**

撰写 `tests/test_metrics.py`：

```python
"""Tests for modules/evaluation/metrics.py — AUROC, AUPR, pixel AUROC, PRO with synthetic arrays."""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from modules.evaluation.metrics import MetricsEvaluator


evaluator = MetricsEvaluator()


def test_auroc_perfect_separation():
    """完美分离时 AUROC == 1.0。"""
    scores = np.array([0.1, 0.1, 0.1, 0.9, 0.9, 0.9], dtype=np.float64)
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
    result = evaluator.compute_image_auroc(scores, labels)
    assert abs(result - 1.0) < 1e-6, f"预期 1.0，实际 {result}"


def test_auroc_chance():
    """分数完全无信号时 AUROC == 0.5。"""
    scores = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float64)
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
    result = evaluator.compute_image_auroc(scores, labels)
    assert abs(result - 0.5) < 1e-6, f"预期 0.5，实际 {result}"


def test_auroc_single_class_guard():
    """仅有一个类别时返回 0.5，不抛出异常。"""
    scores = np.array([0.1, 0.1, 0.1], dtype=np.float64)
    labels = np.array([0, 0, 0], dtype=np.int32)
    result = evaluator.compute_image_auroc(scores, labels)
    assert result == 0.5


def test_aupr_perfect():
    """完美分离时 AUPR == 1.0。"""
    scores = np.array([0.1, 0.1, 0.1, 0.9, 0.9, 0.9], dtype=np.float64)
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
    result = evaluator.compute_image_aupr(scores, labels)
    assert abs(result - 1.0) < 1e-6, f"预期 1.0，实际 {result}"


def test_aupr_single_class_guard():
    """仅有一个类别时返回 0.0，不抛出异常。"""
    scores = np.array([0.1, 0.1], dtype=np.float64)
    labels = np.array([0, 0], dtype=np.int32)
    result = evaluator.compute_image_aupr(scores, labels)
    assert result == 0.0


def test_pixel_auroc_perfect():
    """预测与真实掩码完全一致时 Pixel AUROC == 1.0。"""
    anomaly_maps = np.array([
        [[0, 0], [0, 0]],
        [[1, 1], [1, 1]],
    ], dtype=np.float64)
    gt_masks = np.array([
        [[0, 0], [0, 0]],
        [[1, 1], [1, 1]],
    ], dtype=np.float64)
    result = evaluator.compute_pixel_auroc(anomaly_maps, gt_masks)
    assert abs(result - 1.0) < 1e-6, f"预期 1.0，实际 {result}"


def test_pixel_auroc_single_class():
    """gt_masks 全为零（仅有一个类别）时返回 0.5。"""
    anomaly_maps = np.zeros((2, 4, 4), dtype=np.float64)
    gt_masks = np.zeros((2, 4, 4), dtype=np.float64)
    result = evaluator.compute_pixel_auroc(anomaly_maps, gt_masks)
    assert result == 0.5


def test_pro_synthetic():
    """合成数据：第一张图有 3×3 异常区域，预测 80% 重叠；第二张图全部正常。PRO 应位于 (0.5, 1.0) 之间。"""
    anomaly_maps = np.zeros((2, 10, 10), dtype=np.float64)
    gt_masks = np.zeros((2, 10, 10), dtype=np.float64)

    # 第一张图：异常区域在 (0,0) 处为 3×3 = 1
    gt_masks[0, 0:3, 0:3] = 1.0
    # 预测覆盖重叠区域（约 80% 重叠）
    anomaly_maps[0, 0:3, 0:3] = 0.8

    result = evaluator.compute_pro(anomaly_maps, gt_masks)
    assert result > 0.5, f"PRO 应 > 0.5，实际 {result}"
    assert result < 1.0, f"PRO 应 < 1.0，实际 {result}"
```

- [ ] **步骤 2：运行 test_metrics.py 验证全部通过**

```bash
cd D:\StudyWorks\3.2\Defect-Detect-PreResearch && python -m pytest tests/test_metrics.py -v
```

预期：8 项全部 PASS。

- [ ] **步骤 3：提交**

```bash
git -C "D:\StudyWorks\3.2\Defect-Detect-PreResearch" add tests/test_metrics.py
git -C "D:\StudyWorks\3.2\Defect-Detect-PreResearch" commit -m "$(cat <<'EOF'
test: add metrics unit tests (C1)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### 任务 3：模块 __init__.py 导出（H3）

**文件：**
- 修改：`modules/algorithm/__init__.py`
- 修改：`modules/evaluation/__init__.py`
- 修改：`modules/data_processing/__init__.py`

- [ ] **步骤 1：修改 modules/algorithm/__init__.py**

将当前的空文件（1 行）替换为：

```python
"""
核心算法复现模块

提供异常检测算法的训练、评估和模型查询功能。
"""

from .trainer import (
    AnomalyDetectionTrainer,
    find_latest_checkpoint,
    get_model_from_config,
    get_datamodule_from_config,
    SUPPORTED_MODELS,
    MODEL_INFO,
)

__all__ = [
    'AnomalyDetectionTrainer',
    'find_latest_checkpoint',
    'get_model_from_config',
    'get_datamodule_from_config',
    'SUPPORTED_MODELS',
    'MODEL_INFO',
]
```

- [ ] **步骤 2：修改 modules/evaluation/__init__.py**

将当前的空文件（1 行）替换为：

```python
"""
指标评测模块

提供论文/综述要求的 4 个硬性指标计算功能：
图像级 AUROC、AUPR，像素级 Pixel AUROC、PRO。
"""

from .metrics import (
    MetricsEvaluator,
    AnomalyMetrics,
    load_and_evaluate,
)

__all__ = [
    'MetricsEvaluator',
    'AnomalyMetrics',
    'load_and_evaluate',
]
```

- [ ] **步骤 3：修改 modules/data_processing/__init__.py**

将当前的空文件（1 行）替换为：

```python
"""
数据集处理模块

将原始图片转换为 MVTec AD 标准格式。
"""

from .dataset_formatter import MVTecFormatter

__all__ = ['MVTecFormatter']
```

- [ ] **步骤 4：验证模块级导入正常工作**

```bash
cd D:\StudyWorks\3.2\Defect-Detect-PreResearch && python -c "
from modules.algorithm import AnomalyDetectionTrainer, SUPPORTED_MODELS, find_latest_checkpoint
from modules.evaluation import MetricsEvaluator, AnomalyMetrics, load_and_evaluate
from modules.data_processing import MVTecFormatter
print('All module-level imports OK')
"
```

预期：打印 "All module-level imports OK"（无 ImportError）。

- [ ] **步骤 5：提交**

```bash
git -C "D:\StudyWorks\3.2\Defect-Detect-PreResearch" add modules/algorithm/__init__.py modules/evaluation/__init__.py modules/data_processing/__init__.py
git -C "D:\StudyWorks\3.2\Defect-Detect-PreResearch" commit -m "$(cat <<'EOF'
refactor: add public API exports to module __init__.py files (H3)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### 任务 4：编写 test_trainer_smoke.py

**文件：**
- 创建：`tests/test_trainer_smoke.py`

- [ ] **步骤 1：编写 test_trainer_smoke.py — 全部 4 个测试**

撰写 `tests/test_trainer_smoke.py`：

```python
"""Smoke tests for modules/algorithm/trainer.py — imports, model list, error handling.
No GPU required. No actual training.
"""

import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def test_trainer_import():
    """可从 modules.algorithm 成功导入 AnomalyDetectionTrainer。"""
    from modules.algorithm import AnomalyDetectionTrainer
    assert AnomalyDetectionTrainer is not None


def test_supported_models_contains_all():
    """SUPPORTED_MODELS 包含全部 4 个模型。"""
    from modules.algorithm import SUPPORTED_MODELS
    assert set(SUPPORTED_MODELS) == {'fre', 'patchcore', 'draem', 'padim'}


def test_unsupported_model_raises_valueerror():
    """传入无效模型名称时抛出 ValueError，异常信息包含 model_name。"""
    from modules.algorithm import AnomalyDetectionTrainer
    with pytest.raises(ValueError, match='invalid_model_xyz'):
        AnomalyDetectionTrainer(
            model_name='invalid_model_xyz',
            data_path='.',
            category='test',
        )


def test_find_checkpoint_nonexistent():
    """不存在的目录返回 None。"""
    from modules.algorithm import find_latest_checkpoint
    result = find_latest_checkpoint('./nonexistent_dir_xyz', 'patchcore')
    assert result is None
```

- [ ] **步骤 2：运行 test_trainer_smoke.py 验证全部通过**

```bash
cd D:\StudyWorks\3.2\Defect-Detect-PreResearch && python -m pytest tests/test_trainer_smoke.py -v
```

预期：4 项全部 PASS。

- [ ] **步骤 3：提交**

```bash
git -C "D:\StudyWorks\3.2\Defect-Detect-PreResearch" add tests/test_trainer_smoke.py
git -C "D:\StudyWorks\3.2\Defect-Detect-PreResearch" commit -m "$(cat <<'EOF'
test: add trainer smoke tests (C1)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### 任务 5：更新消费者导入路径（H3 续）

**文件：**
- 修改：`modules/ui/demo.py`
- 修改：`scripts/run_training.py`
- 修改：`scripts/run_evaluation.py`
- 修改：`scripts/run_threshold.py`
- 修改：`scripts/run_data_processing.py`
- 修改：`scripts/run_ui.py`

- [ ] **步骤 1：更新 modules/ui/demo.py 的导入**

将第 185 行的：
```python
        from modules.algorithm.trainer import find_latest_checkpoint
```
替换为：
```python
        from modules.algorithm import find_latest_checkpoint
```

将第 248 行的：
```python
            from modules.algorithm.trainer import get_model_from_config
```
替换为：
```python
            from modules.algorithm import get_model_from_config
```

- [ ] **步骤 2：更新 scripts/run_training.py 的导入**

将第 139-144 行的：
```python
    from modules.algorithm.trainer import (
        AnomalyDetectionTrainer,
        SUPPORTED_MODELS,
        compare_models,
        find_latest_checkpoint,
    )
```
替换为：
```python
    from modules.algorithm import (
        AnomalyDetectionTrainer,
        SUPPORTED_MODELS,
        find_latest_checkpoint,
    )
    from modules.algorithm.trainer import compare_models
```

- [ ] **步骤 3：更新 scripts/run_evaluation.py 的导入**

将第 119 行的：
```python
    from modules.evaluation.metrics import load_and_evaluate
```
替换为：
```python
    from modules.evaluation import load_and_evaluate
```

- [ ] **步骤 4：更新 scripts/run_threshold.py 的导入**

将第 114 行的：
```python
    from modules.algorithm.trainer import find_latest_checkpoint
```
替换为：
```python
    from modules.algorithm import find_latest_checkpoint
```

将第 142 行的：
```python
    from modules.algorithm.trainer import AnomalyDetectionTrainer
```
替换为：
```python
    from modules.algorithm import AnomalyDetectionTrainer
```

- [ ] **步骤 5：更新 scripts/run_data_processing.py 的导入**

将第 101 行的：
```python
    from modules.data_processing.dataset_formatter import MVTecFormatter
```
替换为：
```python
    from modules.data_processing import MVTecFormatter
```

- [ ] **步骤 6：更新 scripts/run_ui.py 的导入**

检查 scripts/run_ui.py：它从 `modules.ui.demo` 导入 — 不需要修改（demo 不是模块级 API 的一部分）。确认第 105 行无需更改后跳过。

- [ ] **步骤 7：验证所有导入路径**

```bash
cd D:\StudyWorks\3.2\Defect-Detect-PreResearch && python -c "
import sys; sys.path.insert(0, '.')
# 验证旧的导入方式不再被使用（检查是否存在残留的直接子模块导入）
from modules.algorithm import AnomalyDetectionTrainer, SUPPORTED_MODELS, find_latest_checkpoint, get_model_from_config
from modules.evaluation import MetricsEvaluator, AnomalyMetrics, load_and_evaluate
from modules.data_processing import MVTecFormatter
print('All updated imports OK')
"
```

预期：打印 "All updated imports OK"。

- [ ] **步骤 8：提交**

```bash
git -C "D:\StudyWorks\3.2\Defect-Detect-PreResearch" add modules/ui/demo.py scripts/run_training.py scripts/run_evaluation.py scripts/run_threshold.py scripts/run_data_processing.py
git -C "D:\StudyWorks\3.2\Defect-Detect-PreResearch" commit -m "$(cat <<'EOF'
refactor: update consumer imports to use module-level API (H3)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### 任务 6：编写 tools/validate_data.py（C2）

**文件：**
- 创建：`tools/validate_data.py`

- [ ] **步骤 1：编写 validate_data.py**

撰写 `tools/validate_data.py`：

```python
#!/usr/bin/env python
"""数据目录验证脚本 — 检查魔法字节、可读性、目录结构和类别分布。

独立运行，不依赖项目的 modules/ 或 torch/anomalib。
用法: python tools/validate_data.py [--data-root ./data]
"""

import os
import sys
import struct
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# —— 已知数据集清单 ——
KNOWN_DATASETS = {
    'bottle', 'carpet', 'region1', 'region2', 'region3', 'region5',
}

# README 中引用的数据集
REFERENCED_DATASETS = {'bottle', 'carpet', 'region1', 'region2', 'region3', 'region4', 'region5'}

ERRORS: list[str] = []
WARNINGS: list[str] = []
INFOS: list[str] = []


def error(msg: str) -> None:
    ERRORS.append(msg)
    print(f"  [ERROR] {msg}")


def warn(msg: str) -> None:
    WARNINGS.append(msg)
    print(f"  [WARN]  {msg}")


def info(msg: str) -> None:
    INFOS.append(msg)
    print(f"  [INFO]  {msg}")


# —— 检查 1：魔法字节 vs 扩展名 ——
def check_magic_bytes(data_root: Path) -> None:
    print("\n[1/7] 检查魔法字节 vs 扩展名...")
    png_files = list(data_root.rglob("*.png"))
    bmp_count = 0
    png_count = 0
    other_count = 0

    for fpath in png_files:
        try:
            with open(fpath, "rb") as fh:
                magic = fh.read(2)
            if magic == b"\x89P":
                png_count += 1
            elif magic == b"BM":
                bmp_count += 1
                error(f"BMP 文件伪装为 .png 后缀: {fpath}")
            else:
                other_count += 1
                error(f"未知格式 (magic={magic!r}): {fpath}")
        except OSError as e:
            error(f"无法读取文件 {fpath}: {e}")

    info(f"统计: {png_count} 个真实 PNG, {bmp_count} 个 BMP 伪装为 PNG, {other_count} 个未知")


# —— 检查 2：空文件 ——
def check_empty_files(data_root: Path) -> None:
    print("\n[2/7] 检查空文件...")
    empty_count = 0
    for fpath in data_root.rglob("*"):
        if fpath.is_file():
            try:
                if fpath.stat().st_size == 0:
                    error(f"空文件: {fpath}")
                    empty_count += 1
            except OSError as e:
                warn(f"无法获取文件大小 {fpath}: {e}")
    info(f"发现 {empty_count} 个空文件")


# —— 检查 3：可读性（PIL + cv2） ——
def check_readability(data_root: Path) -> None:
    print("\n[3/7] 检查图片可读性 (PIL + cv2)...")
    try:
        from PIL import Image
    except ImportError:
        warn("Pillow 未安装，跳过 PIL 可读性检查")
        return
    try:
        import cv2
    except ImportError:
        warn("OpenCV 未安装，跳过 cv2 可读性检查")
        return

    image_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    fail_count = 0
    for fpath in data_root.rglob("*"):
        if fpath.suffix.lower() in image_exts:
            try:
                img = Image.open(fpath)
                img.verify()
            except Exception as e:
                error(f"PIL 无法打开: {fpath} ({e})")
                fail_count += 1
                continue
            try:
                img_cv = cv2.imread(str(fpath))
                if img_cv is None:
                    error(f"cv2.imread 返回 None: {fpath}")
                    fail_count += 1
            except Exception as e:
                error(f"cv2 无法打开: {fpath} ({e})")
                fail_count += 1
    if fail_count == 0:
        info("所有图片均通过 PIL 和 cv2 可读性检查")


# —— 检查 4：目录结构 ——
def check_directory_structure(data_root: Path) -> None:
    print("\n[4/7] 检查 MVTec AD 目录结构...")
    for d in sorted(data_root.iterdir()):
        if d.is_file() or d.name.startswith('.'):
            continue

        train_good = d / "train" / "good"
        if not train_good.exists():
            warn(f"{d.name}: 缺少 train/good/ 目录")

        test_dir = d / "test"
        if not test_dir.exists():
            warn(f"{d.name}: 缺少 test/ 目录")
            continue

        for sub in sorted(test_dir.iterdir()):
            if sub.is_dir() and sub.name != "good":
                gt_dir = d / "ground_truth" / sub.name
                if not gt_dir.exists():
                    warn(f"{d.name}: test/{sub.name}/ 存在但 ground_truth/{sub.name}/ 缺失")


# —— 检查 5：类别分布 ——
def check_class_distribution(data_root: Path) -> None:
    print("\n[5/7] 检查测试集类别分布...")
    for d in sorted(data_root.iterdir()):
        if d.is_file() or d.name.startswith('.'):
            continue
        test_dir = d / "test"
        if not test_dir.exists():
            continue

        for sub in sorted(test_dir.iterdir()):
            if sub.is_dir():
                count = sum(1 for f in sub.iterdir() if f.is_file())
                if sub.name != "good" and count < 5:
                    warn(f"{d.name}/test/{sub.name}: 仅有 {count} 个样本 (< 5)，统计不可靠")


# —— 检查 6：缺失的引用数据集 ——
def check_missing_referenced(data_root: Path) -> None:
    print("\n[6/7] 检查 README 中引用但磁盘不存在的数据集...")
    on_disk = {d.name for d in data_root.iterdir() if d.is_dir() and not d.name.startswith('.')}
    missing_refs = REFERENCED_DATASETS - on_disk
    for ds in sorted(missing_refs):
        warn(f"README 引用的数据集在磁盘上不存在: {ds}")

    unknown = on_disk - REFERENCED_DATASETS
    for ds in sorted(unknown):
        info(f"磁盘存在但未在已知清单中: {ds}")


# —— 检查 7：未知目录 ——
def check_unknown_dirs(data_root: Path) -> None:
    print("\n[7/7] 检查未知目录...")
    on_disk = {d.name for d in data_root.iterdir() if d.is_dir() and not d.name.startswith('.')}
    unknown = on_disk - REFERENCED_DATASETS
    if unknown:
        for ds in sorted(unknown):
            info(f"未知目录: {ds}")
    else:
        info("所有目录均在已知清单中")


# —— 报告生成 ——
def write_report(data_root: str, output_dir: str) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = Path(output_dir) / f"data_validation_{timestamp}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# 数据验证报告",
        f"",
        f"**生成时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**数据目录:** {data_root}",
        f"**退出码:** {'0 (无错误)' if not ERRORS else '1 (存在错误)'}",
        f"",
        f"## 摘要",
        f"",
        f"| 级别 | 数量 |",
        f"|------|------|",
        f"| ERROR | {len(ERRORS)} |",
        f"| WARN  | {len(WARNINGS)} |",
        f"| INFO  | {len(INFOS)} |",
        f"",
    ]

    if ERRORS:
        lines.append("## ERROR")
        lines.append("")
        for e in ERRORS:
            lines.append(f"- {e}")
        lines.append("")

    if WARNINGS:
        lines.append("## WARN")
        lines.append("")
        for w in WARNINGS:
            lines.append(f"- {w}")
        lines.append("")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"\n[SAVE] 报告已保存: {report_path}")


# —— 主入口 ——
def main() -> int:
    parser = argparse.ArgumentParser(description="验证 data/ 目录的数据完整性")
    parser.add_argument("--data-root", default="./data", help="数据目录路径 (默认: ./data)")
    parser.add_argument("--output-dir", default="./results", help="报告输出目录 (默认: ./results)")
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    if not data_root.exists():
        print(f"[FATAL] 数据目录不存在: {data_root}")
        return 1

    print(f"数据验证 — {data_root}")
    print(f"=" * 60)

    check_magic_bytes(data_root)
    check_empty_files(data_root)
    check_readability(data_root)
    check_directory_structure(data_root)
    check_class_distribution(data_root)
    check_missing_referenced(data_root)
    check_unknown_dirs(data_root)

    write_report(str(data_root), args.output_dir)

    print(f"\n{'=' * 60}")
    print(f"验证完成: {len(ERRORS)} ERROR, {len(WARNINGS)} WARN, {len(INFOS)} INFO")
    return 1 if ERRORS else 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **步骤 2：运行 validate_data.py**

```bash
cd D:\StudyWorks\3.2\Defect-Detect-PreResearch && python tools/validate_data.py
```

预期：报告 BMP 伪装 PNG 错误（已知）、缺少 region4 警告（已知）。退出码为 1（BMP 错误是真实存在的）。

- [ ] **步骤 3：确认报告已生成**

```bash
ls D:\StudyWorks\3.2\Defect-Detect-PreResearch\results\data_validation_*.md | tail -1
```

预期：显示刚生成的文件路径。

- [ ] **步骤 4：提交**

```bash
git -C "D:\StudyWorks\3.2\Defect-Detect-PreResearch" add tools/validate_data.py
git -C "D:\StudyWorks\3.2\Defect-Detect-PreResearch" commit -m "$(cat <<'EOF'
feat: add data validation script (C2)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### 任务 7：创建 data/DATASET_REGISTRY.md 并修正 README（C3）

**文件：**
- 创建：`data/DATASET_REGISTRY.md`
- 修改：`README.md`

- [ ] **步骤 1：编写 DATASET_REGISTRY.md**

撰写 `data/DATASET_REGISTRY.md`：

```markdown
# 数据集注册表

所有已存在及引用的数据集清单。最后更新：2026-06-12。

| 数据集 | 来源 | 状态 | 训练集 | 测试集(正常) | 测试集(缺陷) | 已知问题 |
|--------|------|------|--------|-------------|-------------|----------|
| bottle | MVTec AD | 完整 | 209 | 20 | 63 | 无 |
| carpet | MVTec AD | 完整 | 280 | 28 | 89 | 无 |
| region1 | 企业数据 | 不完整 | 待统计 | 91 | 7 (lb=1, ps=4, py=1, tl=1) | BMP 文件伪装为 .png 后缀；严重类别不平衡 (3 个类别仅 1 个样本) |
| region2 | 企业数据 | 不完整 | 待统计 | 91 | 15 (lb=2, ps=9, py=3, tl=1) | BMP 文件伪装为 .png 后缀；严重类别不平衡 (tl 仅 1 个样本) |
| region3 | 企业数据 | 基本完整 | 待统计 | 150 | 17 (lb=9, ps=2, py=1, tl=5) | 轻微类别不平衡 (ps 仅 2 个样本) |
| region4 | — | **缺失** | — | — | — | 磁盘上从未存在此目录。README 原引用 region1-5，实际仅有 4 个自定义数据集。 |
| region5 | 企业数据 | 不完整 | 待统计 | 91 | 23 (lb=9, ps=4, py=8, tl=2) | BMP 文件伪装为 .png 后缀；中度类别不平衡 |

## 数据格式说明

- 所有数据集遵循 MVTec AD 标准目录布局：
  - `train/good/` — 训练集（仅正常样本）
  - `test/good/` — 测试集（正常样本）
  - `test/<defect>/` — 测试集（异常样本，按缺陷类型分目录）
  - `ground_truth/<defect>/` — 像素级标注掩码
- region1/2/5 中的图片文件虽然扩展名为 `.png`，但实际格式为 BMP（Windows 位图）。
  OpenCV 可通过内容检测自动识别格式，因此不会影响当前训练流程。
  若将来切换到依赖扩展名的加载器，需先转换为真正的 PNG 格式。
- 训练集样本数待重新统计（当前为空或需用 `tools/validate_data.py` 扫描）。

## 外部数据集

| 目录 | 说明 |
|------|------|
| `datasets/dtd/` | Describable Textures Dataset — DRAEM 算法用于生成合成异常纹理。需手动下载。 |
```

- [ ] **步骤 2：修正 README.md — anomalib 版本**

将第 207 行：
```
pip install anomalib>=2.0.0
```
替换为：
```
pip install anomalib==2.3.0
```

将第 213 行：
```
- anomalib >= 2.0.0
```
替换为：
```
- anomalib == 2.3.0（固定版本，因 trainer.py 中的 monkey-patch 兼容层依赖此版本。升级 anomalib 前需先更新这些补丁。）
```

- [ ] **步骤 3：修正 README.md — 数据集数量引用**

查找 README 中所有 region1-5 或 "5 个区域" 之类对自定义数据集数量的引用，逐一修正为 4 个 (region1-3, region5)。此步骤需在 README 中全文搜索。

```bash
cd D:\StudyWorks\3.2\Defect-Detect-PreResearch && grep -n "region" README.md
```

若发现 "region1-5" 或暗示 5 个自定义数据集的文本，逐一修正。

- [ ] **步骤 4：提交**

```bash
git -C "D:\StudyWorks\3.2\Defect-Detect-PreResearch" add data/DATASET_REGISTRY.md README.md
git -C "D:\StudyWorks\3.2\Defect-Detect-PreResearch" commit -m "$(cat <<'EOF'
docs: add dataset registry and fix README references (C3)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### 任务 8：创建 requirements.txt 并最终修正 README（M1）

**文件：**
- 创建：`requirements.txt`
- 修改：`README.md`（如果是第一次修正）

- [ ] **步骤 1：创建 requirements.txt**

撰写 `requirements.txt`：

```
anomalib==2.3.0
pytorch-lightning==1.9.5
lightning==2.3.0
torch>=2.0.0
opencv-python-headless>=4.8.0
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.2.0
pandas>=1.5.0
tqdm>=4.65.0
PyYAML>=6.0
```

- [ ] **步骤 2：验证依赖无冲突**

```bash
pip install -r D:\StudyWorks\3.2\Defect-Detect-PreResearch\requirements.txt --dry-run 2>&1 | head -20
```

预期：无冲突（或仅报告已安装的包）。

- [ ] **步骤 3：提交**

```bash
git -C "D:\StudyWorks\3.2\Defect-Detect-PreResearch" add requirements.txt
git -C "D:\StudyWorks\3.2\Defect-Detect-PreResearch" commit -m "$(cat <<'EOF'
build: add pinned requirements.txt (M1)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### 任务 9：更新 CHANGELOG.md

**文件：**
- 修改：`CHANGELOG.md`

- [ ] **步骤 1：在 CHANGELOG 顶部添加第 1 阶段条目**

在 `CHANGELOG.md` 的最新条目之上插入：

```markdown
## 2026-06-12 — Phase 1: 止血

- **test:** 新增测试套件 — config 管理器单元测试、metrics 单元测试、trainer 烟雾测试 (C1)
- **feat:** 新增数据验证脚本 `tools/validate_data.py`，检测 BMP 伪装 PNG、目录结构、类别分布 (C2)
- **docs:** 新增 `data/DATASET_REGISTRY.md` 数据集注册表，记录 region4 缺失及各数据集已知问题 (C3)
- **refactor:** 为 `modules/algorithm`、`modules/evaluation`、`modules/data_processing` 添加公共 API 导出 (H3)
- **refactor:** 更新消费者导入路径为模块级 API (H3)
- **build:** 新增 `requirements.txt` 固定依赖版本 (M1)
- **docs:** README 修正 anomalib 版本 `>=2.0.0` → `==2.3.0`，修正数据集数量引用 (M1, C3)
```

- [ ] **步骤 2：提交**

```bash
git -C "D:\StudyWorks\3.2\Defect-Detect-PreResearch" add CHANGELOG.md
git -C "D:\StudyWorks\3.2\Defect-Detect-PreResearch" commit -m "$(cat <<'EOF'
docs: add Phase 1 changelog entry

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### 任务 10：全面验证

- [ ] **步骤 1：运行全部测试套件**

```bash
cd D:\StudyWorks\3.2\Defect-Detect-PreResearch && python -m pytest tests/ -v
```

预期：全部约 19 项测试 PASS。

- [ ] **步骤 2：运行数据验证脚本**

```bash
cd D:\StudyWorks\3.2\Defect-Detect-PreResearch && python tools/validate_data.py
```

预期：报告已知错误和警告。退出码 1（BMP 错误）。

- [ ] **步骤 3：验证导入完整性**

```bash
cd D:\StudyWorks\3.2\Defect-Detect-PreResearch && python -c "
from modules.algorithm import AnomalyDetectionTrainer, SUPPORTED_MODELS, find_latest_checkpoint, get_model_from_config, get_datamodule_from_config
from modules.evaluation import MetricsEvaluator, AnomalyMetrics, load_and_evaluate
from modules.data_processing import MVTecFormatter
print('All imports OK')
"
```

预期：打印 "All imports OK"。

- [ ] **步骤 4：查看最终 git log**

```bash
git -C "D:\StudyWorks\3.2\Defect-Detect-PreResearch" log --oneline -12
```

预期：显示约 9-10 个第 1 阶段提交，从 `test: add config manager unit tests` 开始，到 `docs: add Phase 1 changelog entry` 结束。

- [ ] **步骤 5：提交验证通过标记（如有 .phase1-done 文件或类似约定）**

若项目有标记约定则执行；否则跳过此步骤。

---

## 验证完成检查清单

实施完毕后逐项确认：

- [ ] `pytest tests/ -v` — 全部 PASS
- [ ] `python tools/validate_data.py` — 报告已知错误（BMP 文件名、region4），退出码 1
- [ ] 模块导入 `modules.algorithm`、`modules.evaluation`、`modules.data_processing` — 无 ImportError
- [ ] `requirements.txt` 存在且内容正确
- [ ] README 中 anomalib 版本已固定为 `==2.3.0`
- [ ] `data/DATASET_REGISTRY.md` 存在且记录 region4 缺失
- [ ] CHANGELOG 包含第 1 阶段条目
