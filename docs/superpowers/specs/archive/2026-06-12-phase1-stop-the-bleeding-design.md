# 第 1 阶段设计：止血

**项目:** Defect-Detect-PreResearch
**日期:** 2026-06-12
**来源:** Ultracode 全面审计 (wf_18c6a0d6-bd3)
**范围:** 审计报告中的 C1、C2、C3、H3、M1

---

## 目标

处理审计中发现的 5 个最高优先级问题：

| ID | 问题 | 方案 |
|----|------|------|
| C1 | 零自动化测试 | 编写最小测试套件（指标、配置、训练烟雾测试） |
| C2 | BMP 文件伪装为 PNG 后缀 | 编写验证脚本；暂不批量转换（待数据来源方确认） |
| C3 | region4 数据缺失 | 在 DATASET_REGISTRY.md 中记录；修正 README 引用 |
| H3 | 空 __init__.py / 跨模块耦合 | 为 3 个子模块添加公共 API 导出 |
| M1 | 依赖版本不一致 | 创建 requirements.txt 固定版本；修正 README |

---

## 文件清单

### 新增文件 (7)

| 路径 | 用途 |
|------|------|
| `tests/__init__.py` | 包标记文件 |
| `tests/test_config.py` | 约 6 个测试：ConfigManager 单例、YAML 加载、get()、get_model_config()、默认值 |
| `tests/test_metrics.py` | 约 8 个测试：用合成数据测试 AUROC、AUPR、像素级 AUROC、PRO |
| `tests/test_trainer_smoke.py` | 约 4 个烟雾测试：导入、SUPPORTED_MODELS、无效模型报错、查找 checkpoint |
| `tools/validate_data.py` | 独立的数据验证脚本（检查魔法字节、可读性、目录结构、类别分布） |
| `data/DATASET_REGISTRY.md` | 规范的数据集清单，包含来源、状态和已知问题 |
| `requirements.txt` | 固定版本的依赖（anomalib==2.3.0、pytorch-lightning==1.9.5 等） |

### 修改文件 (11)

| 路径 | 修改内容 |
|------|----------|
| `modules/algorithm/__init__.py` | 导出 AnomalyDetectionTrainer、find_latest_checkpoint、get_model_from_config、get_datamodule_from_config、SUPPORTED_MODELS、MODEL_INFO |
| `modules/evaluation/__init__.py` | 导出 MetricsEvaluator、AnomalyMetrics、load_and_evaluate |
| `modules/data_processing/__init__.py` | 导出 MVTecFormatter |
| `modules/ui/demo.py` | 更新导入：`modules.algorithm.trainer` → `modules.algorithm` |
| `scripts/run_training.py` | 更新导入为模块级 API |
| `scripts/run_evaluation.py` | 同上 |
| `scripts/run_threshold.py` | 同上 |
| `scripts/run_data_processing.py` | 同上 |
| `scripts/run_ui.py` | 同上 |
| `README.md` | 修正 `anomalib>=2.0.0` → `anomalib==2.3.0`；修正数据集数量引用（5→4） |
| `CHANGELOG.md` | 添加第 1 阶段条目 |

---

## 详细设计

### C1 — 测试套件

所有测试文件位于 `tests/`。在项目根目录执行 `pytest tests/` 运行。

#### `tests/test_config.py`

测试 `modules/config/manager.py`：

```
test_singleton_identity
  两次调用 ConfigManager() 返回同一个实例。

test_yaml_loads_data_section
  加载 configs/config.yaml。验证：
    get('data.train_batch_size') 为 int > 0
    get('data.eval_batch_size') 为 int > 0
    get('data.num_workers') 为 int >= 0

test_get_nested_key
  get('data.train_batch_size') 的值与 YAML 文件一致。

test_get_model_config_returns_dict
  get_model_config('patchcore') 返回字典，包含键：
  backbone、layers、coreset_sampling_ratio、num_neighbors、pre_trained。
  同样检查 'fre'、'draem'、'padim'（各模型键的子集）。

test_threshold_default_fallback
  get('threshold.default', 0.5) 在没有配置时返回 0.5。
  get('nonexistent.key', None) 返回 None。

test_missing_key_returns_none
  get('completely.nonexistent.key.xyz') 返回 None，不报错。
```

#### `tests/test_metrics.py`

测试 `modules/evaluation/metrics.py`。所有测试使用合成 numpy 数组，不涉及文件 I/O。

```
test_auroc_perfect_separation
  scores = [0.1, 0.1, 0.1, 0.9, 0.9, 0.9]
  labels = [0, 0, 0, 1, 1, 1]
  AUROC == 1.0（误差 < 1e-6）

test_auroc_chance
  scores = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
  labels = [0, 0, 0, 1, 1, 1]（分数无区分信息）
  AUROC == 0.5（误差 < 1e-6）

test_auroc_single_class_guard
  scores = [0.1, 0.1, 0.1]
  labels = [0, 0, 0]（仅有正常类别）
  返回 0.5，不抛出异常。

test_aupr_perfect
  数据同 test_auroc_perfect_separation。
  AUPR == 1.0（误差 < 1e-6）

test_aupr_single_class_guard
  labels = [0, 0, 0]
  返回 0.0，不抛出异常。

test_pixel_auroc_perfect
  anomaly_maps  = [[[0,0],[0,0]], [[1,1],[1,1]]]  (2, 2, 2)
  gt_masks      = [[[0,0],[0,0]], [[1,1],[1,1]]]
  像素级 AUROC == 1.0

test_pixel_auroc_single_class
  gt_masks 全部为零。
  返回 0.5。

test_pro_synthetic
  2 张图片，每张 10×10。
  第一张：gt 在 (0,0) 处有 3×3 的异常区域；
          预测在同一区域有 80% 重叠。
  第二张：全部为零。
  PRO > 0.5 且 PRO < 1.0。
```

#### `tests/test_trainer_smoke.py`

轻量级导入测试，无需 GPU。

```
test_trainer_import
  确认可以从 modules.algorithm 成功导入 AnomalyDetectionTrainer。

test_supported_models_contains_all
  SUPPORTED_MODELS == {'fre', 'patchcore', 'draem', 'padim'}

test_unsupported_model_raises_valueerror
  AnomalyDetectionTrainer(model_name='invalid', data_path='.', category='x')
  抛出 ValueError，异常信息中包含 model_name。

test_find_checkpoint_nonexistent
  find_latest_checkpoint('./nonexistent_dir', 'patchcore') 返回 None。
```

### C2 — 数据验证脚本

`tools/validate_data.py` 是独立脚本 — 不导入 `modules/`，不依赖 torch/anomalib。

#### 检查项（按顺序执行）

1. **魔法字节 vs 扩展名**：对 `data/` 下每个 `.png` 文件读取前 2 字节。
   - `\x89PNG` → 正常
   - `BM` → ERROR："BMP 文件伪装为 .png 后缀，路径：<path>"
   - 其他 → ERROR："未知格式，路径：<path>"

2. **空文件**：文件大小为 0 → ERROR

3. **可读性**：分别用 PIL `Image.open()` 和 cv2 `imread()` 打开。任一方失败 → ERROR。

4. **目录结构**：检查每个数据集是否有 `train/good/`。
   对每个 test 缺陷子目录 `test/<defect>/`，检查是否有对应 `ground_truth/<defect>/`。
   缺失 → WARN。

5. **类别分布**：统计每个 test 子目录的文件数。缺陷类别 < 5 个样本 → WARN。

6. **缺失的引用数据集**：将文件系统与 README 中声明的列表对比。
   region4 → WARN。

7. **未知目录**：`data/` 中不在已知清单中的目录 → INFO。

#### 输出

- 带颜色标记（[ERROR]、[WARN]、[INFO]）打印到 stdout
- 将 Markdown 报告保存到 `results/data_validation_<YYYYMMDD_HHMMSS>.md`
- 退出码：无 ERROR → 0；有 ERROR → 1

### C3 — 数据集注册表

`data/DATASET_REGISTRY.md` 记录每个数据集：

| 数据集 | 来源 | 状态 | 训练集 | 测试集(正常) | 测试集(缺陷) | 已知问题 |
|--------|------|------|--------|-------------|-------------|----------|
| bottle | MVTec AD | 完整 | 209 | 20 | 63 | 无 |
| carpet | MVTec AD | 完整 | 280 | 28 | 89 | 无 |
| region1 | 企业数据 | 不完整 | 待定 | 91 | 7 | BMP伪装为PNG；严重不平衡 |
| region2 | 企业数据 | 不完整 | 待定 | 91 | 15 | BMP伪装为PNG；严重不平衡 |
| region3 | 企业数据 | 完整 | 待定 | 150 | 17 | 轻微不平衡 |
| region4 | — | 缺失 | — | — | — | 磁盘上从未存在 |
| region5 | 企业数据 | 不完整 | 待定 | 91 | 23 | BMP伪装为PNG；中度不平衡 |

README 中的引用修正："region1-5 (5 个自定义数据集)" → "4 个自定义数据集 (region1-3、region5)"。

### H3 — 模块公共 API

**`modules/algorithm/__init__.py`：**
```python
from .trainer import (
    AnomalyDetectionTrainer,
    find_latest_checkpoint,
    get_model_from_config,
    get_datamodule_from_config,
    SUPPORTED_MODELS,
    MODEL_INFO,
)
```

**`modules/evaluation/__init__.py`：**
```python
from .metrics import MetricsEvaluator, AnomalyMetrics, load_and_evaluate
```

**`modules/data_processing/__init__.py`：**
```python
from .dataset_formatter import MVTecFormatter
```

**消费者更新：** `modules/ui/demo.py` 和所有 5 个 `scripts/run_*.py` 更新导入路径：
- `from modules.algorithm.trainer import X` → `from modules.algorithm import X`
- `from modules.evaluation.metrics import X` → `from modules.evaluation import X`
- `from modules.data_processing.dataset_formatter import X` → `from modules.data_processing import X`

### M1 — 固定依赖

创建 `requirements.txt`，使用精确版本：

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

README 中修改：`anomalib>=2.0.0` → `anomalib==2.3.0`，附说明：
> "固定为 2.3.0 版本，因 trainer.py 中的 monkey-patch 兼容层依赖此版本。升级 anomalib 前需先更新这些补丁。"

---

## 验证

第 1 阶段实现完成后，以下命令必须通过：

```bash
# 测试
pytest tests/ -v                    # 全部测试通过

# 数据验证
python tools/validate_data.py       # 报告 BMP-as-PNG 错误（已知并已记录）
echo $?                             # 退出码 1（BMP 错误是真实问题）

# 导入完整性
python -c "
from modules.algorithm import AnomalyDetectionTrainer, SUPPORTED_MODELS
from modules.evaluation import MetricsEvaluator, AnomalyMetrics
from modules.data_processing import MVTecFormatter
print('All imports OK')
"

# 依赖
pip install -r requirements.txt --dry-run 2>&1 | grep -c "conflict"  # 0 个冲突
```

---

## 第 1 阶段不做什么

- 不将 BMP 文件转换为 PNG（需数据来源方确认后再决定）
- 不重构 monkey-patch（属于第 2 阶段）
- 不添加 CI/CD（属于第 4 阶段）
- 不修改 trainer.py 的运行时行为
- 不在测试中运行实际的模型训练（避免 GPU 依赖）
