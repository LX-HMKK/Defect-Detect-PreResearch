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

# 参数消融实验
python tools/run_ablation.py -m all -c bottle -d ./data

# 混淆矩阵生成
python tools/run_confusion_matrix.py -m all -c all

# 数据验证
python tools/validate_data.py -d ./data

# PRO 后处理配方评估
python tools/run_post_process_eval.py -m all -c all

# 启动 FastAPI UI (Phase 2 — 默认)
python scripts/run_ui.py
# → http://127.0.0.1:8000

# 启动 Gradio UI (legacy fallback)
python scripts/run_ui.py --gradio
# → http://127.0.0.1:7860

# 自定义端口
python scripts/run_ui.py --port 3000

# 直接指定 Python 路径 (Windows)
"C:\Users\lx_hm\.conda\envs\anomalib\python.exe" scripts/run_training.py -m patchcore -c bottle

# 运行测试（无 GPU 依赖）
python -m pytest tests/ -v

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
  _runtime.py                  # 共享运行时（pycache 重定向、resolve_project_path、get_runtime_cache_dir）
  algorithm/
    trainer.py                 # 核心：AnomalyDetectionTrainer + 模型/数据模块工厂函数
    _anomalib_compat.py        # anomalib 2.3.0 ↔ PyTorch Lightning 1.9.5 猴子补丁兼容层
  config/manager.py            # ConfigManager 单例，管理 configs/config.yaml
  data_processing/dataset_formatter.py  # MVTecFormatter：原始数据 → MVTec AD 结构
  evaluation/metrics.py        # MetricsEvaluator（从零实现 AUROC/AUPR/PRO）
  evaluation/post_processor.py # AnomalyMapProcessor: 异常热力图后处理（7 种配方）
  ui/
    demo.py                   # Gradio UI：AnomalyDetector + create_interface
    styles.css                # Apple 风格设计系统（亮/暗双模式，1050+ 行）
    theme.py                  # 主题管理器：色板定义 + CSS 生成 + 切换按钮 + Favicon
    static/theme.js           # 主题切换交互（localStorage + data-theme）
	    static/inference-interact.js  # 推理结果交互增强（热力图 hover tooltip + bbox 高亮）
configs/
  config.yaml                        # 主配置（路径、训练参数、阈值）
  {patchcore,padim,fre,draem}.yaml   # 各模型 anomalib CLI 格式配置
tests/                          # 测试套件（config 单例、metrics 指标、trainer 烟雾测试）
tools/                          # 分析工具（小样本/消融/基准/混淆矩阵/数据验证/统计/报告/后处理）
results/                        # 训练结果
docs/
  # 演示材料 (讲稿.html/md)、需求/任务书
  superpowers/specs/            # 设计规范（Apple UI 设计规范、Phase 1 前端增强）
  superpowers/plans/            # 实现计划
.cache/                         # 运行时缓存 (pycache, logs, pretrained)
```

### 关键类及其职责

- **`AnomalyDetectionTrainer`** (`modules/algorithm/trainer.py`) — 核心调度器。`setup()` 创建 datamodule + model，`train()` 运行 anomalib `Engine.fit()`，`evaluate()` 运行 `Engine.test()` 并通过 Youden's J 统计量在 [0,1] 全范围搜索计算最优阈值，`_save_results()` 将 JSON 写入 `results/comparison/`，`_update_results_json_threshold()` 将阈值回写已有结果文件。静态方法 `compare_models()` 生成对比 CSV/Markdown。内部使用两个 helper 工厂函数 `get_model_from_config()` 和 `get_datamodule_from_config()`，严格从 YAML 读取所有参数，缺失配置时抛出 `ValueError`。

- **`get_model_from_config(model_name, config)`** — 始终附加 `anomalib.metrics.Evaluator`，包含 6 个指标（图像级 AUROC/AUPR/F1Score + 像素级 AUROC/PRO/F1Score）。参数严格从配置读取——缺失键抛出 `ValueError`。先查 `configs/config.yaml` 的 `models.{name}` section，然后接受传入 config 的覆盖。

- **`get_datamodule_from_config(data_path, category, model_name, config)`** — 自动检测 MVTec AD 与通用 Folder 格式。Folder 格式始终设置 `task='segmentation'`。train_batch_size/eval_batch_size/num_workers 严格从配置读取。

- **`modules/algorithm/_anomalib_compat.py`** — 猴子补丁兼容层，修复 anomalib 2.3.0 ↔ PyTorch Lightning 1.9.5 回调签名不匹配。详见[Trainer 兼容性补丁](#trainer-兼容性补丁)。

- **`modules/_runtime.py`** → `configure_runtime_temp()` — 将 `sys.pycache_prefix` 和 `PYTHONPYCACHEPREFIX` 重定向到 `./.cache/pycache/`。所有 `scripts/run_*.py` 和 `tools/run_*.py` 开头均调用此函数。

- **`ConfigManager`** (`modules/config/manager.py`) — 基于 `configs/config.yaml` 的单例。支持点号分隔取值 `get('data.image_size')`。`get_threshold(model, dataset)` 从已保存的 JSON 结果读取。模块级便捷函数：`get()`、`get_threshold()`、`get_model_config()`、`get_data_config()`。

- **`MetricsEvaluator`** (`modules/evaluation/metrics.py`) — 从零实现的指标计算（scikit-learn + scipy）。包含 `AnomalyMetrics` dataclass（4 个字段 + `to_dict()`/`to_percent_dict()`）。`load_and_evaluate()` 从已有 JSON 加载并打印。训练器端改为委托 anomalib 内置评估器进行在线计算，此模块用于离线评估和单元测试。

- **`MVTecFormatter`** (`modules/data_processing/dataset_formatter.py`) — 将企业原始图像转换为 MVTec AD 目录布局，使用 letterbox 缩放。训练样本上限由 `max_train_samples` 控制。

- **`AnomalyMapProcessor`** (`modules/evaluation/post_processor.py`) — 异常热力图后处理管线。提供 4 种基础算子（高斯滤波/中值滤波/形态学开闭/双线性上采样）及 7 种组合配方，通过 `PRESET_CONFIGS` 字典调用。`process_anomaly_maps()` 为批量处理入口。

- **`AnomalyDetector`** (`modules/ui/demo.py`) — 加载 checkpoint、执行推理、生成热力图叠加层与 base64 编码的原始灰度图（供前端 hover 交互），在独立阈值 (`NMS_BBOX_THRESHOLD = 0.3`) 下生成 NMS 边界框并编码为 JSON 嵌入结果 HTML。`_format_result()` 返回带逐层入场动画（`.reveal-child-*`）的 Apple 风格结果卡片。单模型推理 (`on_run_click`) 和四模型对比 (`on_compare_click`) 均使用 generator `yield` 实现流式渐进渲染（骨架屏 → 加载态 → 结果）。页面加载时通过 `gr.Blocks(head=...)` 注入阻塞式反 FOUC 脚本。

- **`modules/ui/theme.py`** — 主题管理器。`DARK`/`LIGHT` 色板字典 → `build_css_variables()` 编译为 CSS `:root` 变量块。提供 `get_theme_switch_html()`（太阳/月亮 SVG 切换按钮）、`get_theme_js()`（localStorage 持久化逻辑）、`get_favicon_html()`（SVG 菱形图标，亮/暗双模式自适应）。暗色默认变量在 `styles.css` 的 `:root` 块中，亮色变量通过 `gr.HTML("<style>…</style>")` 注入以绕过 Gradio 6 的 CSS 作用域处理。

- **`modules/ui/static/inference-interact.js`** — 推理结果交互增强。MutationObserver 监听 DOM 变化（回调中先 disconnect 防递归），从隐藏的 base64 灰度图读取像素值创建离屏 canvas，在热力图 `<img>` 上 mousemove 显示 tooltip（Apple 风格异常得分）；bbox JSON → 绝对定位透明 overlay，hover 高亮 `var(--accent)` 边框。`_bboxCleanups[]` 追踪所有 resize/load 监听器，`cleanupOverlays()` 统一释放防内存泄漏。

- **`modules/ui/styles.css`** — Apple 风格设计系统（~1350 行）。亮/暗双模式通过 CSS 自定义属性实现，`html[data-theme="light"]` 选择器控制亮色覆盖。包含 15 组件的完整规范（标题区+Logo、标签页、算法卡片、输入控件、按钮、状态面板+进度条、骨架屏、结果卡片、度量数字、热力图图例+tooltip、bbox overlay、对比模式、底部说明+页脚动画、全页加载遮罩、下拉框修复），以及交错入场动画（`.reveal-1` ~ `.reveal-6` + `.footerRise`）、磨砂玻璃效果（`backdrop-filter: blur(20px)`）、弹簧缓动（`cubic-bezier(0.22,0.8,0.3,1.15)`）。`.reveal`/`.reveal-child-*` 不使用 `animation-fill-mode: forwards`（防止 `transform:translateY(0)`/`filter:blur(0)` 为非 none 值创建包含块导致 `position:fixed` 弹出框随滚动偏移）。设计规范详见 `docs/superpowers/specs/2026-06-18-apple-ui-design-spec.md`。

### 数据流

1. 原始图像 → `MVTecFormatter` → MVTec AD 格式 (`train/good/`, `test/<defect>/`, `ground_truth/<defect>/`)
2. `AnomalyDetectionTrainer.__init__()` → 解析 `config_path`（默认 `configs/{model}.yaml`）→ `yaml.safe_load()` 载入配置
3. `setup()` → `get_datamodule_from_config()` + `get_model_from_config()`（均严格从配置读取参数）
4. `train()` → anomalib `Engine.fit()`（PatchCore/PaDiM：仅 coreset 子采样/高斯建模，1 个 epoch）
5. `evaluate()` → `Engine.test()` → 提取 6 个指标 → `_compute_optimal_threshold()` (Youden's J, [0,1] 全域搜索) → `_save_results()` + `_update_results_json_threshold()` → JSON 输出至 `results/comparison/{model}_{category}_results.json`
6. UI 通过 `AnomalyDetector.load_model()` → 动态创建模型（不加载 checkpoint 权重于模型对象，checkpoint 仅传入 `Engine.predict()` 的 `ckpt_path` 参数）→ `predict()` 生成热力图 + NMS 边界框

### 训练器中的三层配置优先级

`AnomalyDetectionTrainer` 读取参数时遵循：1) 显式传入的 `config_path` (YAML) > 2) `configs/config.yaml` 对应 `models.{name}` section > 3) 抛出 `ValueError`。`get_datamodule_from_config()` 和 `get_model_from_config()` 均遵循此模式。

### 早停机制

DRAEM/FRE 监控 `val_image_AUROC`（需在各模型 YAML 的 `early_stopping.enabled: true` 启用）；PatchCore/PaDiM 为单 epoch 特征提取/高斯建模，不需要早停。早停参数（`patience`、`min_delta`、`monitor_metric`）在各模型 `{model}.yaml` 的 `early_stopping` 下配置。`configs/config.yaml` 的 `early_stopping` section 作为后备默认值，默认 `enabled: false`。

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
- **多行提交信息必须使用 `-F` 从文件读取**。禁止使用 `git commit -m @'...'@`（PowerShell here-string 语法）或直接内联多行消息——本项目中 Bash 和 PowerShell 两套 shell 并存，here-string/here-doc 语法在交叉环境下极容易误用，导致 `@` 等无关字符混入提交信息。正确做法：先将消息写入临时文件（如 `.git-msg`），然后执行 `git commit -F .git-msg`，完成后删除。

## 算法推荐

| 算法 | 技术路线 | image_AUROC | pixel_AUROC | PRO | 参数量 | 推荐 |
|------|----------|:---:|:---:|:---:|:---:|:---:|
| **PatchCore** | 特征建模（检索） | 100% | 98.6% | 80.1% | 24.9M | ✅ 首选 |
| **PaDiM** | 特征建模（概率） | 100% | 98.2% | 80.2% | **2.8M** | ✅ 轻量 |
| **DRAEM** | 自监督判别 | 97.7% | 86.2% | 48.3% | 97.4M | 🔬 备选 |
| **FRE** | 特征重构 | 99.4% | 97.5% | 69.1% | 23.0M | 🔬 备选 |

> bottle 数据集代表性结果。完整数据见 `results/comparison/`。
> **核心约束**：只有正常样本可用，无监督设定。

## 关键注意事项

### 训练并发限制

**训练任务必须串行执行，一次只能跑一个。** 本机 GPU (RTX 4060 Laptop GPU, 8GB VRAM) 显存有限，多个训练进程并发会导致 GPU 内存溢出、系统卡死。必须等当前训练完全结束后才能启动下一个。

### 数据集路径

`--data_path` / `data_path` 始终指向**类别父目录** (`./data`)，而非 `./data/bottle`。

### 数据集现状

参考 [data/DATASET_REGISTRY.md](data/DATASET_REGISTRY.md)。region1/2/5 中部分 .png 文件实际为 BMP 格式（OpenCV 可自动识别，当前不影响训练）。**region4 目录不存在**，仅有 region1/2/3/5 四个企业数据集。外部数据集 `datasets/dtd/` 需手动下载（DRAEM 生成合成异常纹理使用）。

### Windows 特定

- 配置中 `num_workers: 0` 以避免多进程问题
- 安全张量取值：使用 `.cpu().max().item()` 而非 `.max().item()`
- **所有新脚本必须：**
  1. 调用 `modules/_runtime.py` 的 `configure_runtime_temp()` 将 pycache 重定向到 `./.cache/pycache`
  2. 将 stdout/stderr 包装为 UTF-8：`sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')`
  3. 设置 `PROJECT_ROOT = Path(__file__).resolve().parents[1]` 并使用绝对路径（不要用相对路径，工作目录不可靠）
- 现有 scripts/ 和 tools/ 下的脚本均已遵循此模式，新脚本可直接参考 `scripts/run_training.py` 作为模板

### 配置加载优先级

当各模型 YAML 配置路径传入 `AnomalyDetectionTrainer` 时，优先使用该配置。否则回退到 `ConfigManager.get()` 从 `configs/config.yaml` 读取。配置键缺失时抛出 `ValueError` 而非静默使用默认值。

### Trainer 兼容性补丁

`modules/algorithm/_anomalib_compat.py` 包含针对 anomalib 2.3.0 与 PyTorch Lightning 1.9.5 的兼容性补丁（回调签名不匹配）。该文件在 `trainer.py` 中通过 `from . import _anomalib_compat` 导入时自动触发。在未验证 Lightning/anomalib 版本组合之前，不要移除这些补丁。

### 测试

测试套件位于 `tests/`，共 3 个文件。设计为无 GPU、无 anomalib 导入依赖（烟雾测试使用 AST 解析源码以避免触发 Heavy import）。运行方式：`python -m pytest tests/ -v`。

### UI 架构 (Phase 2)

**默认 UI**: FastAPI + Alpine.js SPA (`modules/ui/server.py` + `modules/ui/static/`)。

- 5 层动效体系：环境光呼吸 → 鼠标光晕跟随 → 滚动驱动动画 → 微交互（胶囊开关/数字跳动/弹簧按钮）→ 视图过渡
- 三区块自适应连续滚动：算法介绍（SVG 流程图）→ 单模型推理（SSE 流式）→ 四模型对比（并行 SSE）
- 亮/暗双模式：胶囊开关，localStorage 持久化，`prefers-color-scheme` 系统跟随
- 设计规范：`docs/superpowers/specs/2026-06-19-apple-ui-phase2-design.md`

**回退**: `python scripts/run_ui.py --gradio` 启动原有 Gradio UI（`modules/ui/demo.py`），功能完整保留。

### UI 调试 (Phase 2)

```bash
# 启动 FastAPI 开发服务器
python scripts/run_ui.py
# → http://127.0.0.1:8000

# 健康检查
curl http://127.0.0.1:8000/api/health

# 模型列表
curl http://127.0.0.1:8000/api/models

# 独立测试主题模块
python modules/ui/theme.py    # 输出完整 CSS 到 stdout
```

**Gradio 6 CSS 作用域问题（仅影响 legacy Gradio UI）**：`gr.Blocks(css=...)` 传入的 CSS 会被 Gradio 6 做选择器作用域处理——在所有选择器前加 `.gradio-container.xxx .contain`。这会导致 `@media` 查询内的 `:root` 选择器失效（变成 `.contain :root`，无法匹配文档根）。**解决方案**：需要通过 CSS `@media` 动态切换的变量（如亮色模式色板），必须通过 `gr.HTML("<style>…</style>")` 注入，绕过 Gradio 的 CSS 处理器。顶层 `:root` 块（暗色默认值）不受影响。

**Gradio 6 CSS 作用域问题：** `gr.Blocks(css=...)` 传入的 CSS 会被 Gradio 6 做选择器作用域处理——在所有选择器前加 `.gradio-container.xxx .contain`。这会导致 `@media` 查询内的 `:root` 选择器失效（变成 `.contain :root`，无法匹配文档根）。**解决方案**：需要通过 CSS `@media` 动态切换的变量（如亮色模式色板），必须通过 `gr.HTML("<style>…</style>")` 注入，绕过 Gradio 的 CSS 处理器。顶层 `:root` 块（暗色默认值）不受影响。

**亮/暗双模式：** 系统通过 `prefers-color-scheme` 自动检测，并支持手动切换。手动选择存储在 `localStorage.theme`，优先级高于系统设定。CSS 通过 `html[data-theme="light"]` 选择器覆盖变量。切换逻辑在 `modules/ui/static/theme.js`。

**主题开发命令：**
```bash
# 独立测试主题模块
python modules/ui/theme.py    # 输出完整 CSS 到 stdout
```

## 相关文件

- [README.md](README.md) — 完整项目文档，含算法对比表、环境搭建
- [CHANGELOG.md](CHANGELOG.md) — 发布历史，含阶段编号便于追溯
- [data/DATASET_REGISTRY.md](data/DATASET_REGISTRY.md) — 所有数据集清单、已知问题、defect-type 缩写表
- [requirements.txt](requirements.txt) — 固定版本依赖清单
- [configs/config.yaml](configs/config.yaml) — 主配置（所有可调参数集中管理）
- [docs/superpowers/specs/2026-06-18-apple-ui-design-spec.md](docs/superpowers/specs/2026-06-18-apple-ui-design-spec.md) — Apple UI 设计规范（12 组件 + 双模式变量 + 动效参数表）
- [docs/superpowers/specs/2026-06-18-phase1-frontend-enhancement-design.md](docs/superpowers/specs/2026-06-18-phase1-frontend-enhancement-design.md) — Phase 1 前端增强设计
- [docs/superpowers/plans/2026-06-18-phase1-implementation-plan.md](docs/superpowers/plans/2026-06-18-phase1-implementation-plan.md) — Phase 1 实现计划
