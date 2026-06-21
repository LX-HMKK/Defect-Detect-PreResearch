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
# → http://127.0.0.1:8000（自动打开浏览器）

# 不自动打开浏览器（调试用）
python scripts/run_ui.py --no-browser

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

### 典型工作流

```bash
# 1. 准备数据
python scripts/run_data_processing.py -i ./data/raw -o ./data/processed/bottle --max_train 150

# 2. 训练模型
python scripts/run_training.py -m patchcore -c bottle -d ./data

# 3. 计算最优阈值（可选，会更新已有结果）
python scripts/run_threshold.py -m patchcore -c bottle --save

# 4. 评估并查看指标
python scripts/run_evaluation.py -m patchcore -c bottle

# 5. 启动 UI 进行推理
python scripts/run_ui.py
```

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
    server.py                  # FastAPI 服务器 — Phase 2 核心：REST API + SSE 流式推理 + Training Studio 训练端点
    training_backend.py        # Training Studio 后端：任务锁、样本格式化、SSE 指标回调、run_training_job
    demo.py                    # [Legacy] Gradio UI：AnomalyDetector + create_interface
    theme.py                   # 主题管理器：色板定义 + CSS 生成 + Favicon
    styles.css                 # [Legacy] Gradio 专用 CSS（Phase 2 使用 static/css/app.css）
    static/
      index.html               # Alpine.js SPA 入口（CSS snap 全屏滚动，四页吸附）
      theme.js                 # 主题切换交互（localStorage + data-theme）
      inference-interact.js    # [Legacy] Gradio 推理结果交互增强
      css/
        app.css                # Phase 2 主样式表（亮/暗双模式，~2700 行）
        apple-redesign.css     # 训练工作室/单模型推理/四模型对比重设计样式
        flowchart.css          # SVG 流程图动画样式
      js/
        app.js                 # Alpine 全局状态：主题/snap 导航/进度环/推理/健康检查/训练完成监听
        inference.js           # InferenceRunner (SSE) + imageCompare 滑块 + tooltip + bbox
        compare.js             # CompareRunner (SSE) + Alpine compare 组件
        training.js            # TrainingRunner (SSE) + Alpine training 状态机 + 样本上传/参数/loss 曲线
        animations.js          # 滚动驱动淡入动画 (ScrollReveal) + snap 过渡编排（JS WAAPI 统一驱动进出，CSS 退出动画已移除）
        cursor-glow.js         # 鼠标光晕跟随效果
        flowchart.js           # SVG 流程图绘制动画
        hero-visual.js         # 首页 hero 区域 SVG 流程图动画
configs/
  config.yaml                        # 主配置（路径、训练参数、阈值）
  {patchcore,padim,fre,draem}.yaml   # 各模型 anomalib CLI 格式配置
tests/                          # 测试套件（config 单例、metrics 指标、trainer 烟雾测试、Training Studio API、静态资源结构）
tools/                          # 分析工具（小样本/消融/基准/混淆矩阵/数据验证/统计/报告/后处理）
results/                        # 训练结果
docs/
  # 演示材料 (讲稿.html/md)、需求/任务书
  superpowers/specs/            # 设计规范
  superpowers/plans/            # 实现计划
memory/                         # Claude Code 会话记忆
.cache/                         # 运行时缓存 (pycache, logs, pretrained)
```

### 关键类及其职责

- **`AnomalyDetectionTrainer`** (`modules/algorithm/trainer.py`) — 核心调度器。`setup()` 创建 datamodule + model，`train()` 运行 anomalib `Engine.fit()`，`evaluate()` 运行 `Engine.test()` 并通过 Youden's J 统计量在 [0,1] 全范围搜索计算最优阈值，`_save_results()` 将 JSON 写入 `results/comparison/`，`_update_results_json_threshold()` 将阈值回写已有结果文件。静态方法 `compare_models()` 生成对比 CSV/Markdown。内部使用两个 helper 工厂函数 `get_model_from_config()` 和 `get_datamodule_from_config()`，严格从 YAML 读取所有参数，缺失配置时抛出 `ValueError`。支持 `enable_pixel_metrics`（无 `ground_truth` 时关闭像素级指标）和 `learning_rate`（通过 Lightning 回调覆盖 DRAEM/FRE 优化器学习率）。

- **`TrainingTaskManager` / `TrainingMetricsCallback` / `format_uploaded_samples` / `run_training_job`** (`modules/ui/training_backend.py`) — Training Studio 后端。`TrainingTaskManager` 提供全局单训练任务锁；`format_uploaded_samples` 将上传图片整理为临时 MVTec/Folder 结构；`run_training_job` 在线程中执行训练并通过 SSE 队列推送状态/指标/日志；`TrainingMetricsCallback`（PyTorch Lightning Callback）在每个 epoch 结束时将 loss、learning_rate、val_image_AUROC 写入队列。
- **`_model_info.py`** (`modules/ui/_model_info.py`) — 轻量模型/数据集扫描。提供 `MODEL_CONFIGS`、`get_available_datasets()`、`get_self_trained_models()`，避免 API 端点导入 anomalib/torch 等 heavy 依赖。
- **`_training_common.py`** (`modules/ui/_training_common.py`) — Training Studio 公共组件：`TrainingTaskManager`、`format_uploaded_samples`、常量 `MAX_TRAIN_SAMPLES`。`training_backend.py` 通过它复用这些组件。

- **`get_model_from_config(model_name, config)`** — 始终附加 `anomalib.metrics.Evaluator`，包含 6 个指标（图像级 AUROC/AUPR/F1Score + 像素级 AUROC/PRO/F1Score）。参数严格从配置读取——缺失键抛出 `ValueError`。先查 `configs/config.yaml` 的 `models.{name}` section，然后接受传入 config 的覆盖。

- **`get_datamodule_from_config(data_path, category, model_name, config)`** — 自动检测 MVTec AD 与通用 Folder 格式。仅当存在 `train`、`test`、`ground_truth` 三个目录时走 MVTec；否则回退到 `Folder`（`normal_dir='train/good'`，`normal_test_dir='test/good'`）。train_batch_size/eval_batch_size/num_workers 严格从配置读取。

- **`modules/algorithm/_anomalib_compat.py`** — 猴子补丁兼容层，修复 anomalib 2.3.0 ↔ PyTorch Lightning 1.9.5 回调签名不匹配。详见下文「Trainer 兼容性补丁」。

- **`modules/_runtime.py`** → `configure_runtime_temp()` — 将 `sys.pycache_prefix` 和 `PYTHONPYCACHEPREFIX` 重定向到 `./.cache/pycache/`。所有 `scripts/run_*.py` 和 `tools/run_*.py` 开头均调用此函数。

- **`ConfigManager`** (`modules/config/manager.py`) — 基于 `configs/config.yaml` 的单例。支持点号分隔取值 `get('data.image_size')`。`get_threshold(model, dataset)` 从已保存的 JSON 结果读取。模块级便捷函数：`get()`、`get_threshold()`、`get_model_config()`、`get_data_config()`。

- **`MetricsEvaluator`** (`modules/evaluation/metrics.py`) — 从零实现的指标计算（scikit-learn + scipy）。包含 `AnomalyMetrics` dataclass（4 个字段 + `to_dict()`/`to_percent_dict()`）。`load_and_evaluate()` 从已有 JSON 加载并打印。训练器端改为委托 anomalib 内置评估器进行在线计算，此模块用于离线评估和单元测试。

- **`MVTecFormatter`** (`modules/data_processing/dataset_formatter.py`) — 将企业原始图像转换为 MVTec AD 目录布局，使用 letterbox 缩放。训练样本上限由 `max_train_samples` 控制。

- **`AnomalyMapProcessor`** (`modules/evaluation/post_processor.py`) — 异常热力图后处理管线。提供 4 种基础算子（高斯滤波/中值滤波/形态学开闭/双线性上采样）及 7 种组合配方，通过 `PRESET_CONFIGS` 字典调用。`process_anomaly_maps()` 为批量处理入口。

### Phase 2 UI（FastAPI + Alpine.js SPA）

- **`modules/ui/server.py`** → FastAPI 应用 — Phase 2 核心。`/api/predict` (SSE 流式单模型推理，支持 `pretrained`/`self_trained` 两种来源)、`/api/compare` (SSE 流式四模型并行推理)、`/api/upload-samples`（训练样本上传）、`/api/train`（SSE 流式训练）、`/api/train-status`、`/api/train/stop`、`/api/models`（模型与数据集列表）、`/api/self-trained-models`（指定模型下的用户自训练模型）、`/api/test-images`（数据集 test/ 图片列表）、`/api/train-samples`（训练 good 样本列表）等端点。推理与训练通过 `asyncio.to_thread` 在线程池执行，避免阻塞事件循环。`CacheControlMiddleware` 为 `/static/*` 资源添加 `no-cache` 头（ETag 验证）。`/api/theme/light-css` 返回亮色 CSS 变量。

- **`Alpine.data('app')`** (`modules/ui/static/js/app.js`) — 全局状态。主题切换（`toggleTheme` + `prefers-color-scheme` 跟随）、导航滚动（`IntersectionObserver`（以 `.snap-container` 为 `root`）+ 键盘 ↑↓）、数据集/模型列表获取、推理状态机（`idle→uploaded→loading→inferring→done|error`）、SSE 流式推理调度、健康检查轮询（30s）、训练完成后刷新模型/数据集列表。导航下拉选择器使用 Alpine 内联 `x-data="{ open: false }"` 模式（仅遮蔽 `open`，父作用域属性自动透传），替代了原生 `<select>`。

- **`Alpine.data('training')`** (`modules/ui/static/js/training.js`) — Training Studio 组件。样本上传画廊（支持拖拽/点击）、排除样本切换、训练参数配置（epochs/batch_size/learning_rate/seed）、药丸算法选择器、`TrainingRunner.run()` 消费 `/api/train` SSE 流并绘制实时 loss 曲线、训练完成触发全局刷新。

- **`Alpine.data('compare')`** (`modules/ui/static/js/compare.js`) — 四模型对比组件。`compareSlots` 状态机（`pending→active→done|error`）、`CompareRunner.run()` 消费 `/api/compare` SSE 流、`setupCompareBbox()` 为每个槽位创建 bbox overlay 并监听 ResizeObserver 实时定位。

- **`InferenceRunner`** (`modules/ui/static/js/inference.js`) — `/api/predict` SSE 客户端。ReadableStream 读取 + CRLF→LF 归一化 + event/data 行解析。

- **`imageCompare`** Alpine 组件 (`inference.js`) — 原图/热力图对比滑块。`mousemove/touchmove` 拖拽控制 `clipPath: inset()`。

- **`setupHeatmapTooltip()`** (`inference.js`) — 离屏 canvas 从隐藏 `<img>` 读取灰度像素值，mousemove 显示 Apple 风格异常得分 tooltip。

- **`setupBboxOverlays()`** (`inference.js`) — NMS bbox JSON → 绝对定位 overlay，hover 高亮 `var(--accent)` 边框，计入 `object-fit: contain` 居中偏移。

- **`.pipeline`** (CSS grid) — 三列等宽水平流水线（`grid-template-columns: 1fr 1fr 1fr`）。每列 `.pipeline-step`（步骤卡片，带圆形序号 `.pipeline-step-num`）。步骤间用 `::after` 伪元素 `→` 连接。**禁止向 `.pipeline` 添加额外子元素，会破坏 grid 布局。**

- **`.algo-card-accent`** — 算法卡片左侧 3px 色标竖线（`position: absolute`）。颜色通过 `--algo-color` CSS 变量驱动：PatchCore #2997ff, PaDiM #30d158, FRE #ff9f0a, DRAEM #bf5af2。

- **`.snap-dots`** — 右侧固定导航点（改为进度环）。SVG 圆环通过 `snapProgress` 驱动 `stroke-dashoffset` 实现连续填充。移动端移至底部横条。

- **`modules/ui/static/css/app.css`** — Phase 2 主样式表（~2500 行）。CSS 自定义属性实现亮/暗双模式，包含 15+ 组件（导航栏磨砂玻璃、自定义下拉选择器、上传区、骨架屏、进度条微光扫过、一体化仪表盘卡片 `.result-dashboard`、对比滑块、热力图图例 overlay、bbox overlay、四模型对比网格 `.compare-grid`、流水线收缩摘要 `.pipeline-summary`、页脚动画、全页加载遮罩）。系统字体栈（`SF Pro Display` → `PingFang SC` → `Microsoft YaHei`），零外部字体依赖。关键陷阱：`.compare-heatmap` 全局选择器 `position: absolute`（单模型滑块用）曾泄漏到四模型对比槽位导致热力图不可见——已通过 `.compare-container .compare-heatmap` 收缩范围 + `.compare-slot .compare-heatmap { position: relative }` 防御。Chrome 115+ `@supports (animation-timeline: view())` 块已禁用（与 JS scroll-snap 动画编排冲突）。

### Gradio Legacy 组件

- **`AnomalyDetector`** (`modules/ui/demo.py`) — 被 FastAPI 推理与 Gradio 回退共同使用的异常检测器。支持预训练模型与自训练模型加载、执行推理、生成热力图叠加层与 base64 编码的原始灰度图（供前端 hover 交互），在独立阈值 (`NMS_BBOX_THRESHOLD = 0.3`) 下生成 NMS 边界框。`_format_result()` 返回带逐层入场动画（`.reveal-child-*`）的 Apple 风格结果卡片。单模型/四模型对比均使用 generator `yield` 实现流式渐进渲染。页面加载时通过 `gr.Blocks(head=...)` 注入阻塞式反 FOUC 脚本。

- **`modules/ui/theme.py`** — 主题管理器。`DARK`/`LIGHT` 色板字典 → `build_css_variables()` 编译为 CSS `:root` 变量块。`get_light_css()` 供 FastAPI `/api/theme/light-css` 端点使用。暗色默认变量在 `app.css` 的 `:root` 块中，亮色变量通过 `html[data-theme="light"]` 选择器覆盖。

- **`modules/ui/styles.css`** — [Legacy] Gradio 专用样式（~1350 行），Phase 2 不再使用。

- **`modules/ui/static/inference-interact.js`** — [Legacy] Gradio 推理结果交互增强。Phase 2 对应功能在 `js/inference.js`。

### 数据流

1. 原始图像 → `MVTecFormatter` → MVTec AD 格式 (`train/good/`, `test/<defect>/`, `ground_truth/<defect>/`)
2. `AnomalyDetectionTrainer.__init__()` → 解析 `config_path`（默认 `configs/{model}.yaml`）→ `yaml.safe_load()` 载入配置
3. `setup()` → `get_datamodule_from_config()` + `get_model_from_config()`（均严格从配置读取参数）
4. `train()` → anomalib `Engine.fit()`（PatchCore/PaDiM：仅 coreset 子采样/高斯建模，1 个 epoch）
5. `evaluate()` → `Engine.test()` → 提取 6 个指标 → `_compute_optimal_threshold()` (Youden's J, [0,1] 全域搜索) → `_save_results()` + `_update_results_json_threshold()` → JSON 输出至 `results/comparison/{model}_{category}_results.json`
6. UI 通过 FastAPI (`server.py`) 暴露 `/api/predict` 和 `/api/compare` SSE 端点 → `_run_prediction()` 在线程池执行推理，支持 `source=pretrained`（读取 `results/{model}/{ModelName}/default/{category}`）或 `source=self_trained`（读取 `/api/self-trained-models` 返回的用户训练目录）→ 返回 `{image_b64, heatmap_b64, bboxes, score, ...}` → 前端 Alpine.js SPA 消费 SSE 流并渲染结果。测试图片通过 `/api/test-images` 从 `test/` 目录选取，不再依赖前端上传。
7. Training Studio 数据流：前端上传正常样本 → `/api/upload-samples` 调用 `format_uploaded_samples()` 整理为 `train/good + test/good` 临时目录 → `/api/train` 在 `TrainingTaskManager` 锁保护下通过 `run_training_job()` 实例化 `AnomalyDetectionTrainer` → `TrainingMetricsCallback` 经 SSE 推送 epoch loss / learning_rate / val_image_AUROC / ETA → 训练完成后将 checkpoint 重写为仅含 `state_dict` 的安全格式，结果写入 `results/comparison/{model}_{category}_results.json`，并触发前端刷新模型/数据集列表。

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

提交前自检（筛查是否误带协作者签名）：

```bash
git log --all --format="%H %B" | grep -i "Co-authored-by"
```

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

### 自训练模型 checkpoint 安全格式

训练完成后，`run_training_job()` 会将 Lightning checkpoint 重写为仅含 `{'state_dict': ...}` 的安全格式，并在推理时使用 `weights_only=True` 加载。因此自训练模型走 `source='user'` 路径时，`Engine.predict()` 传入 `ckpt_path=None`，状态字典由 `AnomalyDetector.load_self_trained_model()` 手动注入模型。不要直接给 `Engine.predict()` 传入原始 Lightning checkpoint 路径。

### 测试

测试套件位于 `tests/`，共 5 个文件：`test_config.py`、`test_metrics.py`、`test_trainer_smoke.py`、`test_training_api.py`、`test_ui_static.py`。设计为无 GPU、无 anomalib 导入依赖（烟雾测试使用 AST 解析源码以避免触发 Heavy import）。运行方式：`python -m pytest tests/ -v`。

### UI 架构 (Phase 2)

**默认 UI**: FastAPI + Alpine.js SPA (`modules/ui/server.py` + `modules/ui/static/`)。

- 5 层动效体系：环境光呼吸 → 鼠标光晕跟随 → 滚动驱动动画 → 微交互（胶囊开关/数字跳动/弹簧按钮）→ 视图过渡
- 四页 CSS scroll-snap 全屏吸附（`.snap-container` + `scroll-snap-type: y mandatory`）：算法介绍 → 单模型推理 → 训练工作室 → 四模型对比
- 每页 100dvh 全视口，滚动吸附到整页，无半页停留
- 右侧进度环导航点（SVG 圆环 + 页码"1/4"），实时反映滚动位置
- **进出动画**：统一由 JS WAAPI 驱动（`snapPageEnter` / `snapPageExit`），方向向下推送（与滚动方向一致），CSS 退出动画已删除以避免双动画竞争
- **S1 布局**：三列流水线（上传→选择→推理）→ 完成后收缩为 `.pipeline-summary` 步骤摘要 → 一体化仪表盘卡片 `.result-dashboard`（标题栏 + 全宽对比滑块 + 三列指标行 + 底部判决/操作）
- **S2（训练工作室）布局**：左侧样本上传与参数配置，右侧实时监控曲线与指标面板；上传后生成可排除样本的画廊，训练完成触发全局模型列表刷新
- **S3（S4）布局**：共享原图居中显示（max-height: 200px）→ 单行摘要栏 `.compare-summary-row` → 四列对比网格（顶部横线色标 `.compare-slot-accent`，仅热力图 + 得分/置信度）
- 亮/暗双模式：胶囊开关，localStorage 持久化，`prefers-color-scheme` 系统跟随
- 设计规范：`docs/superpowers/specs/2026-06-19-apple-ui-phase2-design.md`
- 布局精修规范：`docs/superpowers/specs/2026-06-19-ui-layout-polish-design.md`
- Training Studio 规范：`docs/superpowers/specs/2026-06-20-training-studio-design.md`

**回退**: `python scripts/run_ui.py --gradio` 启动原有 Gradio UI（`modules/ui/demo.py`），功能完整保留。

### CSS 陷阱：`.pipeline` Grid 子元素数量

`.pipeline` 使用 `grid-template-columns: 1fr 1fr 1fr` 精确三列布局。**禁止在 `.pipeline` 内添加除 `.pipeline-step` 外的任何子元素**——即使是非 `pipeline-step` 的 div 也会被 grid auto-placement 占据列位，将后续步骤推至第二行。步骤间连接线必须使用 `::after` 伪元素，不能添加 DOM 节点。

### Alpine 陷阱：`x-data` 子作用域访问父属性

`section#s2` 带有 `x-data="compare"`（子作用域），其内的 Alpine 表达式无法直接访问 `app` 作用域的属性（如 `resultData`）。`x-show` / `:src` 等绑定必须使用当前作用域内的属性（如 `compareDone`、`compareSlots`）。

### CSS 陷阱：`.compare-heatmap` 选择器泄漏

`app.css` 中 `.compare-heatmap { position: absolute; }` 为单模型对比滑块设计（热力图叠加在原图之上），该选择器会泄漏到四模型对比槽位。`.compare-slot .compare-heatmap` 覆盖规则若未显式设置 `position: relative`，热力图将脱离文档流，导致父容器 `.compare-heatmap-wrap` 高度塌陷至 0px，配合 `overflow: hidden` 将热力图完全裁剪。**任何对 `.compare-heatmap` 的修改必须验证两种用法**：(1) `.compare-container .compare-heatmap` 滑块叠层，(2) `.compare-slot .compare-heatmap` 对比槽位。

### CSS 陷阱：Snap 进出动画双驱动竞争

进出动画**仅由 JS WAAPI 驱动**（`Anim.snapPageEnter` / `Anim.snapPageExit`）。旧的 CSS `@keyframes pageContentExit` 和 `.snap-page--exiting .snap-page-inner > *` 规则已删除。若重新添加 CSS animation 到 `.snap-page--exiting` 选择器，会与 JS WAAPI 形成双动画竞争，导致元素同时执行两套动画（闪烁/跳变）。`@supports (animation-timeline: view())` 块已注释禁用，待 Chrome 原生 scroll-driven animations 成熟后再评估迁移。

### IntersectionObserver 陷阱：`root` 参数缺失

`.snap-container` 为 `overflow-y: auto` 的滚动容器，section 在其内部滚动。若 `IntersectionObserver` 不指定 `root` 参数，默认以 viewport 为根进行观察——而 `.snap-container` 填满 100dvh 视口，其内部所有 section 对 viewport 均 100% 可见，导致 Observer **永远检测不到 section 切换**。务必在 options 中传入 `root: container`（container 指向 `.snap-container` 元素）。

### CSS 陷阱：`border-image` + `border-radius` 互斥

`border-image` 会完全替代 `border-radius` 的渲染——设置 `border-image` 后圆角静默失效，显示为直角。需要渐变边框+圆角共存时，必须用 `::before` 伪元素 + `mask-composite: exclude` 模拟，而非 `border-image`。

示例：
```css
/* ❌ 错误：border-image 会覆盖 border-radius */
.summary {
    border-radius: 12px;
    border-image: linear-gradient(135deg, gold, orange) 1;
}

/* ✅ 正确：::before + mask-composite */
.summary {
    border-radius: 12px;
    position: relative;
}
.summary::before {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: inherit;
    padding: 1px;
    background: linear-gradient(135deg, gold, orange);
    -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
    -webkit-mask-composite: xor;
    mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
    mask-composite: exclude;
}
```

### UI 调试 (Phase 2)

```bash
# 启动 FastAPI 开发服务器
python scripts/run_ui.py
# → http://127.0.0.1:8000

# 健康检查
curl http://127.0.0.1:8000/api/health

# 模型列表
curl http://127.0.0.1:8000/api/models

# 浏览器控制台验证 snap 状态
document.querySelector('.snap-container').style.scrollSnapType  // "y mandatory"
document.querySelector('.snap-dot-label').textContent           // "1 / 4"
```

**Gradio 6 CSS 作用域问题（仅影响 legacy Gradio UI）**：`gr.Blocks(css=...)` 传入的 CSS 会被 Gradio 6 做选择器作用域处理——在所有选择器前加 `.gradio-container.xxx .contain`。这会导致 `@media` 查询内的 `:root` 选择器失效（变成 `.contain :root`，无法匹配文档根）。**解决方案**：需要通过 CSS `@media` 动态切换的变量（如亮色模式色板），必须通过 `gr.HTML("<style>…</style>")` 注入，绕过 Gradio 的 CSS 处理器。顶层 `:root` 块（暗色默认值）不受影响。

**亮/暗双模式：** 系统通过 `prefers-color-scheme` 自动检测，并支持手动切换。手动选择存储在 `localStorage.theme`，优先级高于系统设定。CSS 通过 `html[data-theme="light"]` 选择器覆盖变量。切换逻辑在 `modules/ui/static/theme.js`。

## 相关文件

- [README.md](README.md) — 完整项目文档，含算法对比表、环境搭建
- [CHANGELOG.md](CHANGELOG.md) — 发布历史，含阶段编号便于追溯
- [data/DATASET_REGISTRY.md](data/DATASET_REGISTRY.md) — 所有数据集清单、已知问题、defect-type 缩写表
- [requirements.txt](requirements.txt) — 固定版本依赖清单
- [configs/config.yaml](configs/config.yaml) — 主配置（所有可调参数集中管理）
- [docs/superpowers/specs/2026-06-21-ui-redesign-design.md](docs/superpowers/specs/2026-06-21-ui-redesign-design.md) — UI 重设计规范（Training Studio + 自训练模型推理）
- [docs/superpowers/plans/2026-06-21-ui-redesign-plan.md](docs/superpowers/plans/2026-06-21-ui-redesign-plan.md) — UI 重设计实现计划（13 任务）
- [docs/superpowers/specs/2026-06-19-apple-ui-phase2-design.md](docs/superpowers/specs/2026-06-19-apple-ui-phase2-design.md) — Phase 2 UI 设计规范（FastAPI + Alpine.js SPA）
- [docs/superpowers/specs/2026-06-19-ui-layout-polish-design.md](docs/superpowers/specs/2026-06-19-ui-layout-polish-design.md) — Phase 2 布局精修规范（药丸按钮、玻璃选择器、四模型对比卡片、进度环导航）
- [docs/superpowers/specs/2026-06-20-training-studio-design.md](docs/superpowers/specs/2026-06-20-training-studio-design.md) — Training Studio 设计规范（上传样本、SSE 训练流、实时监控、排除样本）
- [docs/superpowers/specs/2026-06-18-apple-ui-design-spec.md](docs/superpowers/specs/2026-06-18-apple-ui-design-spec.md) — Phase 1 Apple UI 设计规范（12 组件 + 双模式变量 + 动效参数表）
- [memory/](memory/) — Claude Code 会话记忆目录

---

> **Claude Code 提示**：按 `#` 键可快速把本次会话学到的项目约定沉淀到 `CLAUDE.md`。个人本地偏好请写入 `.claude.local.md` 并加入 `.gitignore`，避免与团队共享的 `CLAUDE.md` 混淆。
