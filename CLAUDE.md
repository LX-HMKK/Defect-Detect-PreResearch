# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在本仓库中工作时提供指导。仅保留高频必需内容；细则与低频内容见文末「参考文件」指针。

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

# 从已训练 checkpoint 计算最优阈值
python scripts/run_threshold.py -m patchcore -c bottle
python scripts/run_threshold.py -m all -c all --save

# 评估（加载已保存的 JSON 结果）
python scripts/run_evaluation.py -m patchcore -c bottle
python scripts/run_evaluation.py -m all -c all

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

> 环境配置（mamba/pip 安装）与低频分析工具命令（小样本/消融/混淆矩阵/数据验证/PRO 后处理/基准/报告/统计）见 [docs/COMMANDS.md](docs/COMMANDS.md)。

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

## 架构

```
scripts/run_*.py              # 入口脚本（CLI 轻量封装）
modules/
  _runtime.py                  # 共享运行时（pycache 重定向、resolve_project_path、get_runtime_cache_dir）
  algorithm/
    trainer.py                 # 核心：AnomalyDetectionTrainer + 模型/数据模块工厂函数
    _anomalib_compat.py        # anomalib 2.3.0 ↔ PyTorch Lightning 1.9.5 猴子补丁兼容层（勿移除，见 CODING.md）
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
        app.css                # Phase 2 主样式表（亮/暗双模式，~3300 行）
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
        hero-fluid.js          # 首页 hero 液态扭曲封面（WebGL2 grid-distortion，鼠标驱动；标题画进纹理被低分辨率位移场扭曲）
        hero-visual.js         # [Legacy] 原 SVG 流程图 hero 动画，已被 hero-fluid.js 取代；文件保留但 index.html 不再引用
        algo-carousel.js       # 首页算法轮播（AirPods Pro 风格推近 + closeness 驱动缩放/模糊 + 进度条）
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

### 关键类（一句话职责）

- **`AnomalyDetectionTrainer`** (`algorithm/trainer.py`) — 核心调度器：`setup()` 建 datamodule+model，`train()` 跑 `Engine.fit()`，`evaluate()` 跑 `Engine.test()` 并用 Youden's J 在 [0,1] 全域搜最优阈值，写 JSON 至 `results/comparison/`。
- **`get_model_from_config()` / `get_datamodule_from_config()`** — 严格从 YAML 读参，缺失键抛 `ValueError`；前者始终附加含 6 指标的 `Evaluator`，后者自动检测 MVTec AD vs Folder 格式。
- **`ConfigManager`** (`config/manager.py`) — 基于 `config.yaml` 的单例，点号取值 `get('data.image_size')`。
- **`MetricsEvaluator`** (`evaluation/metrics.py`) — 从零实现 AUROC/AUPR/PRO（sklearn+scipy），用于离线评估与单测。
- **`MVTecFormatter`** (`data_processing/dataset_formatter.py`) — 企业原始图 → MVTec AD 布局，letterbox 缩放。
- **`AnomalyMapProcessor`** (`evaluation/post_processor.py`) — 异常热力图后处理，4 基础算子 + 7 组合配方（`PRESET_CONFIGS`）。
- **Training Studio 后端** (`ui/training_backend.py`) — `TrainingTaskManager` 单任务锁 + `format_uploaded_samples` + `run_training_job`（线程内训练，SSE 推指标）。
- **`_runtime.configure_runtime_temp()`** — pycache 重定向至 `./.cache/pycache`，所有 `run_*.py` 开头调用。

### 数据流

1. 原始图 → `MVTecFormatter` → MVTec AD (`train/good/`, `test/<defect>/`, `ground_truth/<defect>/`)
2. `AnomalyDetectionTrainer` 解析 `configs/{model}.yaml` → `setup()` 建模型+数据模块 → `train()` 跑 `Engine.fit()`（PatchCore/PaDiM 单 epoch 特征提取/高斯建模）
3. `evaluate()` → `Engine.test()` 提取 6 指标 → Youden's J 最优阈值 → 写 `results/comparison/{model}_{category}_results.json`
4. UI 经 FastAPI `/api/predict`、`/api/compare` SSE 流式推理（`source=pretrained` 读预训练 / `source=self_trained` 读用户训练目录）
5. Training Studio：上传正常样本 → `/api/upload-samples` 整理临时目录 → `/api/train` 在任务锁下训练 → SSE 推 loss/AUROC/ETA → checkpoint 重写为安全格式

### 三层配置优先级

`AnomalyDetectionTrainer` 读参：1) 显式 `config_path` (YAML) > 2) `config.yaml` 的 `models.{name}` > 3) 抛 `ValueError`。两个工厂函数同此模式。

### 早停机制

DRAEM/FRE 监控 `train_loss`（mode: min，patience FRE=10/DRAEM=5，在各 `{model}.yaml` 配置）——评估指标只在 `test()` 算，无法用 `val_image_AUROC` 早停，`train_loss` 仅判收敛不防过拟合。PatchCore/PaDiM 单 epoch 不需早停。

## 编码规范要点

- **文档语言**：所有文档/注释/提交信息用中文。
- **导入顺序**：标准库 → 第三方 → anomalib → 本地；**`import cv2` 必须在任何 anomalib 导入前**，否则 Windows DLL 加载失败。
- **命名**：类 PascalCase、函数变量 snake_case、常量 UPPER_SNAKE_CASE、私有 `_` 前缀。
- **类型注解**：必用，Union 用 `|`（如 `str | Path`）。
- **错误处理**：捕获具体异常，禁止空 `except`。

> 完整规范与代码示例见 [docs/CODING.md](docs/CODING.md)。

## Git 提交规范

Angular 协议：`<类型>(<范围>): <主题>`，主题 ≤72 字符，命令式语气。**禁止 `Co-authored-by`**。**多行提交信息必须 `-F` 从文件读取**（Bash/PowerShell 双 shell 下 here-string 易误用致 `@` 混入）：写临时文件 `.git-msg` → `git commit -F .git-msg` → 删除。

> 类型表（feat/fix/docs/style/refactor/perf）与自检命令见 [docs/CODING.md](docs/CODING.md)。

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

各模型 YAML 配置路径传入 `AnomalyDetectionTrainer` 时优先使用；否则回退 `ConfigManager.get()` 从 `config.yaml` 读。配置键缺失抛 `ValueError` 而非静默用默认值。

### 测试

测试套件位于 `tests/`，共 6 个文件：`test_config.py`、`test_metrics.py`、`test_trainer_smoke.py`、`test_training_api.py`、`test_ui_static.py`、`test_viz.py`。无 GPU、无 anomalib 导入依赖（烟雾测试用 AST 解析源码避免触发 Heavy import）。运行：`python -m pytest tests/ -v`。

> Trainer 兼容性补丁、自训练 checkpoint 安全格式、Phase 2 UI 架构详解、全部 CSS/Alpine/IO 陷阱、UI 调试命令见 [docs/CODING.md](docs/CODING.md)。

## 参考文件

日常必需（高频查阅）：
- [README.md](README.md) — 完整项目文档，含算法对比表、环境搭建
- [CHANGELOG.md](CHANGELOG.md) — 发布历史，含阶段编号便于追溯
- [configs/config.yaml](configs/config.yaml) — 主配置（所有可调参数集中管理）
- [requirements.txt](requirements.txt) — 固定版本依赖清单
- [data/DATASET_REGISTRY.md](data/DATASET_REGISTRY.md) — 所有数据集清单、已知问题、defect-type 缩写表
- [memory/](memory/) — Claude Code 会话记忆目录

抽离参考（按需查阅，勿通读）：
- [docs/COMMANDS.md](docs/COMMANDS.md) — 环境配置 + 低频分析工具命令
- [docs/CODING.md](docs/CODING.md) — 编码规范细则与示例 + Git 提交规范 + Phase 2 UI 架构详解 + 全部陷阱 + UI 调试

---

> **Claude Code 提示**：按 `#` 键可快速把本次会话学到的项目约定沉淀到 `CLAUDE.md`。个人本地偏好请写入 `.claude.local.md` 并加入 `.gitignore`，避免与团队共享的 `CLAUDE.md` 混淆。
