# 工业图像异常检测系统

基于 **anomalib 2.3.0** 的无监督工业异常检测算法复现与性能评测系统。

> **版本**: v1.0.0 | **语言**: Python 3.10 | **框架**: PyTorch 2.x + anomalib 2.3.0 | **UI**: FastAPI + Alpine.js SPA

---

## 问题定义

**工业场景的核心矛盾**：缺陷样本极度稀缺（仅占总产量 1% 甚至更低），无法使用传统有监督学习方法。

**无监督异常检测**：仅使用正常样本训练，检测不符合正常分布的样本为异常。

---

## 算法对比

> 以下为 bottle 数据集代表性结果（完整 6 数据集数据见 `results/comparison/`）。
> 支持 4 种算法：PatchCore、PaDiM、FRE、DRAEM。

| 算法 | 技术路线 | image_AUROC | pixel_AUROC | PRO | 参数量 | 推荐 |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **PatchCore** | 特征建模（检索） | 100% | 98.6% | 80.1% | 24.9M | ✅ 首选 |
| **PaDiM** | 特征建模（概率） | 100% | 98.2% | 80.2% | **2.8M** | ✅ 轻量 |
| **DRAEM** | 自监督判别 | 97.7% | 86.2% | 48.3% | 97.4M | 🔬 备选 |
| **FRE** | 特征重构 | 99.4% | 97.5% | 69.1% | 23.0M | 🔬 备选 |

评测维度：图像级 AUROC/AUPR + 像素级 Pixel AUROC/PRO，共 4 项指标。详见 [最终汇报文档](docs/最终汇报文档.md)。


---

## 快速开始

### 1. 数据处理

```bash
python scripts/run_data_processing.py -i ./data/raw -o ./data/processed/bottle --max_train 150
```

### 2. 模型训练

#### 单数据集训练

```bash
# PatchCore（推荐，精度最高、无需训练）
python scripts/run_training.py -m patchcore -c bottle -d ./data

# PaDiM（轻量级，2.8M 参数，适合边缘部署）
python scripts/run_training.py -m padim -c bottle -d ./data

# FRE 重构法
python scripts/run_training.py -m fre -c bottle -d ./data

# DRAEM（合成异常 + 判别网络）
python scripts/run_training.py -m draem -c bottle -d ./data

# 训练所有算法
python scripts/run_training.py -m all -c bottle -d ./data
```

#### 多数据集训练

```bash
# 训练所有算法到所有数据集（bottle, carpet, region1, region2, region3, region5）
python scripts/run_training.py -m all -c all -d ./data
```

#### 命令行参数

| 参数 | 说明 | 示例 |
|:---|:---|:---|
| `-m, --model` | 模型名称 | `patchcore`, `padim`, `fre`, `draem`, `all` |
| `-c, --category` | 数据类别 | `bottle`, `carpet`, `region1`, `region2`, `region3`, `region5`, `all` |
| `-d, --data_path` | 数据根目录 | `./data` |

#### 训练特性

- **早停机制**：DRAEM/FRE 监控 `val_image_AUROC`，PatchCore 监控 `image_AUROC`。patience=10，在各模型 YAML 的 `early_stopping` 字段中配置

### 3. 评估

```bash
# 评估单个模型
python scripts/run_evaluation.py -m patchcore -c bottle

# 评估所有模型
python scripts/run_evaluation.py -m all -c bottle

# 评估所有模型到所有数据集
python scripts/run_evaluation.py -m all -c all
```

### 4. 启动 UI

```bash
python scripts/run_ui.py
# → http://127.0.0.1:8000（自动打开浏览器）

# 不自动打开浏览器
python scripts/run_ui.py --no-browser

# Gradio 回退 (legacy)
python scripts/run_ui.py --gradio
# → http://127.0.0.1:7860
```

FastAPI + Alpine.js SPA，三页 snap 全屏滚动：算法介绍 → 单模型推理（一体化仪表盘）→ 四模型对比。

### 5. 阈值计算

```bash
# 计算单个模型的最佳阈值
python scripts/run_threshold.py -m patchcore -c bottle

# 计算所有模型的阈值
python scripts/run_threshold.py -m all -c bottle

# 计算所有模型+所有类别并持久化结果
python scripts/run_threshold.py -m all -c all --save
```

阈值搜索结果通过 Youden's J 统计量在 [0,1] 全域搜索，并自动回写至 `results/comparison/{model}_{category}_results.json`。

### 6. 分析工具

```bash
# 小样本鲁棒性分析（N=30/60/100/150）
python tools/run_small_sample.py -m all -c all -d ./data

# 参数消融实验（coreset 采样率/backbone/潜在维度）
python tools/run_ablation.py -m all -c bottle -d ./data

# 推理性能基准测试（速度/显存/参数量）
python tools/run_benchmark.py -m all -c bottle -d ./data

# 混淆矩阵生成（静态图片 + CSV）
python tools/run_confusion_matrix.py -m all -c all

# 数据验证（BMP 伪装 PNG 检测/目录结构/类别分布）
python tools/validate_data.py -d ./data

# 数据集统计分析
python tools/run_data_stats.py -d ./data

# 生成综合实验报告（Markdown）
python tools/run_report.py
```

---

## 界面特性

- **亮/暗双模式**：胶囊开关切换，`prefers-color-scheme` 自动跟随
- **CSS scroll-snap 三页吸附**：算法介绍 → 单模型推理仪表盘 → 四模型对比
- **原图/热力图对比滑块**：拖拽分割线实时对比，Apple 风格异常得分 tooltip
- **SSE 流式推理**：实时进度推送，四模型并行对比
- **5 层动效**：环境呼吸光 → 光标光晕 → 滚动入场 → 微交互 → 视图过渡

---

## 指标说明

| 级别 | 指标 | 用途 |
|:---:|:---:|:---|
| 图像级 | AUROC | 区分正常/异常图片的能力 |
| 图像级 | AUPR | 不平衡数据中的稳定评估 |
| 像素级 | Pixel AUROC | 异常区域定位精度 |
| 像素级 | PRO | 连续异常区域检测能力 |

---

## 配置说明

所有算法参数通过 `configs/` 目录下的 YAML 文件管理：

| 配置文件 | 说明 |
|:---|:---|
| `configs/config.yaml` | 主配置：数据集路径、输出目录、加速器设置 |
| `configs/patchcore.yaml` | PatchCore：backbone、coreset 采样率、特征层 |
| `configs/padim.yaml` | PaDiM：backbone、特征层、协方差正则化 |
| `configs/draem.yaml` | DRAEM：学习率、早停参数、合成异常尺度 |
| `configs/fre.yaml` | FRE：潜在维度、早停轮数、特征层 |

早停机制在各算法 YAML 的 `early_stopping` 字段中配置，支持 `patience`、`min_delta` 和 `monitor_metric` 三个参数。

---

## 项目结构

```
Defect-Detect-PreResearch/
├── modules/
│   ├── _runtime.py            # 共享运行时工具（pycache 重定向、项目根路径）
│   ├── algorithm/             # 模型训练与调度
│   │   ├── trainer.py         # AnomalyDetectionTrainer 核心类 + 工厂函数
│   │   └── _anomalib_compat.py  # anomalib 2.3 ↔ PyTorch Lightning 1.9.5 兼容层
│   ├── config/                # 配置管理
│   ├── data_processing/       # 数据集处理（MVTecFormatter）
│   ├── evaluation/            # 指标计算（AUROC/AUPR/PRO）
│   └── ui/                    # FastAPI + Alpine.js SPA（含 Gradio legacy 回退）
├── configs/                   # 算法 YAML 配置（5 个文件）
├── scripts/                   # 核心工作流脚本（训练/评估/阈值/UI/数据处理）
├── tools/                     # 分析工具（7 个：小样本/消融/基准/混淆矩阵/数据验证/统计/报告）
├── tests/                     # 测试套件（config 单例/metrics 指标/trainer 烟雾测试，无 GPU 依赖）
├── results/                   # 实验输出（comparison/confusion_matrices/等）
├── data/                      # 数据集
├── .cache/                    # 运行时缓存（pycache、日志、预训练权重）
├── docs/                      # 项目文档（任务书/需求/综述/汇报）
├── CLAUDE.md                  # AI Agent 开发指南
└── README.md
```

---

## 依赖

本项目在 **Miniforge 虚拟环境** `anomalib` 中开发。

```bash
# 创建环境（推荐使用 mamba，更快）
mamba create -n anomalib python=3.10 -y
mamba activate anomalib

# 安装依赖（用 mamba/conda 安装 torch，再 pip 安装其他）
mamba install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install anomalib==2.3.0
pip install opencv-python==4.8.1.78 timm
```

### 核心依赖

- anomalib == 2.3.0（固定版本：`trainer.py` 中的 monkey-patch 兼容层依赖此版本。升级 anomalib 前需先更新这些补丁。）
- pytorch >= 2.0 (CUDA 11.8)
- pytorch-lightning == 1.9.5
- opencv-python == 4.8.1.78
- timm

> 完整依赖清单及固定版本见 `requirements.txt`。

---

## 论文参考

| 算法 | 论文 |
|:---|:---|
| **PatchCore** | Roth et al. "Towards Total Recall in Industrial Anomaly Detection" (CVPR 2022) |
| **PaDiM** | Defard et al. "PaDiM: A Patch Distribution Modeling Framework for Anomaly Detection and Localization" (ICPR 2021) |
| **DRAEM** | Zavrtanik et al. "DRAEM — A Discriminatively Trained Reconstruction Embedding for Surface Anomaly Detection" (ICCV 2021) |
| **FRE** | Batzner et al. "Feature Reconstruction Error for Anomaly Detection" (2024) |

本项目基于 [anomalib](https://github.com/openvinotoolkit/anomalib) 深度学习异常检测库实现。

---

## Git 提交规范

```
<类型>(<范围>): <主题>

[可选正文]
```

| 类型 | 说明 | 示例 |
|------|------|------|
| feat | 新功能 | `feat(ui): 添加算法切换功能` |
| fix | 修复 bug | `fix(trainer): 修复显存溢出问题` |
| docs | 文档更新 | `docs: 更新 README` |
| style | 代码格式 | `style: 格式化代码` |
| refactor | 重构 | `refactor: 重构模型配置结构` |
| perf | 性能优化 | `perf(patchcore): 启用预训练权重` |

---

## 许可证

本项目仅用于学术研究目的。
