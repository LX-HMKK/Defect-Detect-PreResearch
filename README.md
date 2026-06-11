# 工业图像异常检测系统

基于 **anomalib 2.3.0** 的无监督工业异常检测算法复现与性能评测系统。

> **版本**: v1.0.0 | **语言**: Python 3.10 | **框架**: PyTorch 2.x + anomalib 2.3.0 + Gradio 4.x

---

## 问题定义

**工业场景的核心矛盾**：缺陷样本极度稀缺（仅占总产量 1% 甚至更低），无法使用传统有监督学习方法。

**无监督异常检测**：仅使用正常样本训练，检测不符合正常分布的样本为异常。

---

## 算法对比

| 算法 | 原理 | image_AUROC | pixel_AUROC | 推荐 |
|:---|:---|:---:|:---:|:---:|
| **PatchCore** | 特征记忆库 + 最近邻搜索 | 100% | 98.6% | ✅ 首选 |
| **PaDiM** | patch 高斯分布建模 + 马氏距离 | - | - | 🔬 对照 |
| **FRE** | 特征重构误差 | 95% | - | ✅ 备选 |
| **DRAEM** | 合成异常 + 判别网络 | 99.2% | 93.9% | ✅ 备选 |


---

## 快速开始

### 1. 数据处理

```bash
python scripts/run_data_processing.py -i ./data/raw -o ./data/processed/bottle --max_train 150
```

### 2. 模型训练

#### 单数据集训练

```bash
# PatchCore（推荐，最快效果最好）
python scripts/run_training.py -m patchcore -c bottle -d ./data

# FRE 重构法
python scripts/run_training.py -m fre -c bottle -d ./data

# DRAEM（合成异常 + 判别网络）
python scripts/run_training.py -m draem -c bottle -d ./data

# 训练所有算法
python scripts/run_training.py -m all -c bottle -d ./data
```

#### 多数据集训练

```bash
# 训练所有算法到所有数据集（bottle, carpet, region1）
python scripts/run_training.py -m all -c all -d ./data
```

#### 命令行参数

| 参数 | 说明 | 示例 |
|:---|:---|:---|
| `-m, --model` | 模型名称 | `patchcore`, `fre`, `draem`, `all` |
| `-c, --category` | 数据类别 | `bottle`, `carpet`, `region1`, `all` |
| `-d, --data_path` | 数据根目录 | `./data` |

#### 训练特性

- **早停机制**：5 轮无改善则停止训练（基于 train_loss）
- **DRAEM/FRE**：使用验证集 `val_image_AUROC` 监控
- **PatchCore**：使用 `image_AUROC` 监控

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
# 访问 http://127.0.0.1:7860
```

启动工业暗色主题 Web 界面，支持图像上传、模型推理、异常热力图可视化与结果导出。

### 5. 阈值计算

```bash
# 计算单个模型的最佳阈值
python scripts/run_threshold.py -m patchcore -c bottle

# 计算所有模型的阈值
python scripts/run_threshold.py -m all -c bottle

# 导出结果为 JSON
python scripts/run_threshold.py -m patchcore -c bottle --export results/thresholds.json
```

阈值搜索结果保存至 `results/thresholds/`，包含各模型在各数据集上的最佳 F1 阈值。支持 `--export` 参数导出汇总 JSON。

---

## 界面特性

### 工业级暗色模式 UI

- **深色主题**：#121212 背景，专业沉稳
- **Morandi 色系**：钢蓝主按钮、暗红异常告警、深绿正常状态
- **算法选择**：顶部 Tabs 标签页，下划线高亮
- **数据可视化**：
  - 36-48px 大号异常得分
  - 带轨道的现代化进度条 + shimmer 动画
  - 0-1 热力图色阶图例
- **容器质感**：内描边、微妙分割线

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
| `configs/padim.yaml` | PaDiM：backbone、特征层、预训练开关 |
| `configs/draem.yaml` | DRAEM：学习率、早停参数、合成异常强度 |
| `configs/padim.yaml` | PaDiM：特征 backbone、特征层、预训练开关 |
| `configs/fre.yaml` | FRE：重构损失权重、早停轮数、特征维度 |

早停机制在各算法 YAML 的 `early_stopping` 字段中配置，支持 `patience`、`min_delta` 和 `monitor_metric` 三个参数。

---

## 项目结构

```
Defect-Detect-PreResearch/
├── modules/
│   ├── data_processing/    # 数据集处理
│   ├── algorithm/          # 模型训练
│   ├── config/             # 配置管理
│   ├── evaluation/         # 指标计算
│   └── ui/                 # Web 界面 (Gradio)
│       ├── demo.py         # 界面逻辑
│       └── styles.css      # 工业暗色主题样式
├── configs/                # 算法 YAML 配置
│   ├── config.yaml         # 主配置
│   ├── patchcore.yaml
│   ├── draem.yaml
│   └── fre.yaml
├── scripts/                # 核心工作流脚本（训练、评估、UI）
├── tools/                  # 汇报分析工具（报告、统计、基准）
├── assets/                 # 环境配置与依赖
│   ├── requirements.txt
│   ├── setup_miniforge.bat
│   └── pyrightconfig.json
├── data/                   # 数据集
├── datasets/               # 外部数据集 (DTD)
├── results/                # 训练结果
├── pre_trained/            # 预训练权重缓存
├── docs/                   # 项目文档
│   ├── 任务书.md
│   ├── 需求.md
│   └── 讲稿.md
├── AGENTS.md               # AI Agent 开发指南
├── CHANGELOG.md
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
pip install anomalib>=2.0.0
pip install opencv-python==4.8.1.78 timm
```

### 核心依赖

- anomalib >= 2.0.0
- pytorch >= 2.0 (CUDA 11.8)
- opencv-python == 4.8.1.78
- timm

---

## 论文参考

| 算法 | 论文 |
|:---|:---|
| **PatchCore** | Roth et al. "Towards Total Recall in Industrial Anomaly Detection" (CVPR 2022) |
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
