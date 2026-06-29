# 工业图像异常检测

基于 anomalib 2.3.0 的无监督缺陷检测系统。仅用正常样本训练，四种算法覆盖特征建模、重构误差、自监督判别三条技术路线。

> Python 3.10 · PyTorch 2.x · anomalib 2.3.0 · FastAPI + Alpine.js SPA

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white" alt="Python 3.10">
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch 2.x">
  <img src="https://img.shields.io/badge/anomalib-2.3.0-00A4EF" alt="anomalib 2.3.0">
  <img src="https://img.shields.io/badge/FastAPI-框架-009688?logo=fastapi&logoColor=white" alt="FastAPI">
</p>

## 算法

*bottle 数据集代表结果，完整数据见 `results/comparison/`*

| 算法 | 路线 | image_AUROC | pixel_AUROC | PRO | 参数量 |
|:---|:---|:---:|:---:|:---:|:---:|
| **PatchCore** | 特征记忆库 + 最近邻 | 100% | 98.6% | 80.1% | 24.9M |
| **PaDiM** | patch 高斯建模 + 马氏距离 | 100% | 98.2% | 80.2% | 2.8M |
| **FRE** | 特征重构误差 | 99.4% | 97.5% | 69.1% | 23.0M |
| **DRAEM** | 合成异常 + 判别网络 | 97.7% | 86.2% | 48.3% | 97.4M |

| 论文 | 来源 |
|:---|:---|
| PatchCore | Roth et al., CVPR 2022 |
| PaDiM | Defard et al., ICPR 2021 |
| DRAEM | Zavrtanik et al., ICCV 2021 |
| FRE | Batzner et al., 2024 |

## 安装

```bash
mamba create -n anomalib python=3.10 -y
mamba activate anomalib
mamba install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install -r requirements.txt
```

> 依赖版本已锁定在 `requirements.txt`。anomalib 2.3.0 配套 PyTorch Lightning 1.9.5 兼容补丁位于 `modules/algorithm/_anomalib_compat.py`，升级 anomalib 前请同步更新。

验证安装：

```bash
python -m pytest tests/ -v
python -c "from modules.algorithm.trainer import AnomalyDetectionTrainer; print('OK')"
```

## 快速开始

```bash
mamba activate anomalib
```

### 数据处理

将原始图像整理为 MVTec AD 目录结构：`train/good/` 存放正常样本，`test/<defect>/` 存放测试样本，`ground_truth/<defect>/` 存放掩码（可选）。`--max_train` 限制训练集数量以快速验证。

```bash
python scripts/run_data_processing.py -i ./data/raw -o ./data/processed/bottle --max_train 150
```

### 训练

```bash
python scripts/run_training.py -m patchcore -c bottle -d ./data   # 单模型
python scripts/run_training.py -m all -c all -d ./data             # 全部
```

CLI：`-m` 模型 `patchcore|padim|fre|draem|all`，`-c` 类别 `bottle|carpet|region1|region2|region3|region5|all`，`-d` 数据根目录。

早停：DRAEM/FRE 监控 `train_loss`（mode: min），在各模型 YAML 的 `early_stopping` 字段配置。PatchCore/PaDiM 单 epoch 无需早停。

### 评估 & 阈值

`run_evaluation.py` 读取已保存的 JSON 结果并打印 6 项指标；`run_threshold.py` 用 Youden's J 在 `[0,1]` 全域搜索最优分类阈值，`--save` 会回写结果文件。

```bash
python scripts/run_evaluation.py -m all -c all          # 评估
python scripts/run_threshold.py -m all -c all --save    # 阈值（Youden's J 全域搜索）
```

### UI

启动 FastAPI 服务后自动打开浏览器，默认地址 `http://127.0.0.1:8000`。

```bash
python scripts/run_ui.py                # → http://127.0.0.1:8000
python scripts/run_ui.py --gradio       # legacy 回退 → :7860
```

界面为四页 snap 滚动：
1. **Hero / 算法轮播** - WebGL 液态扭曲封面 + AirPods Pro 风格算法卡片横向轮播
2. **Training Studio** - 上传正常样本、调参、自训练模型，SSE 实时推送 loss/AUROC/ETA
3. **单模型推理** - 选择预训练或自训练模型，SSE 流式返回热力图、bbox 与得分
4. **四模型对比** - 2×2 对比墙，画廊式网格与得分数字滚动

支持亮/暗双模式、`prefers-reduced-motion` 降级，以及 `source=pretrained|self_trained` 模型来源切换。

### 分析工具

`tools/` 下提供训练后分析脚本；`tools/viz/` 为可视化套件，可一键生成论文/汇报所需的 8 张图表。

```bash
python tools/run_small_sample.py -m all -c all -d ./data    # 小样本鲁棒性
python tools/run_ablation.py -m all -c bottle -d ./data      # 参数消融
python tools/run_benchmark.py -m all -c bottle -d ./data     # 推理基准
python tools/run_confusion_matrix.py -m all -c all           # 混淆矩阵
python tools/run_post_process_eval.py -m all -c all          # PRO 后处理
python tools/validate_data.py -d ./data                      # 数据验证
python tools/run_data_stats.py -d ./data                     # 统计
python tools/run_report.py                                   # 综合报告
python tools/viz/run_all.py                                  # 一键生成 8 张图表
```

## 指标

评估指标分为图像级与像素级：图像级指标判断整张图片是否异常；像素级指标衡量异常区域定位的精细程度。

| 级别 | 指标 | 用途 |
|:---:|:---|:---|
| 图像级 | AUROC / AUPR | 区分正常/异常图片 |
| 像素级 | Pixel AUROC / PRO | 异常区域定位精度 |

- **AUROC**：受试者工作特征曲线下面积，衡量分类排序能力。
- **AUPR**：精确率-召回率曲线下面积，在类别不平衡时更稳定。
- **PRO**：Per-Region Overlap，评估异常区域与预测掩码的重叠，关注定位连续性。

## 配置

`configs/` 下 5 个 YAML 文件集中管理所有参数：

| 文件 | 管辖 |
|:---|:---|
| `config.yaml` | 路径、输出目录、加速器 |
| `patchcore.yaml` | backbone、coreset 采样率 |
| `padim.yaml` | backbone、协方差正则化 |
| `fre.yaml` | 潜在维度、早停 |
| `draem.yaml` | 学习率、合成异常尺度 |

## 结构

```
├── modules/algorithm/    # 训练调度 + anomalib 兼容层
├── modules/evaluation/   # AUROC/AUPR/PRO 指标
├── modules/ui/           # FastAPI + Alpine.js SPA
├── configs/              # YAML 配置
├── scripts/              # 入口脚本
├── tools/                # 分析工具
├── tests/                # 无 GPU 依赖测试
├── results/              # 实验输出
└── data/                 # 数据集
```

## 相关文档

- `CLAUDE.md` - 高频约定、命令速查与项目现状
- `docs/COMMANDS.md` - 环境配置与低频分析工具命令
- `docs/CODING.md` - 编码规范、Git 提交规范与 Phase 2 UI 架构详解
- `data/DATASET_REGISTRY.md` - 数据集清单、缺陷类型缩写与已知问题

## 引用

若在你的研究中使用本项目，请引用对应的算法论文：

- PatchCore: Roth et al., "Towards Total Recall in Industrial Anomaly Detection", CVPR 2022
- PaDiM: Defard et al., "PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization", ICPR 2021
- DRAEM: Zavrtanik et al., "DRAEM - A Discriminatively Trained Reconstruction Embedding for Surface Anomaly Detection", ICCV 2021
- FRE: Batzner et al., "Beyond PatchCore: FRee Resolution for Anomaly Detection and Localization", 2024

## 许可

本项目基于 [anomalib](https://github.com/openvinotoolkit/anomalib) 实现，仅用于学术研究。
