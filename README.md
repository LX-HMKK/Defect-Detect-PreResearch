# 工业图像异常检测

基于 anomalib 2.3.0 的无监督缺陷检测系统。仅用正常样本训练，四种算法覆盖特征建模、重构误差、自监督判别三条技术路线。

> Python 3.10 · PyTorch 2.x · anomalib 2.3.0 · FastAPI + Alpine.js SPA

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

## 快速开始

```bash
mamba activate anomalib
```

### 数据处理

```bash
python scripts/run_data_processing.py -i ./data/raw -o ./data/processed/bottle --max_train 150
```

### 训练

```bash
python scripts/run_training.py -m patchcore -c bottle -d ./data   # 单模型
python scripts/run_training.py -m all -c all -d ./data             # 全部
```

CLI：`-m` 模型 `patchcore|padim|fre|draem|all`，`-c` 类别 `bottle|carpet|region1|2|3|5|all`，`-d` 数据根目录。

早停：DRAEM/FRE 监控 `val_image_AUROC`，在各模型 YAML 的 `early_stopping` 字段配置。

### 评估 & 阈值

```bash
python scripts/run_evaluation.py -m all -c all          # 评估
python scripts/run_threshold.py -m all -c all --save    # 阈值（Youden's J 全域搜索）
```

### UI

```bash
python scripts/run_ui.py                # → http://127.0.0.1:8000
python scripts/run_ui.py --gradio       # legacy 回退 → :7860
```

三页 snap 滚动：算法介绍 → 一体化推理仪表盘 → 四模型并行对比。亮/暗双模式，SSE 流式推理。

### 分析工具

```bash
python tools/run_small_sample.py -m all -c all -d ./data    # 小样本鲁棒性
python tools/run_ablation.py -m all -c bottle -d ./data      # 参数消融
python tools/run_benchmark.py -m all -c bottle -d ./data     # 推理基准
python tools/run_confusion_matrix.py -m all -c all           # 混淆矩阵
python tools/run_post_process_eval.py -m all -c all          # PRO 后处理
python tools/validate_data.py -d ./data                      # 数据验证
python tools/run_data_stats.py -d ./data                     # 统计
python tools/run_report.py                                   # 综合报告
```

## 指标

| 级别 | 指标 | 用途 |
|:---:|:---|:---|
| 图像级 | AUROC / AUPR | 区分正常/异常图片 |
| 像素级 | Pixel AUROC / PRO | 异常区域定位精度 |

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

## 环境

```bash
mamba create -n anomalib python=3.10 -y
mamba activate anomalib
mamba install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install anomalib==2.3.0 opencv-python==4.8.1.78 timm
```

> 固定版本清单见 `requirements.txt`。anomalib 版本锁定（含 PyTorch Lightning 1.9.5 兼容补丁，升级前需更新 `_anomalib_compat.py`）。

---

本项目中 [anomalib](https://github.com/openvinotoolkit/anomalib) 实现，仅用于学术研究。
