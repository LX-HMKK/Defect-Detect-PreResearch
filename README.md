# 工业图像异常检测系统

基于 **anomalib 2.x** 库实现的无监督工业异常检测算法复现与性能评测系统。

---

## 项目背景与目标

本项目针对征图Focusight触控面板产线的实际需求，研究无监督异常检测算法在工业场景下的适应性与可用性。

**工业痛点：**
- 缺陷样本极度稀缺（仅占总产量1%甚至更低）
- 缺陷标注成本高昂，需要专业人员
- 缺陷类型多样，存在大量未知缺陷
- 传统有监督方法泛化能力不足

**无监督方法优势：**
- 仅使用正常样本训练，符合工业数据分布特点
- 对未知缺陷具有更好的适应性
- 无需缺陷标注，大幅降低数据准备成本

### 技术选型（3类算法）

| 算法 | 方向 | 原理 | 复现难度 | 训练时间 |
|:---:|:---:|:---|:---:|:---:|
| **Ganomaly** | 重构(GAN) | 训练GAN学习正常样本分布，异常无法良好重构 | ⭐⭐ 中等 | ~20分钟 |
| **PatchCore** | 特征建模 | 预训练CNN + 特征记忆库 + 最近邻搜索 | ⭐ 极易 | ~1分钟 |
| **DRAEM** | 自监督学习 | 合成异常 + 判别网络 | ⭐⭐⭐ 较难 | ~30分钟 |

---

## 项目结构

```
Defect-Detect-PreResearch/
├── modules/
│   ├── data_processing/    # 数据集处理模块
│   ├── algorithm/          # 算法训练与推理
│   ├── evaluation/         # 指标计算
│   └── ui/                 # Gradio 演示界面
├── configs/                 # YAML 算法配置
├── data/                    # 数据目录 (MVTec AD 格式)
├── results/                # 训练结果
├── docs/                   # 任务书和需求文档
├── run_*.py               # 入口脚本
├── requirements.txt
├── README.md
└── AGENTS.md              # Agent 开发指南
```

---

## 快速开始

### 环境配置

```bash
# 使用 miniforge 创建虚拟环境
"C:\ProgramData\miniforge3\Scripts\conda.exe" create -n anomalib python=3.10 -y

# 安装 PyTorch CUDA 版本
"C:\ProgramData\miniforge3\Scripts\conda.exe" install -n anomalib -c pytorch -c nvidia pytorch torchvision pytorch-cuda=11.8 -y

# 安装其他依赖
"C:\ProgramData\miniforge3\Scripts\conda.exe" install -n anomalib -c conda-forge scikit-learn scipy pandas pillow opencv tqdm pyyaml -y

# 安装 anomalib 2.x
"C:\Users\lx_hm\.conda\envs\anomalib\python.exe" -m pip install "anomalib>=2.0.0" --upgrade

# 安装其他依赖
"C:\Users\lx_hm\.conda\envs\anomalib\python.exe" -m pip install timm opencv-python==4.8.1.78 --force-reinstall --no-deps
```

### 数据处理

```bash
python run_data_processing.py --input_dir ./data/raw --output_dir ./data/processed/my_product --max_train 150
```

### 训练

```bash
# 训练单个算法
python run_training.py --model patchcore --category bottle --data_path ./data --device cuda

# 训练所有算法
python run_training.py --model all --category bottle --data_path ./data --device cuda
```

### 评估

```bash
python run_evaluation.py --model all --category bottle --results_dir ./results
```

### UI 演示

```bash
python run_ui.py
# 访问 http://127.0.0.1:7860
```

---

## 支持的算法

| 算法 | 方向 | image_AUROC | pixel_AUROC | 状态 |
|:---:|:---:|:---:|:---:|:---:|
| Ganomaly | GAN重构 | 47.46% | 0% | ⚠️ 效果较差 |
| PatchCore | 特征建模 | 85.71% | 89.63% | ✅ 推荐 |
| DRAEM | 自监督学习 | 94.05% | 71.39% | ✅ 很好 |

---

## 4个硬性指标

| 级别 | 指标 | 用途 |
|:---:|:---:|:---|
| 图像级 | AUROC | 区分正常/异常图片的能力 |
| 图像级 | AUPR | 不平衡数据中的稳定评估 |
| 像素级 | Pixel AUROC | 像素级异常定位精度 |
| 像素级 | PRO | 连续异常区域检测能力 |

---

## 依赖项

- anomalib >= 2.0.0
- pytorch >= 2.0 (CUDA 11.8)
- numpy >= 1.24.0
- opencv-python == 4.8.1.78
- timm

---

## 类型检查

```bash
pyright modules/
```
