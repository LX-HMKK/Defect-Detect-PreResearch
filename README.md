# 🔍 工业图像异常检测系统

基于 **anomalib** 实现的三算法对比演示项目，严格按照 **4 大模块** 组织代码。

## 📋 项目概述

从 4 个主流方向中挑选 **3 个最易复现**的算法：

| 算法 | 方向 | 原理 | 复现难度 | 训练时间 |
|:---:|:---:|:---|:---:|:---:|
| **AutoEncoder** | 重构 | 编码器-解码器 + 重构误差 | ⭐⭐ 容易 | ~15分钟 |
| **PatchCore** | 特征建模 | 预训练CNN + 特征记忆库 + 最近邻搜索 | ⭐ 极易 | ~1分钟 |
| **DRAEM** | 自监督学习 | 合成异常 + 判别网络 | ⭐⭐⭐ 中等 | ~30分钟 |

---

## 🗂️ 项目结构（4大模块）

```
anomaly_detection_demo/
│
├── 📁 modules/                          # ===== 核心代码模块 =====
│   ├── 📁 data_processing/             # 模块 1: 数据集处理模块
│   │   └── dataset_formatter.py       #   └─ 转换为 MVTec AD 格式
│   ├── 📁 algorithm/                   # 模块 2: 核心算法复现模块
│   │   └── trainer.py                 #   └─ 训练和测试脚本
│   ├── 📁 evaluation/                  # 模块 3: 指标评测模块
│   │   └── metrics.py                 #   └─ 计算 4 个硬性指标
│   └── 📁 ui/                          # 模块 4: UI 界面演示模块
│       └── demo.py                    #   └─ Gradio 交互界面
│
├── 📁 configs/                          # 算法配置文件
│   ├── autoencoder.yaml               # AutoEncoder 配置
│   ├── patchcore.yaml                 # PatchCore 配置
│   └── draem.yaml                     # DRAEM 配置
│
├── 📁 data/                             # 数据目录
│   ├── 📁 raw/                        # 原始企业数据
│   └── 📁 processed/                  # 处理后数据 (MVTec AD 格式)
│
├── 📁 results/                          # 训练结果
│
├── 🐍 run_data_processing.py          # 入口 1: 数据处理
├── 🐍 run_training.py                 # 入口 2: 模型训练
├── 🐍 run_evaluation.py               # 入口 3: 指标评测
├── 🐍 run_ui.py                       # 入口 4: 启动 UI
│
├── 📄 requirements.txt                  # Python 依赖
├── 📄 README.md                         # 本文件
└── 📄 AGENTS.md                        # Agent 开发指南
```

---

## 🚀 快速开始

### 步骤 1: 环境配置

```bash
# 创建虚拟环境
python -m venv venv

# 激活环境
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 安装依赖
pip install -r requirements.txt
```

### 步骤 2: 模块 1 - 数据集处理

将企业原始数据转换为 MVTec AD 标准格式，**训练集正常样本限制 ≤ 150 张**。

```bash
python run_data_processing.py \
    --input_dir ./data/raw/enterprise_data \
    --output_dir ./data/processed/my_product \
    --max_train 150
```

**支持的输入结构：**
```
# 结构 1: 已分割
raw/
├── train/good/          # 训练集正常样本
└── test/
    ├── good/           # 测试集正常样本
    ├── scratch/        # 异常类型1
    └── dent/           # 异常类型2

# 结构 2: 未分割（自动按比例分割，训练集不超过150张）
raw/
├── good/               # 所有正常样本
└── defect/             # 所有异常样本
```

### 步骤 3: 模块 2 - 核心算法复现

训练 3 种算法，**只用正常样本训练**（无监督设定）。

```bash
# 训练所有算法
python run_training.py \
    --model all \
    --category my_product \
    --data_path ./data/processed/my_product

# 或训练单个算法
python run_training.py --model patchcore --category my_product --data_path ./data/processed/my_product
```

### 步骤 4: 模块 3 - 指标评测

输出 **4 个硬性指标**：
- 【图像级】AUROC, AUPR
- 【像素级】Pixel AUROC, PRO

```bash
python run_evaluation.py \
    --model all \
    --category my_product \
    --results_dir ./results
```

**对比报告生成位置：**
- CSV 表格: `results/comparison/comparison_my_product.csv`
- Markdown 报告: `results/comparison/report_my_product.md`

### 步骤 5: 模块 4 - UI 界面演示

启动 Gradio 交互界面：

```bash
python run_ui.py
```

访问 `http://127.0.0.1:7860`，界面功能：
- ✅ 下拉菜单切换 3 种算法
- ✅ 上传测试图片
- ✅ "开始推理"按钮
- ✅ 并排显示"原图"和"缺陷热力图"

---

## 📊 4 个硬性指标说明

| 级别 | 指标 | 用途 | 范围 |
|:---:|:---:|:---|:---:|
| **图像级** | AUROC | 评估区分正常/异常图片的能力 | 0-1，越接近1越好 |
| **图像级** | AUPR | 在不平衡数据集上的稳定评估 | 0-1，越接近1越好 |
| **像素级** | Pixel AUROC | 评估像素级异常定位精度 | 0-1，越接近1越好 |
| **像素级** | PRO | 评估连续异常区域的检测能力 | 0-1，越接近1越好 |

---

## 💡 各模块详细说明

### 模块 1: 数据集处理

**文件**: `modules/data_processing/dataset_formatter.py`

**功能**:
- 自动检测输入文件夹结构
- 限制训练集正常样本 ≤ 150 张
- 生成 MVTec AD 标准格式
- 自动生成空白掩膜占位符

**约束条件**:
- 训练集只用正常样本
- 正常样本数量 ≤ 150 张
- 支持多种输入结构自动识别

### 模块 2: 核心算法复现

**文件**: `modules/algorithm/trainer.py`

**复现的算法**:

| 算法 | 方向 | 原理 | 特点 |
|:---:|:---:|:---|:---|
| AutoEncoder | 重构 | 编码器-解码器，重构误差 | 经典基线 |
| PatchCore | 特征建模 | 预训练CNN + 记忆库 + KNN | 工业最佳 |
| DRAEM | 自监督 | 合成异常 + 判别网络 | 无需异常样本 |

**约束条件**:
- 基于 anomalib 库，不手写神经网络底层
- 只用正常样本训练
- 输出 4 个硬性指标

### 模块 3: 指标评测

**文件**: `modules/evaluation/metrics.py`

**硬性指标**:
- `compute_image_auroc()` - 图像级 AUROC
- `compute_image_aupr()` - 图像级 AUPR
- `compute_pixel_auroc()` - 像素级 AUROC
- `compute_pro()` - Per-Region Overlap

### 模块 4: UI 界面演示

**文件**: `modules/ui/demo.py`

**界面组件**:
1. 下拉菜单: 切换 3 种算法
2. 上传按钮: 选择/上传样本
3. 开始推理按钮: 执行检测
4. 图片展示区: 并排显示原图和缺陷热力图

---

## ⚡ 完整流程示例

```bash
# 1. 数据处理
python run_data_processing.py \
    -i ./data/raw/my_data \
    -o ./data/processed/product_a \
    --max_train 150

# 2. 修改配置文件中的 category 和 path
# 编辑: configs/autoencoder.yaml, patchcore.yaml, draem.yaml

# 3. 训练所有算法
python run_training.py --model all --category product_a --data_path ./data/processed/product_a

# 4. 查看指标
python run_evaluation.py --model all --category product_a

# 5. 启动 UI
python run_ui.py
```

---

## 📁 配置文件修改

编辑 `configs/` 目录下的 YAML 文件：

```yaml
dataset:
  path: ./data/processed/my_product    # ← 修改为你的数据路径
  category: my_product                  # ← 修改为你的产品类别
```

---

## 📚 依赖项

```bash
pip install anomalib gradio opencv-python numpy pandas scikit-learn
```

详见 `requirements.txt`

---

## 🔧 类型检查与测试

### 类型检查
```bash
# 使用 pyright
pyright modules/
pyright .
```

### 单模块测试
```bash
# 测试数据处理
python -c "from modules.data_processing.dataset_formatter import MVTecFormatter; print('OK')"

# 测试评测指标
python -c "from modules.evaluation.metrics import MetricsEvaluator; print('OK')"

# 测试训练器
python -c "from modules.algorithm.trainer import AnomalyDetectionTrainer; print('OK')"
```

---

## 📧 问题反馈

如有问题，请通过 GitHub Issues 反馈。
