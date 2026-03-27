# 🔍 工业图像异常检测系统

基于 **anomalib** 库实现的无监督工业异常检测算法复现与性能评测系统。

---

## 📋 项目背景与目标

### 项目背景

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

### 项目要求（任务书约束）

| 约束项 | 要求 |
|:---|:---|
| 训练样本 | 仅使用正常样本，数量 ≤ 150 张 |
| 评估指标 | AUROC, AUPR (图像级), Pixel-AUROC, PRO (像素级) |
| 算法覆盖 | 4类方向选3类：重构类、特征建模类、自监督类、预训练大模型类 |

### 技术选型（3类算法）

| 算法 | 方向 | 原理 | 复现难度 | 训练时间 |
|:---:|:---:|:---|:---:|:---:|
| **AutoEncoder** | 重构 | 编码器-解码器 + 重构误差 | ⭐⭐ 容易 | ~15分钟 |
| **PatchCore** | 特征建模 | 预训练CNN + 特征记忆库 + 最近邻搜索 | ⭐ 极易 | ~1分钟 |
| **DRAEM** | 自监督学习 | 合成异常 + 判别网络 | ⭐⭐⭐ 中等 | ~30分钟 |

---

## ⚠️ 数据约束与已知问题

### 关键约束

| 约束 | 说明 | 当前状态 |
|:---|:---|:---|
| 训练样本 ≤ 150 | 任务书要求，仅用正常样本 | ⚠️ 当前 train/good 有 200 张 |
| 数据格式 | MVTec AD 格式 | ✅ 已符合 |
| Ground Truth | 需与测试图片命名匹配 | ⚠️ mask 有 `_mask` 后缀 |

### 已知问题：AUROC = 50%

**现象：** 训练完成但评估指标接近随机基线（0.5）

**可能原因：**
1. 训练样本超限（200 > 150），不符合任务书要求
2. 图像尺寸（409x1421）与 MVTec AD 标准不同，可能在预处理中被裁剪
3. Ground truth mask 与测试图片的匹配可能存在边界对齐问题

**建议：**
- 先用 MVTec AD 公开数据集验证算法基线
- 企业数据需严格通过 `run_data_processing.py` 处理并限制 150 张

---

## 🗂️ 项目结构（4大模块）

```
Defect-Detect-PreResearch/
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
├── 📁 configs/                          # 算法配置文件（YAML）
│   ├── autoencoder.yaml               # AutoEncoder 配置
│   ├── patchcore.yaml                 # PatchCore 配置（可直接运行）
│   └── draem.yaml                     # DRAEM 配置
│
├── 📁 data/                             # 数据目录
│   ├── 📁 region1/                    # 企业数据 region1 (MVTec AD 格式)
│   ├── 📁 region5/                    # 企业数据 region5 (MVTec AD 格式)
│   └── ...
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

## 📁 数据目录结构

### 目录概览

```
data/
├── region1/                    # 企业数据 region1
├── region2/                    # 企业数据 region2
├── region3/                    # 企业数据 region3
├── region5/                    # 企业数据 region5
├── region1_scale/              # 缩放版本
├── region2_scale/
├── region3_scale256/
├── region5_scale/
└── processed/                  # 处理后的数据
    └── my_product/            # 用户自定义产品
```

### regionX 数据格式（已转换为 MVTec AD 变体）

```
regionX/
├── train/
│   └── good/                   # 正常训练样本 (*-OK.png)
├── test/
│   ├── good/                  # 测试正常样本 (*-OK.png)
│   ├── lb/                    # 缺陷类型1 - 漏斑 (*-NG.png)
│   ├── ps/                    # 缺陷类型2 - 凹凸/凸起
│   ├── py/                    # 缺陷类型3 - 移印
│   └── tl/                    # 缺陷类型4 - 划伤/裂纹
└── ground_truth/
    ├── lb/                    # lb 对应 mask（*_mask.png）
    ├── ps/
    ├── py/
    └── tl/
```

### 目录内容说明

| 目录 | 内容 | 命名规则 | 说明 |
|:---|:---|:---|:---|
| `train/good/` | 正常训练样本 | `*-OK.png` | ⚠️ 当前有 200 张，超过任务书 ≤150 限制 |
| `test/good/` | 测试正常样本 | `*-OK.png` | 用于评估正常样本的误报率 |
| `test/lb/` | 漏斑缺陷测试 | `*-NG.png` | Liquid Breakage（漏液） |
| `test/ps/` | 凹凸缺陷测试 | `*-NG.png` | Push/Surface（凸起/凹陷） |
| `test/py/` | 移印缺陷测试 | `*-NG.png` | Print Yield（移印偏移） |
| `test/tl/` | 划伤缺陷测试 | `*-NG.png` | Thin Line（细线/划痕） |
| `ground_truth/xx/` | 对应 mask | `*_mask.png` | 与测试图同名，多 `_mask` 后缀 |

### 命名规范

- **正常图片**: `T{批次}-P{位置}-C{相机}-I{光源}-{设备编号}-{OK/NG}.png`
- **Mask 图片**: 同名 + `_mask` 后缀，如 `T000016-P000016-C6-I2-HXMHE2002M70000J43+83YN-NG_mask.png`

### 注意事项

1. **训练样本限制**: 任务书要求 `train/good/` ≤ 150 张，当前超标需通过 `run_data_processing.py` 处理
2. **Mask 匹配**: mask 文件名比测试图片多 `_mask` 后缀，评估时需注意匹配逻辑
3. **图片后缀**: `-OK.png` = 正常，`-NG.png` = 缺陷

---

## 🚀 快速开始

### 步骤 1: 环境配置

```bash
# 使用 miniforge 创建虚拟环境
"C:\ProgramData\miniforge3\Scripts\conda.exe" create -n anomalib python=3.10 -y

# 激活环境
"C:\ProgramData\miniforge3\Scripts\conda.exe" activate anomalib

# 安装 PyTorch CUDA 版本（RTX 4060 需要 CUDA 11.8）
"C:\ProgramData\miniforge3\Scripts\conda.exe" install -n anomalib -c pytorch -c conda-forge pytorch torchvision pytorch-lightning lightning scikit-learn scipy pandas pillow opencv tqdm pyyaml -y

# 安装其他依赖
"C:\Users\lx_hm\.conda\envs\anomalib\python.exe" -m pip install anomalib==0.7.0 gradio omegaconf "numpy<2" "setuptools==68.0.0" packaging albumentations==1.3.1
```

### 步骤 2: 数据集处理

**重要：** 任务书要求训练样本 ≤ 150 张，必须使用 `run_data_processing.py` 处理数据。

```bash
python run_data_processing.py \
    --input_dir ./data/raw \
    --output_dir ./data/processed/my_product \
    --max_train 150
```

### 步骤 3: 修改 YAML 配置

编辑 `configs/patchcore.yaml` 中的 `run` 配置块：

```yaml
run:
  model: patchcore              # 算法：patchcore / autoencoder / draem
  data_path: ./data             # 数据目录（指向region5的父目录）
  category: region5             # 类别名称
  device: cuda                 # 计算设备：cuda / cpu
  output_dir: ./results         # 输出目录
```

### 步骤 4: 训练

```bash
# 直接运行（从 patchcore.yaml 读取配置）
python run_training.py

# 或命令行覆盖配置
python run_training.py --model patchcore --category region5 --data_path ./data --device cuda
```

### 步骤 5: 评估与 UI

```bash
# 指标评测
python run_evaluation.py --model all --category region5 --results_dir ./results

# 启动 UI
python run_ui.py
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

## ⚡ 直接运行（YAML 配置）

**最简单的运行方式：** 修改 `configs/patchcore.yaml` 中的 `run` 配置块，然后直接运行：

```bash
python run_training.py
```

**配置示例：**
```yaml
run:
  model: patchcore              # 算法
  data_path: ./data            # 数据目录（region5的父目录）
  category: region5             # 类别名称
  device: cuda                 # cuda / cpu
  output_dir: ./results        # 输出目录
```

**切换算法：** 直接修改 `run.model` 为 `autoencoder` 或 `draem`，然后再运行即可。

---

## 📚 依赖项

详见 `requirements.txt`

**核心依赖版本（重要）：**
- `torch==2.2.2+cu118` (CUDA 11.8 for RTX 4060)
- `numpy<2` (numpy 2.x 与 torch 不兼容)
- `anomalib==0.7.0` (不是 1.x 或 2.x)

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
