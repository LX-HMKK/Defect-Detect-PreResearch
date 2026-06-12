# 数据集说明

## 概述

本目录包含 6 个数据集（4 个可用），均遵循 MVTec AD 标准目录布局。

## 数据集清单

| 数据集 | 来源 | 状态 | 训练集 | 测试集(正常) | 测试集(缺陷类别) | 格式问题 |
|--------|------|------|--------|-------------|-----------------|----------|
| bottle | MVTec AD 公开数据集 | 完整 | 209 | 20 | 63 (3 类) | 无 |
| carpet | MVTec AD 公开数据集 | 完整 | 280 | 28 | 89 (5 类) | 无 |
| region1 | 企业数据 | 不完整 | 91 | 91 | 7 (4 类) | BMP 伪装 PNG |
| region2 | 企业数据 | 不完整 | 91 | 91 | 15 (4 类) | BMP 伪装 PNG |
| region3 | 企业数据 | 基本完整 | 150 | 150 | 17 (4 类) | 无 |
| region4 | — | **缺失** | — | — | — | 磁盘上从未存在 |
| region5 | 企业数据 | 不完整 | 91 | 91 | 23 (4 类) | BMP 伪装 PNG |

## 目录结构

每个数据集遵循 MVTec AD 标准布局：

```
<dataset>/
├── train/
│   └── good/          # 训练集（仅正常样本）
├── test/
│   ├── good/          # 测试集正常样本
│   └── <defect>/      # 测试集异常样本（按缺陷类型分目录）
└── ground_truth/
    └── <defect>/      # 像素级标注掩码
```

## BMP 伪装 PNG 问题

region1（298 文件）、region2（306 文件）、region5（314 文件）中的图片虽然扩展名为 `.png`，但实际格式为 BMP（Windows 位图，魔数字节 `0x42 0x4D`）。

**影响：** OpenCV (`cv2.imread`) 通过内容检测自动识别格式，当前训练流程不受影响。若将来切换到依赖扩展名的加载器（如某些 web 服务或 PIL 显式指定格式），需先转换。

**验证：** 运行 `python tools/validate_data.py` 可检测所有 BMP 伪装 PNG 文件。

**修复方案：** 转换命令（需数据来源方确认后执行）：
```bash
# 使用 PIL 转换（保留原始文件）
python -c "
from PIL import Image
for f in Path('data/region1/train/good').glob('*.png'):
    img = Image.open(f)
    img.save(f.with_suffix('.png'), 'PNG')
"
```

## 类别不平衡

自定义数据集的测试集严重不平衡：

| 数据集 | 样本分布 |
|--------|----------|
| region1 | lb=1, ps=4, py=1, tl=1 — 3 个类别仅 1 个样本 |
| region2 | lb=2, ps=9, py=3, tl=1 — tl 仅 1 个样本 |
| region3 | lb=9, ps=2, py=1, tl=5 — ps 仅 2 个样本 |
| region5 | lb=9, ps=4, py=8, tl=2 |

样本数 < 5 的类别，其单类别指标（AUROC、AUPR、PRO）在统计上不可靠。汇总报告以宏平均或加权平均为准。

## 数据获取

- **MVTec AD** (bottle, carpet)：[官方网站](https://www.mvtec.com/company/research/datasets/mvtec-ad) 下载，需填写申请表单
- **DTD** (Describable Textures Dataset)：[官方下载](https://www.robots.ox.ac.uk/~vgg/data/dtd/)，DRAEM 训练时用于生成合成异常纹理
- **region1-5**：企业数据，不可公开

## 外部数据集

| 目录 | 说明 |
|------|------|
| `datasets/dtd/` | DTD 纹理数据集，DRAEM 用于合成异常纹理，需手动下载 |
