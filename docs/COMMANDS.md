# 命令与环境参考

> 本文件从 `CLAUDE.md` 抽离：环境配置（一次性搭建）与低频分析工具命令。
> 高频命令（训练/评估/阈值/UI/测试）仍在 `CLAUDE.md`。

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

固定版本依赖清单见 [requirements.txt](../requirements.txt)。

## 低频分析工具命令

```bash
# 小样本鲁棒性分析（30/60/100/150 张正常样本）
python tools/run_small_sample.py -m all -c all -d ./data

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
```

CLI 参数：`-m` 模型 (`patchcore|padim|fre|draem|all`)，`-c` 类别 (`bottle|carpet|region1|region2|region3|region5|all`)，`-d` 数据根目录（始终指向类别父目录 `./data`）。
