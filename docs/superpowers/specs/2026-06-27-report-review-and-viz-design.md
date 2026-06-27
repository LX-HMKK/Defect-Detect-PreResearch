# 最终汇报文档审查、训练参数核验与表格可视化改进

> **日期**：2026-06-27
> **类型**：设计规范（spec）
> **范围**：A 改 `docs/最终汇报文档.md` 呈现与数据修复 + B 训练参数最优性审查 + C 三类表格可视化改进
> **关联**：`docs/任务书.md`（验收标准）、`docs/需求.md`（项目要求）

## 一、背景与目标

最终汇报文档（`docs/最终汇报文档.md`）已完成 4 算法 × 6 数据集全覆盖实验，但在三方面存在改进空间：

1. **内容硬伤**：混淆矩阵正文与附录数据冲突、第七章可视化平台描述过时（仍写 Gradio）、image_size 自相矛盾（正文 224 vs 配置 256）。
2. **参数最优性论证不足**：文档描述的早停机制与实际 YAML 配置矛盾（文档写 `val_image_AUROC`，配置实为 `train_loss`）；PaDiM/FRE 的 backbone/latent_dim 取舍未点明；DRAEM 关键参数零消融。
3. **表格呈现信息密度低**：基准对比大表（6×5×4 指标）、消融表、小样本表均用纯数值表格，读者难以一眼看出"参数/算法对指标的影响趋势"。

本 spec 同时解决以上三点，产出可复现的可视化脚本与修订后的文档。

### 验收标准对照

- 任务书「评测完整性」：4 项指标（AUROC/AUPR/Pixel AUROC/PRO）——本设计补全 PRO 在小样本场景的图示（当前缺失）。
- 任务书「小样本验证」：绘制性能随样本量变化曲线——本设计将单指标曲线升级为双轴（AUROC+PRO）折线。
- 需求「企业需要一份能长期参考的综述」——数据冲突修复确保报告可信，可视化提升可读性。

## 二、参数来源确认（关键前提）

文档所用参数 = 直接运行 `scripts/run_training.py` 的默认参数，来源链：

```
scripts/run_training.py（L153-157）
  → 未传 --config 时，默认加载 configs/{model}.yaml
  → 传给 AnomalyDetectionTrainer(config_path=...)
```

**4 份 per-model YAML 是实际生效的参数源**，`configs/config.yaml` 的 `models.{name}` section 为冗余副本（二者一致）。本次参数审查以 4 份 per-model YAML 为准。**不使用 UI 用户自训练结果**。

## 三、设计决策（已与用户逐项确认）

| 决策点 | 选择 | 理由 |
|--------|------|------|
| 改进载体 | A+B（改文档 + 扩展工具脚本） | 不含 UI 平台 |
| 基准对比大表呈现 | 热力图矩阵 | 颜色梯度压缩"算法×数据集×指标"三维信息，强弱分布跃然纸上 |
| 消融表呈现 | 敏感性折线图 | 最能论证"选参合理性"与边际收益递减 |
| 小样本表呈现 | 双轴折线 + 鲁棒性评分卡 | 同时呈现 AUROC 与 PRO 退化趋势，补全当前 PRO 无图的缺口 |
| PRO 配色 | YlOrRd 暖色（单独） | 传达"PRO 是共性瓶颈"的核心结论 |
| 基准热力图布局 | 4 张横向并排 | 信息密度高，便于横向对比指标 |
| 可视化技术栈 | matplotlib 静态 PNG | 与现有 `images/report/*.png` 一致，Markdown 嵌入稳定 |
| DRAEM 消融 | 不做，文档如实说明 | 受限于项目周期与 GPU 资源 |
| 参数配置改动 | 不改 `configs/*.yaml` | 参数审查结论是"PatchCore 最优 / PaDiM·FRE 有意取舍 / DRAEM 如实说明"，无需改配置 |

## 四、A 项：文档内容修复（5 处硬伤）

### A.1 混淆矩阵正文与附录数据冲突

**问题**：5.3 节正文 6 组混淆矩阵数据与附录 E（由 `run_confusion_matrix.py` 生成）多处冲突：

| 组合 | 正文 | 附录 E（为准） |
|------|------|----------------|
| PaDiM @ bottle | TP63/FP0/TN20/FN0「100%」 | TP62/FP0/TN20/FN1「98.8%」 |
| FRE @ bottle | 62/1/19/1 | 55/0/20/8 |
| DRAEM @ bottle | 63/8/12/0 | 60/0/20/3 |
| PatchCore @ region1 | FP1/TN90 | FP8/TN83 |
| PaDiM @ region2 | 17/28/51/10「28 误报」 | 4/0/91/11「0 误报」 |

**修复**：以附录 E 为准重写 5.3 节 6 组数据。正文叙述（如"PaDiM@region2 超过半数正常误报"）须依附录 E 实际为"0 误报、11 漏检、召回 26.7%"重写，避免与附录矛盾。

### A.2 第七章可视化平台整章过时

**问题**：仍写"基于 Gradio 框架""`http://127.0.0.1:7860`"，实际已重构为 FastAPI + Alpine.js SPA（端口 8000），Gradio 退为 legacy。

**修复**：重写 7.1/7.2/7.3 节：
- 7.1：默认 UI 改为 FastAPI + Alpine.js SPA，访问 `http://127.0.0.1:8000`；Gradio 作为 legacy fallback（`--gradio`）。
- 7.2 核心功能表：更新为当前 5 页 snap 架构（算法介绍 → 训练工作室 → 单模型推理 → 四模型对比）。
- 7.3 技术架构：三层改为 FastAPI（REST+SSE）+ Alpine.js SPA + anomalib 推理线程池；附 F.1 环境命令修正（去掉 `gradio`，确保 `anomalib` 在列）。

### A.3 image_size 自相矛盾

**问题**：正文 4.1.3 / 7.3 节写"224×224"，`configs/config.yaml` 与 4 份 per-model YAML 实为 `image_size: [256, 256]`。

**修复**：全文统一为 **256×256**（4.1.3 数据预处理、7.3 推理数据流描述）。

### A.4 "补齐"过渡措辞残留

**问题**：5.1 节并存"PatchCore 已完成全部 6 个数据集"与"在补齐全部 6 个数据集后"，像增量编辑未清理。

**修复**：统一为"24 组对比实验（4 算法 × 6 数据集全覆盖）"最终态，删除"补齐""已完成"等过渡措辞。

### A.5 综合排名与结论措辞对齐

**问题**：平均 AUROC 实为 PaDiM 92.44 > PatchCore 92.22，但 PatchCore 因 Pixel AUROC/PRO 居综合第一。8.1 结论"特征建模类综合最优"未点明依据。

**修复**：5.6 综合排名表与 8.1 结论点明——PatchCore 综合第一依据是 Pixel AUROC（97.94%）与 PRO（51.06%）领先，而非平均图像级 AUROC；PaDiM 平均 AUROC 略高但像素级定位次之。

## 五、B 项：训练参数最优性补强

### B.1 早停机制文档/配置矛盾（核心修复）

**问题**：文档 4.5 节与 CLAUDE.md「早停机制」均称 DRAEM/FRE 监控 `val_image_AUROC`，但 `fre.yaml` / `draem.yaml` 实为 `monitor: train_loss, mode: min`。

**根因**（YAML 注释已点明）：DRAEM/FRE 的评估指标（image_AUROC 等）只在 `test()` 时计算，不在训练时计算，故使用 `train_loss` 作为早停指标。

**修复**：
1. 文档 4.5 节如实修正为 `train_loss`（mode: min），删除"`val_image_AUROC`"描述。
2. 补充说明：`train_loss` 早停仅判断"训练收敛"，**无法防止过拟合**——这是 DRAEM 像素定位 PRO 偏低、小样本下 PRO 暴跌的可能诱因之一，记入 8.2 工作局限性。
3. patience 如实标注：FRE=10、DRAEM=5（文档原统一写 10）。
4. CLAUDE.md「早停机制」段同步修正（说明 val_image_AUROC 不可行的原因）。

### B.2 PaDiM/FRE 参数取舍标注

**问题**：PaDiM `backbone: resnet18`、FRE `latent_dim: 220` 是有意取舍，但文档未点明依据。

**修复**（在 3.3.4 / 3.4.4 关键超参数表后补说明）：
- PaDiM：`resnet18` 是边缘轻量取向（2.8M 参数）；精度优先应换 `wide_resnet50_2`（消融 PRO +5.97%，但参数 25× 至 ~69M）。已在 6.2 场景 B 体现。
- FRE：`latent_dim=220` 图像级分类最优（消融 AUROC 峰值）；定位精度优先应降至 100（PRO +0.93%，信息瓶颈效应）。

### B.3 DRAEM 参数未消融如实说明

**问题**：`draem.yaml` 的 `model.init_args: {}`（全 anomalib 默认），`run_ablation.py` 未覆盖 DRAEM，参数最优性无依据。

**修复**（5.2 节新增"DRAEM 参数说明"小节，如实写明）：
- DRAEM 使用 anomalib 默认参数，未做消融。
- 三点理由：① `model.init_args` 为空 dict 采用库默认；② 早停监控 `train_loss` 仅判收敛；③ 受限于项目周期与 GPU 资源（DRAEM 100 epoch × 多组参数耗时过长）未开展。
- 指向 8.3 未来工作的 DRAEM 调优方向（anomaly_scales / lr 网格搜索）。

### B.4 seed 不一致（次要，记录不改）

patchcore/padim YAML 写 `seed: 0`，fre/draem 写 `seed: 42`，`run_training.py` 默认 42。文档 4.4.2 已写"seed=42"，实际 patchcore/padim 因 YAML 覆盖为 0。本设计**不改配置**（避免影响已发表结果），仅在文档 F.3 可复现性声明补注："各模型 seed 以 per-model YAML 为准"。

## 六、C 项：三类表格可视化改进

### C.1 基准对比热力图矩阵

**文件**：`results/figures/benchmark_heatmap_{metric}.png`（4 张：AUROC / AUPR / PixelAUROC / PRO）

**规格**：
- 每张：行=6 数据集，列=4 算法
- 配色：AUROC/AUPR/PixelAUROC 用顺序型 `Blues`；PRO 单独用 `YlOrRd`（传达"瓶颈"）
- 单元格标注数值（2 位小数），最优列加粗描边
- 4 张横向并排展示

**文档改法**：
- 5.1.1 节：表 5-1（AUROC）+ AUPR 表 → 替换为 AUROC + AUPR 两张热力图
- 5.1.2 节：表 5-2（Pixel AUROC）+ 表 5-3（PRO）→ 替换为对应热力图
- "关键发现"文字保留（是分析非数据复述）
- 附录 A.1 精确数值汇总表保留作查阅底表

### C.2 消融敏感性折线图

**文件**：`results/figures/ablation_{model}_{param}.png`（3 张）

**规格**：
- 每张：横轴=参数值，纵轴=指标百分比，3 条线=Image AUROC（蓝）/ Pixel AUROC（橙）/ PRO（绿）
- 默认参数值处加竖虚线 + 顶部"★ 默认"
- PRO 标全数值，AUROC 全程近 100% 只标端点

**三类消融差异处理**：
- PatchCore coreset_ratio（4 连续值）→ 标准折线，横轴按值均匀
- PaDiM backbone（2 离散值 resnet18/wide_resnet50_2）→ 2 点连线，横轴用类别名，附注"参数量 2.8M→69M"
- FRE latent_dim（3 值 100/220/500）→ 标准折线，附注信息瓶颈效应（PRO 单调下降）

**文档改法**：
- 5.2.1/5.2.2/5.2.3 节：各自数值表 → 替换为折线图
- "分析"文字精简（删复述数值句，保留"为何选默认"论证）
- 附录 B 三张消融数值表保留
- DRAEM：5.2 节新增"B.3 DRAEM 参数说明"（见 B.3）

### C.3 小样本双轴折线 + 鲁棒性评分卡

**文件**：`results/figures/small_sample_dual_axis.png`（1 张，图 + 内嵌评分卡）

**规格**：
- 上半：双轴折线
  - 横轴 N=30/60/100/150
  - 左轴（实线）：Image AUROC，4 算法线（PC 蓝/PaDiM 绿/FRE 橙/DRAEM 紫）
  - 右轴（虚线）：PRO，4 算法线（同色虚线）
  - 数据点标关键值（N=30 与 N=150 两端必标）
  - DRAEM PRO 暴跌段（N=100 谷值 35.5%）加浅红阴影注释"定位退化"
- 下半：鲁棒性评分卡（4 行：算法/N30 AUROC/N150 AUROC/鲁棒性分数/等级）
  - PatchCore 行高亮（★★★★★，分数 1.000）
  - DRAEM 行脚注 ¹（合成异常随机性补偿，沿用文档现有注释）

**文档改法**：
- 5.5.1 AUROC 表 + 5.5.2 PRO 表 → 合并替换为双轴折线 + 评分卡图
- 5.5.3 鲁棒性排序表 → 删除（并入评分卡），保留排序结论文字
- 附录 C.1-C.4 四张分 N 表 → 删除（数据已在图中）
- 附录 C.5 汇总表（带 Δ(N30→N150) 列）→ 保留
- 现有 `images/report/small_sample_curve.png`（单轴 AUROC）→ 被本图取代，原文件保留不删

## 七、工具脚本架构

### 新增 `tools/viz/` 目录

```
tools/viz/
  __init__.py
  _common.py                  # 共享：配色常量、数据加载（从 results/comparison/*.json 读）、字体设置
  benchmark_heatmap.py        # 基准对比热力图（4 张）
  ablation_sensitivity.py     # 消融敏感性折线图（3 张）
  small_sample_dual_axis.py   # 小样本双轴折线（1 张）
  run_all.py                  # 一键生成全部图表 → results/figures/
```

### 扩展现有脚本

- `tools/run_ablation.py`：DRAEM 分支显式注释"未消融，见文档说明"，避免 `all` 时报错。**不改消融逻辑**。
- `tools/run_report.py`：末尾追加图表引用提示（"详见 results/figures/"），不重写表格逻辑。

### 关键设计原则

1. **数据源单一**：可视化脚本从 `results/comparison/*.json` 与 `results/small_sample/` 读真实数据，不从文档反推，保证图表与结果 JSON 一致。
2. **Windows 规范**：每个脚本遵循 `configure_runtime_temp()` + UTF-8 包装 + `PROJECT_ROOT` 绝对路径（CLAUDE.md 强制）。
3. **配色常量统一**：`_common.py` 定义 4 算法标准色（与 UI `--algo-color` 一致：PC #2997ff / PaDiM #30d158 / FRE #ff9f0a / DRAEM #bf5af2），3 指标色（AUROC 蓝 / PixelAUROC 橙 / PRO 绿），全报告视觉统一。
4. **可复现**：`run_all.py` 一键重生成所有图表，输出路径固定。

## 八、文件改动清单

| 文件 | 改动 |
|------|------|
| `docs/最终汇报文档.md` | A.1-A.5 数据修复 + B.1-B.3 参数说明 + C.1-C.3 图表替换 |
| `tools/viz/__init__.py` | 新增（空） |
| `tools/viz/_common.py` | 新增（配色/数据加载/字体） |
| `tools/viz/benchmark_heatmap.py` | 新增（4 张热力图） |
| `tools/viz/ablation_sensitivity.py` | 新增（3 张折线图） |
| `tools/viz/small_sample_dual_axis.py` | 新增（1 张双轴图） |
| `tools/viz/run_all.py` | 新增（一键生成） |
| `tools/run_ablation.py` | DRAEM 分支补注释（不改逻辑） |
| `tools/run_report.py` | 末尾追加图表引用提示 |
| `results/figures/` | 新增目录，存放 8 张生成的图表 |
| `CLAUDE.md` | 「早停机制」段同步修正（B.1） |
| `configs/*.yaml` | **不改**（参数审查只改文档描述） |

## 九、YAGNI 裁剪

- 不做 plotly 交互式（已选 matplotlib）
- 不做 DRAEM 消融实验（已确认）
- 不集成进 UI 平台（范围 A+B，不含 C）
- `run_report.py` 只加图表引用提示，不重写表格逻辑

## 十、验证方式

1. `python tools/viz/run_all.py` 生成全部 8 张图表，确认输出到 `results/figures/`。
2. 逐一核对图表数值与 `results/comparison/*.json` 一致。
3. 通读修订后的 `docs/最终汇报文档.md`，确认 A.1-A.5 数据自洽、B.1-B.3 说明到位、C.1-C.3 图表就位。
4. `python -m pytest tests/ -v` 确保现有测试不受影响（脚本新增不破坏测试）。
