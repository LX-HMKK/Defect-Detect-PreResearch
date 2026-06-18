# results/ 目录 .gitignore 规则设计

> 最后更新: 2026-06-14

## 设计目标

- `results/` 目录 17GB，绝大多数是模型权重 (.ckpt) 和中间数据缓存
- **仅追踪结论性文件** — 即可以直接用于汇报的 JSON、CSV、MD、PNG
- **中间产物全部忽略** — 模型检查点、训练日志、临时数据缓存
- 规则数最少，无 `!` 否定递归目录规则（避免意外追踪子树）

## results/ 目录结构分析

```
results/
├── .gitkeep                          ← 追踪 (空占位)
├── comparison/                       ← 结论性目录
│   ├── *.json (24 个对比结果)
│   ├── *.csv  (对比表格)
│   ├── *.md   (报告)
│   └── post_process/*.json *.md
├── confusion_matrices/               ← 结论性目录
│   ├── *.png (24 个混淆矩阵图)
│   └── *.json (24 个混淆矩阵数据)
├── small_sample/                     ← 混合
│   ├── *.json (汇总)                ← 追踪
│   ├── *.csv  (汇总)                ← 追踪
│   └── _temp_data/                  ← 忽略 (463MB 数据缓存)
├── patchcore/  ← 模型目录 (权重 .ckpt + 训练日志)   ← 全部忽略
├── draem/      ← 同上
├── fre/        ← 同上
├── padim/      ← 同上
└── data_validation_*.md              ← 追踪 (数据验证报告)
```

## 规则方案

```
# 先忽略全部
results/*

# 放行结论性文件
!results/.gitkeep
!results/comparison/
!results/comparison/*.json
!results/comparison/*.csv
!results/comparison/*.md
!results/comparison/post_process/
!results/comparison/post_process/*.json
!results/comparison/post_process/*.md
!results/confusion_matrices/
!results/confusion_matrices/*.png
!results/confusion_matrices/*.json
!results/small_sample/*.json
!results/small_sample/*.csv
!results/data_validation_*.md
```

**关键技巧**: 不对 `small_sample/` 目录本身用 `!` 规则，只对 `small_sample/*.json` 和 `small_sample/*.csv` 放行。这样 `_temp_data/` 子目录自动被 `results/*` 忽略。

## 验证清单

- [ ] `results/patchcore/Patchcore/MVTec/bottle/v3/weights/lightning/model.ckpt` 被忽略
- [ ] `results/small_sample/_temp_data/bottle/train/good/001.png` 被忽略
- [ ] `results/comparison/patchcore_bottle_results.json` 被追踪
- [ ] `results/confusion_matrices/patchcore_bottle_confusion.png` 被追踪
- [ ] `results/small_sample/small_sample_summary.json` 被追踪
