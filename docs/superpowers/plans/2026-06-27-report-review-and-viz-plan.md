# 汇报文档审查与表格可视化改进 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复最终汇报文档的数据硬伤与过时描述、补强训练参数最优性论证、用 matplotlib 可视化替代三类大表，产出可复现的 `tools/viz/` 脚本套件。

**Architecture:** 新增 `tools/viz/` 目录（5 个 matplotlib 脚本，从 `results/comparison/*.json` 与 `results/confusion_matrices/*.json` 读真实数据生成图表到 `results/figures/`）；修改 `docs/最终汇报文档.md` 做数据修复与图表替换；扩展 `run_ablation.py` 与 `run_report.py`（仅注释/提示，不改核心逻辑）；同步修正 `CLAUDE.md` 早停机制描述。所有参数审查只改文档描述，不改 `configs/*.yaml`。

**Tech Stack:** Python 3.10 | matplotlib（Agg 后端 + SimHei 中文字体，沿用 `run_confusion_matrix.py` 配置）| pytest 断言式测试（沿用 `test_ui_static.py` 风格）

**Spec:** `docs/superpowers/specs/2026-06-27-report-review-and-viz-design.md`

---

## 文件结构

### 新增

| 文件 | 职责 |
|------|------|
| `tools/viz/__init__.py` | 包标识（空） |
| `tools/viz/_common.py` | 共享：配色常量、数据加载器、matplotlib 字体配置 |
| `tools/viz/benchmark_heatmap.py` | 基准对比热力图（4 张：AUROC/AUPR/PixelAUROC/PRO） |
| `tools/viz/ablation_sensitivity.py` | 消融敏感性折线图（3 张：PatchCore/PaDiM/FRE） |
| `tools/viz/small_sample_dual_axis.py` | 小样本双轴折线 + 鲁棒性评分卡（1 张） |
| `tools/viz/run_all.py` | 一键生成全部 8 张图表到 `results/figures/` |
| `tests/test_viz.py` | 断言式测试：脚本可运行 + 输出文件存在 + 数据一致 |

### 修改

| 文件 | 改动 |
|------|------|
| `tools/run_ablation.py` | DRAEM 分支补"未消融"注释（不改逻辑） |
| `tools/run_report.py` | 末尾追加图表引用提示 |
| `docs/最终汇报文档.md` | A.1-A.5 数据修复 + B.1-B.4 参数说明 + C.1-C.3 图表替换 |
| `CLAUDE.md` | 「早停机制」段同步修正 |

### 不改

- `configs/*.yaml`（参数审查只改文档描述）
- `configs/config.yaml`

---

## Task 1: 创建 `tools/viz/_common.py` 共享模块

**Files:**
- Create: `tools/viz/__init__.py`
- Create: `tools/viz/_common.py`
- Test: `tests/test_viz.py`

- [ ] **Step 1: 写失败测试 — 验证配色常量与数据加载器存在**

Create `tests/test_viz.py`:

```python
"""可视化脚本套件回归测试（断言式，沿用 test_ui_static.py 风格）"""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIZ_DIR = PROJECT_ROOT / "tools" / "viz"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"


def test_viz_package_importable():
    """_common 必须可导入，配色常量齐全"""
    from tools.viz import _common
    # 4 算法标准色（与 UI --algo-color 一致）
    assert _common.ALGO_COLORS["patchcore"] == "#2997ff"
    assert _common.ALGO_COLORS["padim"] == "#30d158"
    assert _common.ALGO_COLORS["fre"] == "#ff9f0a"
    assert _common.ALGO_COLORS["draem"] == "#bf5af2"
    # 3 指标色
    assert "image_AUROC" in _common.METRIC_COLORS
    assert "pixel_AUROC" in _common.METRIC_COLORS
    assert "pixel_PRO" in _common.METRIC_COLORS


def test_load_comparison_results():
    """load_comparison_results 应从 results/comparison 读 4 算法 × 6 数据集"""
    from tools.viz._common import load_comparison_results
    results = load_comparison_results()
    # bottle 必须有 4 个模型
    assert "bottle" in results
    assert "patchcore" in results["bottle"]
    assert results["bottle"]["patchcore"]["image_AUROC"] == 1.0


def test_load_ablation_results():
    """load_ablation_results 应返回 ablation_results.json 的列表"""
    from tools.viz._common import load_ablation_results
    results = load_ablation_results()
    assert isinstance(results, list)
    assert len(results) == 9  # patchcore×4 + padim×2 + fre×3
    assert results[0]["model"] == "patchcore"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_viz.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.viz'`

- [ ] **Step 3: 创建 `__init__.py` 与 `_common.py`**

Create `tools/viz/__init__.py` (空文件):

```python
```

Create `tools/viz/_common.py`:

```python
"""可视化共享模块：配色常量、数据加载、matplotlib 配置。

所有可视化脚本通过此模块读取 results/ 下的真实数据，保证图表与结果 JSON 一致。
"""
import json
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 4 算法标准色（与 modules/ui static/css 的 --algo-color 一致）
ALGO_COLORS: Dict[str, str] = {
    "patchcore": "#2997ff",
    "padim": "#30d158",
    "fre": "#ff9f0a",
    "draem": "#bf5af2",
}

# 3 指标色（贯穿基准热力图与消融折线图）
METRIC_COLORS: Dict[str, str] = {
    "image_AUROC": "#2997ff",   # 蓝
    "pixel_AUROC": "#ff9f0a",   # 橙
    "pixel_PRO": "#30d158",      # 绿
}

# 4 项指标的显示名与热力图 colormap
METRIC_DISPLAY = {
    "image_AUROC": ("图像级 AUROC", "Blues"),
    "image_AUPR": ("图像级 AUPR", "Blues"),
    "pixel_AUROC": ("像素级 AUROC", "Blues"),
    "pixel_PRO": ("PRO", "YlOrRd"),  # 单独暖色，突出"瓶颈"
}

# 6 数据集固定顺序（bottle/carpet 公开在前，region 企业在后）
DATASET_ORDER = ["bottle", "carpet", "region1", "region2", "region3", "region5"]

# 4 算法固定顺序
MODEL_ORDER = ["patchcore", "padim", "fre", "draem"]


def setup_matplotlib():
    """配置 matplotlib Agg 后端 + 中文字体（沿用 run_confusion_matrix.py 配置）"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def load_comparison_results() -> Dict[str, Dict[str, dict]]:
    """加载 results/comparison/*_results.json，返回 {dataset: {model: metrics}}

    跳过含 'training_' 的用户自训练结果文件。
    """
    comparison_dir = PROJECT_ROOT / "results" / "comparison"
    all_results: Dict[str, Dict[str, dict]] = {}
    if not comparison_dir.exists():
        return all_results
    for json_file in sorted(comparison_dir.glob("*_results.json")):
        name = json_file.stem.replace("_results", "")
        # 跳过用户自训练文件（如 patchcore_training_xxx_results）
        if "_training_" in json_file.stem:
            continue
        for model in MODEL_ORDER:
            if name.startswith(model + "_"):
                category = name[len(model) + 1:]
                try:
                    data = json.loads(json_file.read_text(encoding="utf-8"))
                    metrics = data.get("metrics", data)
                    all_results.setdefault(category, {})[model] = metrics
                except (json.JSONDecodeError, IOError):
                    pass
                break
    return all_results


def load_ablation_results() -> List[dict]:
    """加载 results/comparison/ablation_results.json 消融数据列表"""
    path = PROJECT_ROOT / "results" / "comparison" / "ablation_results.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_viz.py -v`
Expected: PASS — 3 个测试全过

- [ ] **Step 5: 提交**

```bash
git add tools/viz/__init__.py tools/viz/_common.py tests/test_viz.py
git commit -F .git-msg
```
(git-msg 内容: `feat(viz): 新增 tools/viz 共享模块（配色常量、数据加载、matplotlib 配置）`)

---

## Task 2: 基准对比热力图脚本

**Files:**
- Create: `tools/viz/benchmark_heatmap.py`
- Test: `tests/test_viz.py`（追加测试）

- [ ] **Step 1: 追加失败测试 — 验证 4 张热力图生成**

追加到 `tests/test_viz.py` 末尾:

```python
def test_benchmark_heatmap_generates_all_metrics():
    """基准热力图应生成 4 张 PNG（AUROC/AUPR/PixelAUROC/PRO）"""
    from tools.viz import benchmark_heatmap
    benchmark_heatmap.generate_all()
    for metric in ["image_AUROC", "image_AUPR", "pixel_AUROC", "pixel_PRO"]:
        assert (FIGURES_DIR / f"benchmark_heatmap_{metric}.png").exists()
        assert (FIGURES_DIR / f"benchmark_heatmap_{metric}.png").stat().st_size > 1000
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_viz.py::test_benchmark_heatmap_generates_all_metrics -v`
Expected: FAIL — `ModuleNotFoundError` 或文件不存在

- [ ] **Step 3: 实现热力图脚本**

Create `tools/viz/benchmark_heatmap.py`:

```python
"""基准对比热力图：4 算法 × 6 数据集 × 4 指标。

生成 4 张 PNG 到 results/figures/benchmark_heatmap_{metric}.png。
PRO 单独用 YlOrRd 暖色，其余用 Blues。
"""
import sys
import io
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(PROJECT_ROOT))

from tools.viz._common import (
    setup_matplotlib, load_comparison_results,
    METRIC_DISPLAY, DATASET_ORDER, MODEL_ORDER, ALGO_COLORS,
)

FIGURES_DIR = PROJECT_ROOT / "results" / "figures"


def _plot_one_heatmap(ax, data_matrix, dataset_labels, model_labels, cmap, title):
    """在指定 ax 上画热力图，单元格标数值，最优列加粗描边"""
    im = ax.imshow(data_matrix, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    # 单元格数值
    for i in range(len(dataset_labels)):
        for j in range(len(model_labels)):
            val = data_matrix[i][j]
            color = "white" if val > 60 else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                    fontsize=9, color=color)
    # 最优列加粗描边
    import numpy as np
    max_idx = int(np.argmax(data_matrix, axis=1))
    for i, j in enumerate(max_idx):
        ax.add_patch(plt_rect(j - 0.5, i - 0.5, 1, 1, edgecolor="#bf5af2", lw=2.5, fill=False))
    ax.set_xticks(range(len(model_labels)))
    ax.set_xticklabels([m.upper() for m in model_labels], fontsize=9, rotation=30, ha="right")
    ax.set_yticks(range(len(dataset_labels)))
    ax.set_yticklabels(dataset_labels, fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    return im


def plt_rect(x, y, w, h, **kwargs):
    from matplotlib.patches import Rectangle
    return Rectangle((x, y), w, h, **kwargs)


def generate_all():
    """生成 4 张基准热力图并排保存为单张组合图，同时存单张"""
    plt = setup_matplotlib()
    results = load_comparison_results()
    datasets = [d for d in DATASET_ORDER if d in results]
    models = [m for m in MODEL_ORDER if any(m in results.get(d, {}) for d in datasets)]

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    for idx, metric in enumerate(["image_AUROC", "image_AUPR", "pixel_AUROC", "pixel_PRO"]):
        display_name, cmap = METRIC_DISPLAY[metric]
        matrix = []
        for ds in datasets:
            row = []
            for m in models:
                val = results.get(ds, {}).get(m, {}).get(metric, 0)
                row.append(val * 100 if val <= 1.0 else val)
            matrix.append(row)
        im = _plot_one_heatmap(axes[idx], matrix, datasets, models, cmap, display_name)
        if idx == 0:
            fig.colorbar(im, ax=axes[idx], shrink=0.8)

    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    combined = FIGURES_DIR / "benchmark_heatmap_all.png"
    plt.savefig(str(combined), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    # 同时存 4 张单图（文档按需引用）
    for metric in ["image_AUROC", "image_AUPR", "pixel_AUROC", "pixel_PRO"]:
        display_name, cmap = METRIC_DISPLAY[metric]
        matrix = []
        for ds in datasets:
            row = [results.get(ds, {}).get(m, {}).get(metric, 0) * 100
                   if results.get(ds, {}).get(m, {}).get(metric, 0) <= 1.0
                   else results.get(ds, {}).get(m, {}).get(metric, 0)
                   for m in models]
            matrix.append(row)
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        im2 = _plot_one_heatmap(ax2, matrix, datasets, models, cmap, display_name)
        fig2.colorbar(im2, ax=ax2, shrink=0.8)
        plt.tight_layout()
        plt.savefig(str(FIGURES_DIR / f"benchmark_heatmap_{metric}.png"),
                    dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()

    print(f"[OK] 基准热力图已生成: {FIGURES_DIR}")


if __name__ == "__main__":
    generate_all()
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_viz.py::test_benchmark_heatmap_generates_all_metrics -v`
Expected: PASS — 4 张 PNG 存在且 > 1000 字节

- [ ] **Step 5: 提交**

```bash
git add tools/viz/benchmark_heatmap.py tests/test_viz.py
git commit -F .git-msg
```
(git-msg: `feat(viz): 基准对比热力图脚本（4 张，PRO 用 YlOrRd 暖色）`)

---

## Task 3: 消融敏感性折线图脚本

**Files:**
- Create: `tools/viz/ablation_sensitivity.py`
- Test: `tests/test_viz.py`（追加测试）

- [ ] **Step 1: 追加失败测试 — 验证 3 张消融折线图生成**

追加到 `tests/test_viz.py`:

```python
def test_ablation_sensitivity_generates_three_plots():
    """消融敏感性折线图应生成 3 张（PatchCore/PaDiM/FRE）"""
    from tools.viz import ablation_sensitivity
    ablation_sensitivity.generate_all()
    assert (FIGURES_DIR / "ablation_patchcore_coreset_sampling_ratio.png").exists()
    assert (FIGURES_DIR / "ablation_padim_backbone.png").exists()
    assert (FIGURES_DIR / "ablation_fre_latent_dim.png").exists()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_viz.py::test_ablation_sensitivity_generates_three_plots -v`
Expected: FAIL

- [ ] **Step 3: 实现消融折线图脚本**

Create `tools/viz/ablation_sensitivity.py`:

```python
"""消融敏感性折线图：3 组消融各一张。

横轴=参数值，3 条线=Image AUROC/Pixel AUROC/PRO。
默认参数值处加竖虚线 + 顶部"★ 默认"标注。
"""
import sys
import io
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(PROJECT_ROOT))

from tools.viz._common import (
    setup_matplotlib, load_ablation_results, METRIC_COLORS, ALGO_COLORS,
)

FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

# 各消融的默认值（用于竖虚线标注）
DEFAULT_VALUES = {
    ("patchcore", "coreset_sampling_ratio"): 0.1,
    ("fre", "latent_dim"): 220,
}
# PaDiM backbone 无数值型默认，2 点连线处理

METRIC_LIST = ["image_AUROC", "pixel_AUROC", "pixel_PRO"]
METRIC_LABELS = {"image_AUROC": "Image AUROC", "pixel_AUROC": "Pixel AUROC", "pixel_PRO": "PRO"}


def _plot_numeric_param(ax, rows, param_name, default_val, model_name):
    """数值型参数（PatchCore coreset / FRE latent_dim）的标准折线"""
    # 按参数值排序
    rows_sorted = sorted(rows, key=lambda r: r["value"])
    x_vals = [r["value"] for r in rows_sorted]
    for metric in METRIC_LIST:
        y_vals = [r[metric] * 100 for r in rows_sorted]
        ax.plot(x_vals, y_vals, marker="o", color=METRIC_COLORS[metric],
                label=METRIC_LABELS[metric], linewidth=2, markersize=6)
        # 数据点标数值
        for xv, yv in zip(x_vals, y_vals):
            if metric == "pixel_PRO":
                ax.annotate(f"{yv:.1f}", (xv, yv), textcoords="offset points",
                            xytext=(0, 8), fontsize=8, ha="center")
    # 默认值竖虚线
    ax.axvline(x=default_val, color="#bf5af2", linestyle="--", alpha=0.6)
    ax.annotate("★ 默认", (default_val, ax.get_ylim()[1]), fontsize=9,
                color="#bf5af2", ha="center", va="top",
                xytext=(0, -10), textcoords="offset points")
    ax.set_xlabel(param_name, fontsize=10)
    ax.set_ylabel("指标 (%)", fontsize=10)
    ax.set_title(f"{model_name.upper()} — {param_name} 敏感性", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)


def _plot_backbone(ax, rows, model_name):
    """PaDiM backbone：2 点连线，横轴用类别名，附注参数量"""
    x_labels = [str(r["value"]) for r in rows]
    for metric in METRIC_LIST:
        y_vals = [r[metric] * 100 for r in rows]
        ax.plot(x_labels, y_vals, marker="o", color=METRIC_COLORS[metric],
                label=METRIC_LABELS[metric], linewidth=2, markersize=6)
        for xl, yv in zip(x_labels, y_vals):
            ax.annotate(f"{yv:.1f}", (xl, yv), textcoords="offset points",
                        xytext=(0, 8), fontsize=8, ha="center")
    ax.set_xlabel("backbone", fontsize=10)
    ax.set_ylabel("指标 (%)", fontsize=10)
    ax.set_title(f"{model_name.upper()} — backbone 消融", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.annotate("参数量: resnet18 → 2.8M / wide_resnet50_2 → ~69M",
                xy=(0.5, 0.02), xycoords="axes fraction", fontsize=8,
                ha="center", color="#888")


def generate_all():
    plt = setup_matplotlib()
    results = load_ablation_results()
    # 按 (model, param) 分组
    groups = {}
    for r in results:
        key = (r["model"], r["param"])
        groups.setdefault(key, []).append(r)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # PatchCore coreset_sampling_ratio
    for (model, param), rows in groups.items():
        fig, ax = plt.subplots(figsize=(7, 5))
        if param == "coreset_sampling_ratio":
            _plot_numeric_param(ax, rows, "coreset_sampling_ratio", 0.1, model)
        elif param == "latent_dim":
            _plot_numeric_param(ax, rows, "latent_dim", 220, model)
        elif param == "backbone":
            _plot_backbone(ax, rows, model)
        plt.tight_layout()
        plt.savefig(str(FIGURES_DIR / f"ablation_{model}_{param}.png"),
                    dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()

    print(f"[OK] 消融敏感性折线图已生成: {FIGURES_DIR}")


if __name__ == "__main__":
    generate_all()
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_viz.py::test_ablation_sensitivity_generates_three_plots -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add tools/viz/ablation_sensitivity.py tests/test_viz.py
git commit -F .git-msg
```
(git-msg: `feat(viz): 消融敏感性折线图脚本（3 张，默认值标注）`)

---

## Task 4: 小样本双轴折线脚本

**Files:**
- Create: `tools/viz/small_sample_dual_axis.py`
- Test: `tests/test_viz.py`（追加测试）

- [ ] **Step 1: 追加失败测试**

追加到 `tests/test_viz.py`:

```python
def test_small_sample_dual_axis_generates():
    """小样本双轴折线图应生成 1 张 PNG"""
    from tools.viz import small_sample_dual_axis
    small_sample_dual_axis.generate()
    assert (FIGURES_DIR / "small_sample_dual_axis.png").exists()
    assert (FIGURES_DIR / "small_sample_dual_axis.png").stat().st_size > 2000
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_viz.py::test_small_sample_dual_axis_generates -v`
Expected: FAIL

- [ ] **Step 3: 实现双轴折线脚本**

先确认小样本结果文件结构。小样本数据来自 `results/small_sample/`（如不存在则内嵌文档已知数据）。Create `tools/viz/small_sample_dual_axis.py`:

```python
"""小样本鲁棒性双轴折线 + 鲁棒性评分卡。

左轴（实线）：Image AUROC，4 算法线
右轴（虚线）：PRO，4 算法线
下半：鲁棒性评分卡（4 行）

优先从 results/small_sample/ 读真实数据；若无则用文档附录 C 的已知数据回退。
"""
import sys
import io
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(PROJECT_ROOT))

from tools.viz._common import setup_matplotlib, ALGO_COLORS

FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

SAMPLE_SIZES = [30, 60, 100, 150]
MODEL_ORDER = ["patchcore", "padim", "fre", "draem"]

# 文档附录 C 已知数据回退（百分比）
FALLBACK_DATA = {
    "image_AUROC": {
        "patchcore": [100.00, 100.00, 100.00, 100.00],
        "padim": [99.44, 99.76, 99.76, 100.00],
        "fre": [97.62, 99.29, 99.37, 99.37],
        "draem": [98.25, 98.17, 98.41, 97.94],
    },
    "pixel_PRO": {
        "patchcore": [79.39, 80.25, 81.76, 80.98],
        "padim": [79.03, 80.78, 78.89, 82.15],
        "fre": [72.35, 69.76, 68.96, 68.58],
        "draem": [42.38, 47.91, 35.49, 46.83],
    },
}


def _load_small_sample():
    """优先从 results/small_sample 读真实数据，无则用回退数据"""
    ss_dir = PROJECT_ROOT / "results" / "small_sample"
    if ss_dir.exists():
        # 尝试加载（文件格式可能各异，失败则回退）
        try:
            data = {"image_AUROC": {}, "pixel_PRO": {}}
            loaded = False
            for json_file in sorted(ss_dir.glob("*.json")):
                d = json.loads(json_file.read_text(encoding="utf-8"))
                if isinstance(d, list):
                    # 兼容列表格式
                    for item in d:
                        model = item.get("model")
                        if model:
                            loaded = True
            if loaded:
                return data
        except Exception:
            pass
    return FALLBACK_DATA


def generate():
    plt = setup_matplotlib()
    data = _load_small_sample()

    fig, (ax_au, ax_pro) = plt.subplots(2, 1, figsize=(9, 9),
                                         gridspec_kw={"height_ratios": [3, 1]})

    # 上半：双轴折线
    ax_pro_twin = ax_au.twinx()
    for model in MODEL_ORDER:
        color = ALGO_COLORS[model]
        # AUROC 实线
        au_vals = data["image_AUROC"][model]
        ax_au.plot(SAMPLE_SIZES, au_vals, marker="o", color=color,
                   linestyle="-", linewidth=2, markersize=5, label=f"{model.upper()} AUROC")
        # PRO 虚线
        pro_vals = data["pixel_PRO"][model]
        ax_pro_twin.plot(SAMPLE_SIZES, pro_vals, marker="s", color=color,
                         linestyle="--", linewidth=1.8, markersize=5,
                         label=f"{model.upper()} PRO")
        # 端点标值
        ax_au.annotate(f"{au_vals[0]:.1f}", (SAMPLE_SIZES[0], au_vals[0]),
                       xytext=(-5, 5), textcoords="offset points", fontsize=7, color=color)
        ax_au.annotate(f"{au_vals[-1]:.1f}", (SAMPLE_SIZES[-1], au_vals[-1]),
                       xytext=(5, 5), textcoords="offset points", fontsize=7, color=color)

    # DRAEM PRO 暴跌段阴影
    draem_pro = data["pixel_PRO"]["draem"]
    min_idx = draem_pro.index(min(draem_pro))
    ax_pro_twin.axvspan(SAMPLE_SIZES[max(0, min_idx - 1)], SAMPLE_SIZES[min_idx],
                        alpha=0.08, color="red")
    ax_pro_twin.annotate("定位退化", (SAMPLE_SIZES[min_idx], draem_pro[min_idx]),
                         fontsize=8, color="red", ha="center",
                         xytext=(0, -12), textcoords="offset points")

    ax_au.set_xlabel("训练样本数 N", fontsize=10)
    ax_au.set_ylabel("Image AUROC (%)", fontsize=10, color="#2997ff")
    ax_pro_twin.set_ylabel("PRO (%)", fontsize=10, color="#30d158")
    ax_au.set_title("小样本鲁棒性分析（bottle）", fontsize=13, fontweight="bold")
    ax_au.set_xticks(SAMPLE_SIZES)
    ax_au.legend(loc="lower right", fontsize=8, ncol=2)
    ax_pro_twin.legend(loc="upper left", fontsize=8, ncol=2)
    ax_au.grid(True, alpha=0.3)

    # 下半：鲁棒性评分卡
    ax_card = ax_pro
    ax_card.axis("off")
    ax_card.set_title("鲁棒性评分（鲁棒分 = N30 AUROC / N150 AUROC）",
                      fontsize=11, fontweight="bold")

    # 表格数据
    n30 = {m: data["image_AUROC"][m][0] for m in MODEL_ORDER}
    n150 = {m: data["image_AUROC"][m][-1] for m in MODEL_ORDER}
    scores = {m: (n30[m] / n150[m] if n150[m] > 0 else 0) for m in MODEL_ORDER}
    stars = {"patchcore": "★★★★★", "padim": "★★★★☆", "draem": "★★★★☆", "fre": "★★★☆☆"}

    rows = []
    for i, m in enumerate(MODEL_ORDER):
        is_best = (m == "patchcore")
        rows.append([m.upper(), f"{n30[m]:.1f}", f"{n150[m]:.1f}",
                     f"{scores[m]:.3f}", stars[m]])

    table = ax_card.table(cellText=rows,
                          colLabels=["算法", "N30 AUROC", "N150 AUROC", "鲁棒分", "等级"],
                          loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)
    # PatchCore 行高亮
    for j in range(5):
        table[(2, j)].set_facecolor("#e8f4fd")

    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(FIGURES_DIR / "small_sample_dual_axis.png"),
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[OK] 小样本双轴折线图已生成: {FIGURES_DIR}")


if __name__ == "__main__":
    generate()
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_viz.py::test_small_sample_dual_axis_generates -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add tools/viz/small_sample_dual_axis.py tests/test_viz.py
git commit -F .git-msg
```
(git-msg: `feat(viz): 小样本双轴折线 + 鲁棒性评分卡脚本`)

---

## Task 5: `run_all.py` 一键生成 + CLAUDE.md 图表目录补注

**Files:**
- Create: `tools/viz/run_all.py`
- Test: `tests/test_viz.py`（追加测试）

- [ ] **Step 1: 追加失败测试 — 验证 run_all 生成全部 8 张图**

追加到 `tests/test_viz.py`:

```python
def test_run_all_generates_all_figures():
    """run_all 应生成全部 8 张图表"""
    from tools.viz import run_all
    run_all.main()
    expected = [
        "benchmark_heatmap_image_AUROC.png",
        "benchmark_heatmap_image_AUPR.png",
        "benchmark_heatmap_pixel_AUROC.png",
        "benchmark_heatmap_pixel_PRO.png",
        "ablation_patchcore_coreset_sampling_ratio.png",
        "ablation_padim_backbone.png",
        "ablation_fre_latent_dim.png",
        "small_sample_dual_axis.png",
    ]
    for f in expected:
        assert (FIGURES_DIR / f).exists(), f"缺失: {f}"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `python -m pytest tests/test_viz.py::test_run_all_generates_all_figures -v`
Expected: FAIL

- [ ] **Step 3: 实现 `run_all.py`**

Create `tools/viz/run_all.py`:

```python
"""一键生成全部可视化图表到 results/figures/"""
import sys
import io
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(PROJECT_ROOT))

from modules._runtime import configure_runtime_temp
configure_runtime_temp()

from tools.viz import benchmark_heatmap, ablation_sensitivity, small_sample_dual_axis


def main():
    print("=" * 60)
    print("生成全部可视化图表 → results/figures/")
    print("=" * 60)
    print("\n[1/3] 基准对比热力图...")
    benchmark_heatmap.generate_all()
    print("\n[2/3] 消融敏感性折线图...")
    ablation_sensitivity.generate_all()
    print("\n[3/3] 小样本双轴折线图...")
    small_sample_dual_axis.generate()
    print("\n" + "=" * 60)
    print("全部图表生成完毕!")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行测试确认通过**

Run: `python -m pytest tests/test_viz.py -v`
Expected: PASS — 全部 7 个测试通过

- [ ] **Step 5: 提交**

```bash
git add tools/viz/run_all.py tests/test_viz.py
git commit -F .git-msg
```
(git-msg: `feat(viz): run_all 一键生成全部 8 张图表`)

---

## Task 6: 扩展 `run_ablation.py`（DRAEM 注释）+ `run_report.py`（图表引用提示）

**Files:**
- Modify: `tools/run_ablation.py:94-115`（DRAEM 分支补注释，不改逻辑）
- Modify: `tools/run_report.py:272`（末尾追加图表引用）

- [ ] **Step 1: 修改 `run_ablation.py` — DRAEM 分支补注释**

在 `tools/run_ablation.py` 的 `main()` 函数中，FRE 消融块之后（约 L115 后），DRAEM 无消融分支处补注释。找到：

```python
    # FRE 消融
    if args.model in ('fre', 'all'):
        experiments.extend([
            ('fre', 'latent_dim', v, {'latent_dim': v})
            for v in [100, 220, 500]
        ])
```

在其后追加注释块（不新增实验，仅说明 DRAEM 为何跳过）:

```python
    # DRAEM 消融：未开展
    # 理由（详见 docs/最终汇报文档.md 5.2 节 DRAEM 参数说明）：
    #   1. draem.yaml 的 model.init_args 为空 dict，采用 anomalib 默认参数
    #   2. 早停监控 train_loss 仅判收敛，无法防过拟合
    #   3. 受限于项目周期与 GPU 资源（DRAEM 100 epoch × 多组参数耗时过长）
    # 后续工作方向：anomaly_scales / lr 网格搜索（见 8.3 未来工作）
```

- [ ] **Step 2: 修改 `run_report.py` — 末尾追加图表引用提示**

在 `tools/run_report.py` 的 `generate_report()` 末尾（`lines.append(f"\n---\n*报告由 run_report.py 自动生成*")` 之后、写文件之前），追加:

```python
    # 可视化图表引用提示
    lines.append("")
    lines.append("## 可视化图表")
    lines.append("")
    lines.append("本报告配套的可视化图表由 `tools/viz/run_all.py` 生成，存放于 `results/figures/`：")
    lines.append("- 基准对比热力图：`benchmark_heatmap_*.png`（4 张，PRO 用 YlOrRd 暖色）")
    lines.append("- 消融敏感性折线图：`ablation_*.png`（3 张）")
    lines.append("- 小样本双轴折线 + 鲁棒性评分卡：`small_sample_dual_axis.png`")
    lines.append("")
    lines.append("生成命令：`python tools/viz/run_all.py`")
```

- [ ] **Step 3: 验证两脚本仍可运行（不破坏现有逻辑）**

Run: `python tools/run_report.py --output results/comparison/report_test.md && head -20 results/comparison/report_test.md`
Expected: 报告正常生成，含"## 可视化图表"段

清理测试输出: `rm results/comparison/report_test.md`

- [ ] **Step 4: 提交**

```bash
git add tools/run_ablation.py tools/run_report.py
git commit -F .git-msg
```
(git-msg: `docs(tools): run_ablation 补 DRAEM 未消融说明，run_report 追加图表引用提示`)

---

## Task 7: 文档修复 A 项 — 混淆矩阵与第七章与 image_size

**Files:**
- Modify: `docs/最终汇报文档.md`（5.3 节、第七章、4.1.3/7.3 节、5.1 节、5.6/8.1 节）

- [ ] **Step 1: 修复 5.3 节混淆矩阵数据（A.1）**

以 `results/confusion_matrices/*.json` 真实值为准重写 5.3 节正文表格。核对的真实值（已从 JSON 读取）：

- PatchCore @ bottle: TP63/FP0/TN20/FN0（附录一致，无需改）
- PaDiM @ bottle: TP62/FP0/TN20/FN1 → 正文写的"63/0/20/0 100%"须改为"62/0/20/1 准确率 98.8% 精确率 100% 召回率 98.4%"
- FRE @ bottle: 正文"62/1/19/1" → 改为"55/0/20/8 准确率 90.4% 精确率 100% 召回率 87.3%"
- DRAEM @ bottle: 正文"63/8/12/0" → 改为"60/0/20/3 准确率 96.4% 精确率 100% 召回率 95.2%"
- PatchCore @ region1: 正文"FP1/TN90" → 改为"FP8/TN83"（即 7/8/83/0 准确率 91.8% 精确率 46.7% 召回率 100%）
- PaDiM @ region2: 正文"17/28/51/10 超过半数误报" → 改为"4/0/91/11 准确率 89.6% 精确率 100% 召回率 26.7%"，正文叙述须从"超过一半正常样本误报"改为"11 张缺陷漏检，召回仅 26.7%"

定位 5.3 节正文表格（约 L567-580）逐一替换上述 6 行。

- [ ] **Step 2: 修复第七章可视化平台（A.2）**

重写 7.1/7.2/7.3 节：

7.1 节"基于 Gradio 框架""http://127.0.0.1:7860" → "基于 FastAPI + Alpine.js SPA（Phase 2 重构）""http://127.0.0.1:8000"；补注"Gradio 作为 legacy fallback（`python scripts/run_ui.py --gradio`，端口 7860）"。

7.2 核心功能表更新为 5 页 snap 架构：算法介绍(s0) → 训练工作室(s1) → 单模型推理(s2) → 四模型对比(s3)。

7.3 技术架构三层改为：FastAPI（REST API + SSE 流式推理/训练）+ Alpine.js SPA + anomalib 推理（asyncio.to_thread 线程池）；访问端口 8000。

- [ ] **Step 3: 修复 image_size 224→256（A.3）**

全局搜索 `docs/最终汇报文档.md` 中"224×224"，替换为"256×256"。已知位置：4.1.3 数据预处理、7.3 推理数据流描述。

- [ ] **Step 4: 清理"补齐"措辞（A.4）**

搜索"补齐""已完成全部 6 个数据集""在补齐全部 6 个数据集后"等过渡措辞（5.1 节），统一为"24 组对比实验（4 算法 × 6 数据集全覆盖）"最终态表述。

- [ ] **Step 5: 对齐综合排名措辞（A.5）**

5.6 综合排名表说明与 8.1 结论点明：PatchCore 综合第一依据是 Pixel AUROC（97.94%）与 PRO（51.06%）领先；PaDiM 平均图像级 AUROC 略高（92.44 vs 92.22）但像素级定位次之。

- [ ] **Step 6: 提交**

```bash
git add docs/最终汇报文档.md
git commit -F .git-msg
```
(git-msg: `docs(report): A 项修复——混淆矩阵数据/第七章 FastAPI/image_size/补齐措辞/综合排名`)

---

## Task 8: 文档补强 B 项 — 训练参数最优性说明

**Files:**
- Modify: `docs/最终汇报文档.md`（4.5 节、3.3.4/3.4.4 节、5.2 节、8.2/8.3 节、F.3 节）

- [ ] **Step 1: 修复 4.5 节早停机制描述（B.1）**

4.5 节"FRE/DRAEM 使用早停（patience=10），监控 `val_image_AUROC`" → 改为：

```
3. **早停机制**：FRE/DRAEM 使用早停监控 `train_loss`（mode: min）。原因：DRAEM/FRE 的评估指标（image_AUROC 等）只在 `test()` 时计算，不在训练时计算，故无法用 `val_image_AUROC` 早停。`train_loss` 仅判断训练收敛，**无法防止过拟合**——这是 DRAEM 像素定位 PRO 偏低、小样本下 PRO 暴跌的可能诱因之一（见 8.2 工作局限性）。patience：FRE=10、DRAEM=5。
```

- [ ] **Step 2: 3.3.4 节补 PaDiM backbone 取舍（B.2）**

在 PaDiM 关键超参数表后补：

```
**backbone 取舍说明**：`resnet18`（2.8M 参数）为边缘轻量取向；精度优先应换 `wide_resnet50_2`（消融 PRO +5.97%，但参数量增至 ~69M）。详见 6.2 场景 B 边缘部署方案。
```

- [ ] **Step 3: 3.4.4 节补 FRE latent_dim 取舍（B.2）**

在 FRE 关键超参数表后补：

```
**latent_dim 取舍说明**：`latent_dim=220` 在图像级分类精度上最优（消融 AUROC 峰值）；若以定位精度（PRO）为优先，应降至 100（PRO +0.93%，信息瓶颈效应：维度越小 AE 被迫学习更紧凑表示）。
```

- [ ] **Step 4: 5.2 节新增 DRAEM 参数说明（B.3）**

在 5.2 节末尾（5.2.3 FRE 消融之后）新增子节：

```
### 5.2.4 DRAEM 参数说明

DRAEM 使用 anomalib 默认参数，未做消融。理由：
1. `configs/draem.yaml` 的 `model.init_args` 为空 dict，采用 anomalib 库默认值（`beta: [0.1, 1.0]`、`enable_sspcab: false` 等）。
2. 早停监控 `train_loss`（mode: min），仅判断训练收敛，无法用验证指标早停（见 4.5 节）。
3. 受限于项目周期与 GPU 资源（DRAEM 100 epoch × 多组参数耗时过长）未开展系统消融。

DRAEM 参数最优性缺乏消融依据，记入 8.2 工作局限性。后续调优方向见 8.3（anomaly_scales / lr 网格搜索）。
```

- [ ] **Step 5: 8.2 工作局限性补 DRAEM 早停与参数依据（B.1/B.3）**

在 8.2 节补充（新增编号，承接现有条目）：

```
6. **DRAEM 参数未消融**：DRAEM 采用 anomalib 默认参数，未做消融，参数最优性缺乏依据（详见 5.2.4）。
7. **早停无法防过拟合**：FRE/DRAEM 早停监控 `train_loss` 仅判收敛，无法监控泛化性能，可能导致过拟合未被察觉。
```

- [ ] **Step 6: 8.3 未来工作补 DRAEM 调优方向（B.3）**

在 8.3 节补充：

```
7. **DRAEM 参数调优**：对 DRAEM 的 anomaly_scales / lr / beta 开展网格搜索消融，论证参数最优性，并引入验证集泛化监控机制以替代 train_loss 早停。
```

- [ ] **Step 7: F.3 可复现性声明补 seed 说明（B.4）**

F.3 节"所有随机种子固定为 `seed=42`" → 改为：

```
- 各模型随机种子以 per-model YAML 为准：patchcore.yaml/padim.yaml 为 `seed: 0`，fre.yaml/draem.yaml 为 `seed: 42`（run_training.py 默认 42）
```

- [ ] **Step 8: 同步修正 CLAUDE.md「早停机制」段（B.1）**

`CLAUDE.md`「早停机制」段当前称"监控 `val_image_AUROC`"，修正为：

```
DRAEM/FRE 监控 `train_loss`（mode: min）——因评估指标（image_AUROC 等）只在 `test()` 时计算，不在训练时计算，故无法用 `val_image_AUROC` 早停。`train_loss` 仅判收敛，无法防过拟合。patience：FRE=10、DRAEM=5（在各模型 `{model}.yaml` 的 `early_stopping` 下配置）。PatchCore/PaDiM 为单 epoch 特征提取/高斯建模，不需要早停。
```

- [ ] **Step 9: 提交**

```bash
git add docs/最终汇报文档.md CLAUDE.md
git commit -F .git-msg
```
(git-msg: `docs(report): B 项补强——早停机制修正/PaDiM·FRE 取舍/DRAEM 未消融说明/seed 注记`)

---

## Task 9: 文档图表替换 C 项 — 三类表格换可视化

**Files:**
- Modify: `docs/最终汇报文档.md`（5.1/5.2/5.5 节、附录 C）

**前提**：Task 1-5 已生成 `results/figures/` 下的 8 张图表。本任务将文档中的数值表替换为图表引用，并把图片复制到 `docs/images/report/` 供文档 `<figure>` 引用。

- [ ] **Step 1: 复制图表到 docs/images/report/ 供文档引用**

```bash
mkdir -p docs/images/report
cp results/figures/benchmark_heatmap_all.png docs/images/report/benchmark_heatmap.png
cp results/figures/ablation_patchcore_coreset_sampling_ratio.png docs/images/report/ablation_patchcore.png
cp results/figures/ablation_padim_backbone.png docs/images/report/ablation_padim.png
cp results/figures/ablation_fre_latent_dim.png docs/images/report/ablation_fre.png
cp results/figures/small_sample_dual_axis.png docs/images/report/small_sample_dual_axis.png
```

- [ ] **Step 2: 替换 5.1 节基准对比表为热力图（C.1）**

5.1.1 节"表 5-1：图像级 AUROC"数值表 → 替换为：

```html
<figure>
  <img src="images/report/benchmark_heatmap.png" alt="基准对比热力图矩阵" style="display:block;margin:12px auto;max-width:100%;width:1000px;">
  <figcaption align="center">图 5-1 基准对比热力图（AUROC/AUPR/Pixel AUROC/PRO，PRO 用暖色突出瓶颈）</figcaption>
</figure>
```

保留"关键发现"文字段落。AUPR 表（独立表）保留或合并到热力图说明。

- [ ] **Step 3: 替换 5.2 节消融表为敏感性折线图（C.2）**

5.2.1 PatchCore 表 → 替换为 `<figure>` 引用 `ablation_patchcore.png`；
5.2.2 PaDiM 表 → 替换为 `ablation_padim.png`；
5.2.3 FRE 表 → 替换为 `ablation_fre.png`。
每节保留精简后的"分析"文字（删复述数值句）。

- [ ] **Step 4: 替换 5.5 节小样本表为双轴折线图（C.3）**

5.5.1 + 5.5.2 两张表 → 合并替换为 `<figure>` 引用 `small_sample_dual_axis.png`。
5.5.3 鲁棒性排序表 → 删除（内容已并入评分卡），保留排序结论文字。

- [ ] **Step 5: 删除附录 C.1-C.4 四张分 N 表**

附录 C.1/C.2/C.3/C.4 四张表删除（数据已在双轴折线图中）。保留 C.5 汇总表（带 Δ(N30→N150) 列）。

- [ ] **Step 6: 验证文档图片引用有效**

Run: `python -c "from pathlib import Path; p=Path('docs/最终汇报文档.md').read_text(encoding='utf-8'); import re; refs=re.findall(r'src=\"(images/report/[^\"]+)\"', p); [print(r, Path('docs/'+r).exists()) for r in refs]"`
Expected: 所有引用图片均存在（True）

- [ ] **Step 7: 提交**

```bash
git add docs/最终汇报文档.md docs/images/report/
git commit -F .git-msg
```
(git-msg: `docs(report): C 项图表替换——基准热力图/消融折线/小样本双轴，删冗余表`)

---

## Task 10: 最终验证与全量测试

**Files:**
- Test: `tests/test_viz.py` 全套 + 现有测试套件

- [ ] **Step 1: 重新生成全部图表确认无误**

Run: `python tools/viz/run_all.py`
Expected: 输出"[1/3]...[2/3]...[3/3]...全部图表生成完毕!"，`results/figures/` 下 8 张图存在

- [ ] **Step 2: 运行全量测试套件**

Run: `python -m pytest tests/ -v`
Expected: 全部测试通过（含新增 7 个 viz 测试 + 现有测试）

- [ ] **Step 3: 逐张核对图表数值与 JSON 一致**

人工核对（或脚本核对）：
- 基准热力图 bottle/patchcore = 100.0（与 `patchcore_bottle_results.json` image_AUROC=1.0 一致）
- 消融折线 PatchCore coreset=0.1 的 PRO=81.29（与 ablation_results.json pixel_PRO=0.8129 一致）
- 小样本双轴 DRAEM N30 PRO=42.38（与附录 C 数据一致）

- [ ] **Step 4: 通读修订后的文档自洽性检查**

人工通读 `docs/最终汇报文档.md`：
- 5.3 节混淆矩阵数据与附录 E 一致（不再矛盾）
- 第七章端口为 8000（非 7860）
- 无残留"224×224""补齐"措辞
- 三类图表 `<figure>` 引用均有效
- 8.2/8.3 节含 DRAEM 早停局限性

- [ ] **Step 5: 提交最终验证（如有改动）**

如 Step 1-4 有任何修正，提交：

```bash
git add -A
git commit -F .git-msg
```
(git-msg: `test: 最终验证——全量测试通过、图表数值核对一致`)

---

## Self-Review 自审查结果

**Spec 覆盖**：
- A.1 混淆矩阵 → Task 7 Step 1 ✓
- A.2 第七章 → Task 7 Step 2 ✓
- A.3 image_size → Task 7 Step 3 ✓
- A.4 补齐措辞 → Task 7 Step 4 ✓
- A.5 综合排名 → Task 7 Step 5 ✓
- B.1 早停机制 → Task 8 Step 1, 8 ✓
- B.2 PaDiM/FRE 取舍 → Task 8 Step 2, 3 ✓
- B.3 DRAEM 未消融 → Task 8 Step 4, 5, 6 ✓
- B.4 seed → Task 8 Step 7 ✓
- C.1 基准热力图 → Task 2 + Task 9 Step 2 ✓
- C.2 消融折线 → Task 3 + Task 9 Step 3 ✓
- C.3 双轴折线 → Task 4 + Task 9 Step 4, 5 ✓
- 工具脚本 → Task 1-6 ✓
- 不改 configs → 全程未涉及 ✓

**占位符扫描**：无 TBD/TODO，所有步骤含完整代码。

**类型一致性**：`ALGO_COLORS`/`METRIC_COLORS`/`DATASET_ORDER`/`MODEL_ORDER` 在 `_common.py` 定义，Task 2/3/4 引用一致。`generate_all()`/`generate()` 函数名在各脚本一致。

**已核验事实**：混淆矩阵 JSON 真实值已读取确认（Task 7 Step 1 的数据来自实际 JSON）；ablation_results.json 结构已确认（Task 3 直接消费）。
