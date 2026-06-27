"""基准对比热力图：4 算法 × 6 数据集 × 4 指标。

生成 4 张 PNG 到 results/figures/benchmark_heatmap_{metric}.png。
PRO 单独用 YlOrRd 暖色，其余用 Blues。
"""
import sys
import io
from pathlib import Path

import numpy as np
from matplotlib.patches import Rectangle

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PROJECT_ROOT))

from tools.viz._common import (
    setup_matplotlib, load_comparison_results,
    METRIC_DISPLAY, DATASET_ORDER, MODEL_ORDER, ALGO_COLORS,
)

FIGURES_DIR = PROJECT_ROOT / "results" / "figures"


def _build_matrix(results, datasets, models, metric):
    """从结果构造 [datasets × models] 百分比矩阵，值 0-1 转 0-100"""
    matrix = []
    for ds in datasets:
        row = []
        for m in models:
            v = results.get(ds, {}).get(m, {}).get(metric, 0)
            row.append(v * 100 if v <= 1.0 else v)
        matrix.append(row)
    return matrix


def _plot_one_heatmap(ax, data_matrix, dataset_labels, model_labels, cmap, title):
    """在指定 ax 上画热力图，单元格标数值，最优列加粗描边"""
    im = ax.imshow(data_matrix, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    # 单元格数值
    for i in range(len(dataset_labels)):
        for j in range(len(model_labels)):
            val = data_matrix[i][j]
            color = "white" if val > 60 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=9, color=color)
    # 最优列加粗描边（每行各自的最大值单元格）
    max_idx = np.argmax(data_matrix, axis=1)
    for i, j in enumerate(max_idx):
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                               edgecolor=ALGO_COLORS["draem"], lw=2.5, fill=False))
    ax.set_xticks(range(len(model_labels)))
    ax.set_xticklabels([m.upper() for m in model_labels], fontsize=9, rotation=30, ha="right")
    ax.set_yticks(range(len(dataset_labels)))
    ax.set_yticklabels(dataset_labels, fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    return im


def generate_all():
    """生成 4 张基准热力图并排保存为单张组合图，同时存单张"""
    plt = setup_matplotlib()
    results = load_comparison_results()
    datasets = [d for d in DATASET_ORDER if d in results]
    models = [m for m in MODEL_ORDER if any(m in results.get(d, {}) for d in datasets)]

    metrics = ["image_AUROC", "image_AUPR", "pixel_AUROC", "pixel_PRO"]

    # 组合图（1×4 并排）
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    for idx, metric in enumerate(metrics):
        display_name, cmap = METRIC_DISPLAY[metric]
        matrix = _build_matrix(results, datasets, models, metric)
        im = _plot_one_heatmap(axes[idx], matrix, datasets, models, cmap, display_name)
        # 每个子图都加 colorbar：PRO 用 YlOrRd 暖色需自己的色标，Blues 面板共享 vmin/vmax=0/100
        fig.colorbar(im, ax=axes[idx], shrink=0.8)

    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    combined = FIGURES_DIR / "benchmark_heatmap_all.png"
    plt.savefig(str(combined), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    # 同时存 4 张单图（文档按需引用）
    for metric in metrics:
        display_name, cmap = METRIC_DISPLAY[metric]
        matrix = _build_matrix(results, datasets, models, metric)
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        im2 = _plot_one_heatmap(ax2, matrix, datasets, models, cmap, display_name)
        fig2.colorbar(im2, ax=ax2, shrink=0.8)
        plt.tight_layout()
        plt.savefig(str(FIGURES_DIR / f"benchmark_heatmap_{metric}.png"),
                    dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()

    print(f"[OK] 基准热力图已生成: {FIGURES_DIR}")


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    from modules._runtime import configure_runtime_temp
    configure_runtime_temp()
    generate_all()
