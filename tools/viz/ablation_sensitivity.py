"""消融敏感性折线图：3 组消融各一张。

横轴=参数值，3 条线=Image AUROC/Pixel AUROC/PRO。
默认参数值处加竖虚线 + 顶部"★ 默认"标注。
"""
import sys
import io
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if __name__ == "__main__" and sys.platform == "win32":
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
        # 标注：PRO 全标，AUROC 只标首尾端点
        for idx, (xv, yv) in enumerate(zip(x_vals, y_vals)):
            if metric == "pixel_PRO":
                ax.annotate(f"{yv:.1f}", (xv, yv), textcoords="offset points",
                            xytext=(0, 8), fontsize=8, ha="center")
            elif metric == "image_AUROC" and idx in (0, len(x_vals) - 1):
                ax.annotate(f"{yv:.1f}", (xv, yv), textcoords="offset points",
                            xytext=(0, -12), fontsize=8, ha="center",
                            color=METRIC_COLORS[metric])
    # 默认值竖虚线
    ax.axvline(x=default_val, color=ALGO_COLORS["draem"], linestyle="--", alpha=0.6)
    ax.annotate("★ 默认", (default_val, ax.get_ylim()[1]), fontsize=9,
                color=ALGO_COLORS["draem"], ha="center", va="top",
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
        for idx, (xl, yv) in enumerate(zip(x_labels, y_vals)):
            if metric == "pixel_PRO":
                ax.annotate(f"{yv:.1f}", (xl, yv), textcoords="offset points",
                            xytext=(0, 8), fontsize=8, ha="center")
            elif metric == "image_AUROC":
                ax.annotate(f"{yv:.1f}", (xl, yv), textcoords="offset points",
                            xytext=(0, -12), fontsize=8, ha="center",
                            color=METRIC_COLORS[metric])
            # pixel_AUROC: 不标注（与 _plot_numeric_param 一致，保持简洁）
    ax.set_xlabel("backbone", fontsize=10)
    ax.set_ylabel("指标 (%)", fontsize=10)
    ax.set_title(f"{model_name.upper()} — backbone 消融", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.annotate("参数量: resnet18 → 2.8M / wide_resnet50_2 → ~69M",
                xy=(0.5, 0.02), xycoords="axes fraction", fontsize=8,
                ha="center", color="#888")


def generate_all(root: Path | None = None):
    plt = setup_matplotlib()
    results = load_ablation_results(root)
    # 按 (model, param) 分组
    groups = {}
    for r in results:
        key = (r["model"], r["param"])
        groups.setdefault(key, []).append(r)

    figures_dir = (root or PROJECT_ROOT) / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # PatchCore coreset_sampling_ratio
    for (model, param), rows in groups.items():
        fig, ax = plt.subplots(figsize=(7, 5))
        default_val = DEFAULT_VALUES.get((model, param))
        if param == "coreset_sampling_ratio":
            _plot_numeric_param(ax, rows, "coreset_sampling_ratio", default_val, model)
        elif param == "latent_dim":
            _plot_numeric_param(ax, rows, "latent_dim", default_val, model)
        elif param == "backbone":
            _plot_backbone(ax, rows, model)
        plt.tight_layout()
        plt.savefig(str(figures_dir / f"ablation_{model}_{param}.png"),
                    dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()

    print(f"[OK] 消融敏感性折线图已生成: {figures_dir}")


if __name__ == "__main__":
    from modules._runtime import configure_runtime_temp
    configure_runtime_temp()
    generate_all()
