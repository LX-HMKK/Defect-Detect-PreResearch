"""小样本鲁棒性双轴折线 + 鲁棒性评分卡。

左轴（实线）：Image AUROC，4 算法线
右轴（虚线）：PRO，4 算法线
下半：鲁棒性评分卡（4 行，PatchCore 行高亮）

优先从 results/small_sample/small_sample_summary.json 读真实数据；若无则用文档附录 C 的已知数据回退。
"""
import sys
import io
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if __name__ == "__main__" and sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(PROJECT_ROOT))

from tools.viz._common import setup_matplotlib, ALGO_COLORS, METRIC_COLORS

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
    """优先从 results/small_sample/small_sample_summary.json 读真实数据，无则用回退数据。

    真实文件结构：{category, sample_sizes: ["N30",...], results: {N30: {model: {metrics}}}}
    """
    summary_path = PROJECT_ROOT / "results" / "small_sample" / "small_sample_summary.json"
    if summary_path.exists():
        try:
            raw = json.loads(summary_path.read_text(encoding="utf-8"))
            results = raw.get("results", {})
            size_keys = [f"N{n}" for n in SAMPLE_SIZES]
            data = {"image_AUROC": {}, "pixel_PRO": {}}
            for model in MODEL_ORDER:
                au_list, pro_list = [], []
                for sk in size_keys:
                    entry = results.get(sk, {}).get(model, {})
                    au_list.append(float(entry.get("image_AUROC", 0)))
                    pro_list.append(float(entry.get("pixel_PRO", 0)))
                # 完整性校验：4 个尺寸均存在且非零
                if len(au_list) == 4 and len(pro_list) == 4 and all(au_list) and all(pro_list):
                    data["image_AUROC"][model] = au_list
                    data["pixel_PRO"][model] = pro_list
                else:
                    raise ValueError(f"{model} 数据不完整")
            print(f"[viz] 已加载真实小样本数据: {summary_path.name}")
            return data
        except Exception as e:
            print(f"[viz] 解析小样本数据失败，回退到附录 C 已知数据: {e}",
                  file=sys.stderr)
    return FALLBACK_DATA


def generate():
    plt = setup_matplotlib()
    data = _load_small_sample()

    # 上半：双轴折线；下半：鲁棒性评分卡（高度比 3:1）
    fig, (ax_au, ax_card) = plt.subplots(
        2, 1, figsize=(9, 9), gridspec_kw={"height_ratios": [3, 1]}
    )
    ax_pro = ax_au.twinx()  # PRO 共享上半坐标区（右轴）

    au_color = METRIC_COLORS["image_AUROC"]
    pro_color = METRIC_COLORS["pixel_PRO"]

    for model in MODEL_ORDER:
        color = ALGO_COLORS[model]
        au_vals = data["image_AUROC"][model]
        pro_vals = data["pixel_PRO"][model]
        # AUROC 实线（左轴）
        ax_au.plot(SAMPLE_SIZES, au_vals, marker="o", color=color,
                   linestyle="-", linewidth=2, markersize=5,
                   label=f"{model.upper()} AUROC")
        # PRO 虚线（右轴）
        ax_pro.plot(SAMPLE_SIZES, pro_vals, marker="s", color=color,
                    linestyle="--", linewidth=1.8, markersize=5,
                    label=f"{model.upper()} PRO")
        # AUROC 端点标值
        ax_au.annotate(f"{au_vals[0]:.1f}", (SAMPLE_SIZES[0], au_vals[0]),
                       xytext=(-5, 5), textcoords="offset points",
                       fontsize=7, color=color)
        ax_au.annotate(f"{au_vals[-1]:.1f}", (SAMPLE_SIZES[-1], au_vals[-1]),
                       xytext=(5, 5), textcoords="offset points",
                       fontsize=7, color=color)

    # DRAEM PRO 暴跌段阴影（最低点前后区间）
    draem_pro = data["pixel_PRO"]["draem"]
    min_idx = draem_pro.index(min(draem_pro))
    ax_pro.axvspan(SAMPLE_SIZES[max(0, min_idx - 1)], SAMPLE_SIZES[min_idx],
                   alpha=0.08, color="red")
    ax_pro.annotate("定位退化", (SAMPLE_SIZES[min_idx], draem_pro[min_idx]),
                    fontsize=8, color="red", ha="center",
                    xytext=(0, -12), textcoords="offset points")

    ax_au.set_xlabel("训练样本数 N", fontsize=10)
    ax_au.set_ylabel("Image AUROC (%)", fontsize=10, color=au_color)
    ax_pro.set_ylabel("PRO (%)", fontsize=10, color=pro_color)
    ax_au.set_title("小样本鲁棒性分析（bottle）", fontsize=13, fontweight="bold")
    ax_au.set_xticks(SAMPLE_SIZES)
    ax_au.legend(loc="lower right", fontsize=8, ncol=2)
    ax_pro.legend(loc="upper left", fontsize=8, ncol=2)
    ax_au.grid(True, alpha=0.3)

    # 下半：鲁棒性评分卡
    ax_card.axis("off")
    ax_card.set_title("鲁棒性评分（鲁棒分 = N30 AUROC / N150 AUROC）",
                      fontsize=11, fontweight="bold")

    n30 = {m: data["image_AUROC"][m][0] for m in MODEL_ORDER}
    n150 = {m: data["image_AUROC"][m][-1] for m in MODEL_ORDER}
    scores = {m: (n30[m] / n150[m] if n150[m] > 0 else 0) for m in MODEL_ORDER}
    stars = {"patchcore": "★★★★★", "padim": "★★★★☆",
             "draem": "★★★★☆", "fre": "★★★☆☆"}

    rows = []
    for m in MODEL_ORDER:
        rows.append([m.upper(), f"{n30[m]:.1f}", f"{n150[m]:.1f}",
                     f"{scores[m]:.3f}", stars[m]])

    table = ax_card.table(
        cellText=rows,
        colLabels=["算法", "N30 AUROC", "N150 AUROC", "鲁棒分", "等级"],
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)
    # PatchCore 行高亮：表首行(row 0)为表头，PatchCore 是第一条数据行 = row 1
    for j in range(5):
        table[(1, j)].set_facecolor("#e8f4fd")

    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(FIGURES_DIR / "small_sample_dual_axis.png"),
                dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[OK] 小样本双轴折线图已生成: {FIGURES_DIR}")


if __name__ == "__main__":
    from modules._runtime import configure_runtime_temp
    configure_runtime_temp()
    generate()
