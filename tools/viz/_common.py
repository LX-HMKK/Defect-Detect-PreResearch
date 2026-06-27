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
