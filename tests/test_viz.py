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
