"""可视化脚本套件回归测试（断言式，沿用 test_ui_static.py 风格）"""
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
    from tools.viz._common import load_comparison_results, DATASET_ORDER, MODEL_ORDER
    results = load_comparison_results()
    # 锁定 6 数据集 × 4 模型契约
    assert len(results) == 6, f"应为 6 个数据集，实际 {len(results)}"
    for ds in DATASET_ORDER:
        assert ds in results, f"缺失数据集: {ds}"
        assert len(results[ds]) == 4, f"{ds} 应有 4 个模型，实际 {len(results[ds])}"
        for m in MODEL_ORDER:
            assert m in results[ds], f"{ds} 缺失模型: {m}"
    # bottle/patchcore 精确值校验
    assert results["bottle"]["patchcore"]["image_AUROC"] == 1.0


def test_load_ablation_results():
    """load_ablation_results 应返回 ablation_results.json 的列表"""
    from tools.viz._common import load_ablation_results
    results = load_ablation_results()
    assert isinstance(results, list)
    assert len(results) == 9  # patchcore×4 + padim×2 + fre×3
    assert results[0]["model"] == "patchcore"


def test_benchmark_heatmap_generates_all_metrics():
    """基准热力图应生成 4 张 PNG（AUROC/AUPR/PixelAUROC/PRO）"""
    from tools.viz import benchmark_heatmap
    benchmark_heatmap.generate_all()
    for metric in ["image_AUROC", "image_AUPR", "pixel_AUROC", "pixel_PRO"]:
        assert (FIGURES_DIR / f"benchmark_heatmap_{metric}.png").exists()
        assert (FIGURES_DIR / f"benchmark_heatmap_{metric}.png").stat().st_size > 1000
    # 组合图（1×4 并排，文档 Task 9 复制到 docs/images/report/）
    assert (FIGURES_DIR / "benchmark_heatmap_all.png").exists()
    assert (FIGURES_DIR / "benchmark_heatmap_all.png").stat().st_size > 1000


def test_ablation_sensitivity_generates_three_plots():
    """消融敏感性折线图应生成 3 张（PatchCore/PaDiM/FRE）"""
    from tools.viz import ablation_sensitivity
    ablation_sensitivity.generate_all()
    assert (FIGURES_DIR / "ablation_patchcore_coreset_sampling_ratio.png").exists()
    assert (FIGURES_DIR / "ablation_padim_backbone.png").exists()
    assert (FIGURES_DIR / "ablation_fre_latent_dim.png").exists()
