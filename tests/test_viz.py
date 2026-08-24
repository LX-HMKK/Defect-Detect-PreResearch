"""可视化脚本套件回归测试（断言式）"""
import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIZ_DIR = PROJECT_ROOT / "tools" / "viz"

DATASETS = ["bottle", "carpet", "region1", "region2", "region3", "region5"]
MODELS = ["patchcore", "padim", "fre", "draem"]


@pytest.fixture
def viz_root(tmp_path: Path) -> Path:
    """创建临时 results 目录，提供可视化脚本所需的全部 fixture 数据。"""
    comparison_dir = tmp_path / "results" / "comparison"
    comparison_dir.mkdir(parents=True)
    small_sample_dir = tmp_path / "results" / "small_sample"
    small_sample_dir.mkdir(parents=True)

    # 6 数据集 × 4 模型的 comparison JSON
    for ds in DATASETS:
        for m in MODELS:
            metrics = {
                "image_AUROC": 1.0 if (ds == "bottle" and m == "patchcore") else 0.97,
                "image_AUPR": 0.99,
                "pixel_AUROC": 0.98,
                "pixel_PRO": 0.75,
            }
            (comparison_dir / f"{m}_{ds}_results.json").write_text(
                json.dumps({"metrics": metrics}, ensure_ascii=False),
                encoding="utf-8",
            )

    # 消融结果：patchcore×4 + padim×2 + fre×3
    ablation = [
        {
            "model": "patchcore",
            "param": "coreset_sampling_ratio",
            "value": v,
            "image_AUROC": 1.0,
            "image_AUPR": 0.998,
            "pixel_AUROC": 0.985,
            "pixel_PRO": 0.79 + i * 0.005,
        }
        for i, v in enumerate([0.01, 0.05, 0.1, 0.2])
    ] + [
        {
            "model": "padim",
            "param": "backbone",
            "value": backbone,
            "image_AUROC": 1.0,
            "image_AUPR": 0.998,
            "pixel_AUROC": 0.982,
            "pixel_PRO": 0.80 if backbone == "resnet18" else 0.86,
        }
        for backbone in ["resnet18", "wide_resnet50_2"]
    ] + [
        {
            "model": "fre",
            "param": "latent_dim",
            "value": v,
            "image_AUROC": 0.99,
            "image_AUPR": 0.997,
            "pixel_AUROC": 0.975,
            "pixel_PRO": 0.70 - i * 0.01,
        }
        for i, v in enumerate([100, 220, 500])
    ]
    (comparison_dir / "ablation_results.json").write_text(
        json.dumps(ablation, ensure_ascii=False), encoding="utf-8"
    )

    # 小样本总结（百分比，与真实 summary 格式一致）
    summary = {
        "category": "bottle",
        "sample_sizes": ["N30", "N60", "N100", "N150"],
        "results": {
            f"N{n}": {
                m: {
                    "image_AUROC": 100.0 - i * 0.5,
                    "image_AUPR": 99.5,
                    "pixel_AUROC": 98.0 - i * 0.3,
                    "pixel_PRO": 80.0 - i * 1.2,
                }
                for i, m in enumerate(MODELS)
            }
            for n in [30, 60, 100, 150]
        },
    }
    (small_sample_dir / "small_sample_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False), encoding="utf-8"
    )

    return tmp_path


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


def test_load_comparison_results(viz_root: Path):
    """load_comparison_results 应从 root/results/comparison 读 4 算法 × 6 数据集"""
    from tools.viz._common import load_comparison_results, DATASET_ORDER, MODEL_ORDER
    results = load_comparison_results(viz_root)
    # 锁定 6 数据集 × 4 模型契约
    assert len(results) == 6, f"应为 6 个数据集，实际 {len(results)}"
    for ds in DATASET_ORDER:
        assert ds in results, f"缺失数据集: {ds}"
        assert len(results[ds]) == 4, f"{ds} 应有 4 个模型，实际 {len(results[ds])}"
        for m in MODEL_ORDER:
            assert m in results[ds], f"{ds} 缺失模型: {m}"
    # bottle/patchcore 精确值校验
    assert results["bottle"]["patchcore"]["image_AUROC"] == 1.0


def test_load_ablation_results(viz_root: Path):
    """load_ablation_results 应返回 ablation_results.json 的列表"""
    from tools.viz._common import load_ablation_results
    results = load_ablation_results(viz_root)
    assert isinstance(results, list)
    assert len(results) == 9  # patchcore×4 + padim×2 + fre×3
    assert results[0]["model"] == "patchcore"


def test_benchmark_heatmap_generates_all_metrics(viz_root: Path):
    """基准热力图应生成 4 张 PNG（AUROC/AUPR/PixelAUROC/PRO）"""
    from tools.viz import benchmark_heatmap
    benchmark_heatmap.generate_all(viz_root)
    figures_dir = viz_root / "results" / "figures"
    for metric in ["image_AUROC", "image_AUPR", "pixel_AUROC", "pixel_PRO"]:
        assert (figures_dir / f"benchmark_heatmap_{metric}.png").exists()
        assert (figures_dir / f"benchmark_heatmap_{metric}.png").stat().st_size > 1000
    # 组合图（1×4 并排，文档 Task 9 复制到 docs/images/report/）
    assert (figures_dir / "benchmark_heatmap_all.png").exists()
    assert (figures_dir / "benchmark_heatmap_all.png").stat().st_size > 1000


def test_ablation_sensitivity_generates_three_plots(viz_root: Path):
    """消融敏感性折线图应生成 3 张（PatchCore/PaDiM/FRE）"""
    from tools.viz import ablation_sensitivity
    ablation_sensitivity.generate_all(viz_root)
    figures_dir = viz_root / "results" / "figures"
    assert (figures_dir / "ablation_patchcore_coreset_sampling_ratio.png").exists()
    assert (figures_dir / "ablation_padim_backbone.png").exists()
    assert (figures_dir / "ablation_fre_latent_dim.png").exists()


def test_small_sample_dual_axis_generates(viz_root: Path):
    """小样本双轴折线图应生成 1 张 PNG"""
    from tools.viz import small_sample_dual_axis
    small_sample_dual_axis.generate(viz_root)
    figures_dir = viz_root / "results" / "figures"
    assert (figures_dir / "small_sample_dual_axis.png").exists()
    assert (figures_dir / "small_sample_dual_axis.png").stat().st_size > 2000


def test_run_all_generates_all_figures(viz_root: Path):
    """run_all 应生成全部 8 张图表"""
    from tools.viz import run_all
    run_all.main(viz_root)
    figures_dir = viz_root / "results" / "figures"
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
        assert (figures_dir / f).exists(), f"缺失: {f}"
