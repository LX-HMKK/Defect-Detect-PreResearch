"""一键生成全部可视化图表到 results/figures/

依次调用基准热力图、消融敏感性折线、小样本双轴折线三个脚本，
共生成 8 张图表（4+3+1）。供文档图表替换与持续复现使用。
"""
import sys
import io
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# stdout/stderr 包装仅在 __main__ 执行，避免破坏 pytest 捕获（同 benchmark_heatmap 等模块约定）
if __name__ == "__main__" and sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(PROJECT_ROOT))

from modules._runtime import configure_runtime_temp

if __name__ == "__main__":
    configure_runtime_temp()

from tools.viz import benchmark_heatmap, ablation_sensitivity, small_sample_dual_axis


def main(root: Path | None = None):
    print("=" * 60)
    print("生成全部可视化图表 → results/figures/")
    print("=" * 60)
    print("\n[1/3] 基准对比热力图...")
    benchmark_heatmap.generate_all(root)
    print("\n[2/3] 消融敏感性折线图...")
    ablation_sensitivity.generate_all(root)
    print("\n[3/3] 小样本双轴折线图...")
    small_sample_dual_axis.generate(root)
    print("\n" + "=" * 60)
    print("全部图表生成完毕!")
    print("  基准热力图: benchmark_heatmap_*.png (4 张 + 1 组合)")
    print("  消融折线图: ablation_*.png (3 张)")
    print("  小样本图:   small_sample_dual_axis.png (1 张)")
    print("=" * 60)


if __name__ == "__main__":
    main()
