#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据集统计分析脚本

统计所有数据集的样本量、缺陷类型分布、训练/测试集构成等。
用于实验报告中的数据集描述。

用法:
    python tools/run_data_stats.py -d ./data
"""

import io
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(PROJECT_ROOT))


def get_all_categories(data_path: str) -> list[str]:
    data_dir = Path(data_path)
    if not data_dir.exists():
        return []
    categories: list[str] = []
    for item in sorted(data_dir.iterdir()):
        if item.is_file() or item.name.startswith('.'):
            continue
        if (item / 'train').exists():
            categories.append(item.name)
    return categories


def analyze_category(data_path: Path, category: str) -> dict:
    """分析单个类别的数据统计"""
    cat_path = data_path / category
    train_good = cat_path / 'train' / 'good'
    test_dir = cat_path / 'test'
    gt_dir = cat_path / 'ground_truth'

    # 训练正常样本
    train_normal = count_images(train_good) if train_good.exists() else 0

    # 测试正常样本
    test_normal = count_images(test_dir / 'good') if (test_dir / 'good').exists() else 0

    # 测试异常样本
    test_defects = {}
    total_defect = 0
    if test_dir.exists():
        for subdir in sorted(test_dir.iterdir()):
            if subdir.is_dir() and subdir.name != 'good':
                count = count_images(subdir)
                test_defects[subdir.name] = count
                total_defect += count

    # Ground truth
    gt_defects = {}
    if gt_dir.exists():
        for subdir in sorted(gt_dir.iterdir()):
            if subdir.is_dir():
                gt_defects[subdir.name] = count_images(subdir)

    total_test = test_normal + total_defect
    defect_ratio = (total_defect / total_test * 100) if total_test > 0 else 0

    return {
        'category': category,
        'train_normal': train_normal,
        'test_normal': test_normal,
        'test_defect': total_defect,
        'test_total': total_test,
        'defect_ratio': round(defect_ratio, 1),
        'defect_types': sorted(test_defects.keys()),
        'defect_counts': test_defects,
        'gt_counts': gt_defects,
        'is_public': category in ('bottle', 'carpet', 'leather', 'grid', 'tile', 'wood',
                                   'cable', 'capsule', 'hazelnut', 'metal_nut', 'pill',
                                   'screw', 'toothbrush', 'transistor', 'zipper'),
    }


def count_images(directory: Path) -> int:
    """统计目录中的图像文件数量"""
    if not directory.exists():
        return 0
    exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}
    return sum(1 for f in directory.iterdir()
               if f.is_file() and f.suffix.lower() in exts)


def generate_stats_report(all_stats: list, output_path: Path):
    """生成数据集统计 Markdown 报告"""
    lines = []
    lines.append("# 数据集统计分析\n")
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 汇总表
    lines.append("## 数据集概览\n")
    lines.append("| Category | 类型 | 训练(正常) | 测试(正常) | 测试(缺陷) | 缺陷比 | 缺陷类型 |")
    lines.append("|:---|:---|:---:|:---:|:---:|:---:|:---|")

    total_train = 0
    total_test = 0
    total_defects = 0

    for s in all_stats:
        dtype = "公开" if s['is_public'] else "企业"
        defects_str = ', '.join(s['defect_types']) if s['defect_types'] else '-'
        lines.append(
            f"| {s['category']} | {dtype} | {s['train_normal']} | "
            f"{s['test_normal']} | {s['test_defect']} | "
            f"{s['defect_ratio']}% | {defects_str} |"
        )
        total_train += s['train_normal']
        total_test += s['test_total']
        total_defects += s['test_defect']

    lines.append(f"| **合计** | | **{total_train}** | **{total_test - total_defects}** | "
                 f"**{total_defects}** | **{round(total_defects / total_test * 100, 1) if total_test > 0 else 0}%** | |")
    lines.append("")

    # 详细分布
    lines.append("## 缺陷类型详细分布\n")
    for s in all_stats:
        if s['defect_counts']:
            lines.append(f"### {s['category']}\n")
            lines.append("| 缺陷类型 | 测试集数量 | GT 标注数量 |")
            lines.append("|:---|:---:|:---:|")
            for dtype in s['defect_types']:
                test_cnt = s['defect_counts'].get(dtype, 0)
                gt_cnt = s['gt_counts'].get(dtype, 0)
                lines.append(f"| {dtype} | {test_cnt} | {gt_cnt} |")
            lines.append("")

    # 训练样本约束分析
    lines.append("## 训练样本约束分析\n")
    for s in all_stats:
        available = s['train_normal']
        status = "✅ 满足" if available >= 150 else f"⚠️ 不足 ({available}/150)"
        lines.append(f"- **{s['category']}**: 可用 {available} 张正常样本，150 张约束: {status}")

    lines.append(f"\n---\n*报告由 run_data_stats.py 自动生成*")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    return output_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description='数据集统计分析')
    parser.add_argument('--data_path', '-d', type=str, default='./data')
    parser.add_argument('--output', '-o', type=str, default='./results/comparison/data_stats.md')
    args = parser.parse_args()

    categories = get_all_categories(args.data_path)
    if not categories:
        print("[ERROR] 未找到数据集类别")
        raise SystemExit(1)

    print(f"发现 {len(categories)} 个数据集: {', '.join(categories)}")

    all_stats = []
    for cat in categories:
        stats = analyze_category(Path(args.data_path), cat)
        all_stats.append(stats)
        print(f"  {cat}: train={stats['train_normal']}, "
              f"test=[normal={stats['test_normal']}, defect={stats['test_defect']}], "
              f"defect_ratio={stats['defect_ratio']}%")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generate_stats_report(all_stats, output_path)
    print(f"\n报告已保存: {output_path}")


if __name__ == '__main__':
    main()
