#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
小样本鲁棒性分析脚本

在 30/60/100/150 张正常样本下训练并评估各算法，
分析不同算法在小样本条件下的性能变化规律。

用法:
    python tools/run_small_sample.py -m all -c all -d ./data
    python tools/run_small_sample.py -m patchcore -c bottle -d ./data
"""

import io
import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(PROJECT_ROOT))

SAMPLE_SIZES = [30, 60, 100, 150]
RESULTS_FILENAME = "small_sample_results.json"


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='小样本鲁棒性分析 - 评估算法在不同样本量下的性能'
    )
    parser.add_argument('--model', '-m', type=str, default='all',
                        choices=['fre', 'patchcore', 'draem', 'padim', 'all'])
    parser.add_argument('--data_path', '-d', type=str, default='./data')
    parser.add_argument('--category', '-c', type=str, default='bottle')
    parser.add_argument('--output_dir', '-o', type=str, default='./results/small_sample')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--sample_sizes', '-s', type=str, default='30,60,100,150',
                        help='逗号分隔的样本量，默认 30,60,100,150')
    return parser.parse_args()


def sample_training_data(
    src_data_dir: Path,
    category: str,
    n_samples: int,
    temp_root: Path,
    seed: int = 42,
) -> Path:
    """从原始数据集中采样 N 张正常图像，创建临时数据集目录"""
    import random
    random.seed(seed)

    src_category = src_data_dir / category
    train_good = src_category / 'train' / 'good'

    if not train_good.exists():
        raise FileNotFoundError(f"训练数据不存在: {train_good}")

    all_images = sorted(list(train_good.glob('*')))
    all_images = [f for f in all_images if f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')]

    if len(all_images) < n_samples:
        print(f"   [WARN] 可用训练图像 ({len(all_images)}) 少于请求样本量 ({n_samples})，使用全部图像")
        n_samples = len(all_images)

    sampled = random.sample(all_images, n_samples)

    # 创建临时数据集
    temp_category = temp_root / category
    temp_train_good = temp_category / 'train' / 'good'
    temp_train_good.mkdir(parents=True, exist_ok=True)

    for img in sampled:
        dest = temp_train_good / img.name
        if not dest.exists():
            shutil.copy2(img, dest)

    # 复制 test 和 ground_truth（不做修改）
    for subdir in ['test', 'ground_truth']:
        src_sub = src_category / subdir
        if src_sub.exists():
            dst_sub = temp_category / subdir
            if dst_sub.exists():
                shutil.rmtree(dst_sub)
            shutil.copytree(src_sub, dst_sub)

    return temp_category


def run_experiment(
    model_name: str,
    data_path: str,
    category: str,
    output_dir: str,
    config_path: Optional[str],
    device: str,
    seed: int,
    max_epochs: Optional[int],
) -> dict:
    """运行单次训练+评估实验，返回指标字典"""
    from modules.algorithm.trainer import AnomalyDetectionTrainer

    trainer = AnomalyDetectionTrainer(
        model_name=model_name,
        data_path=data_path,
        category=category,
        output_dir=output_dir,
        config_path=config_path,
        device=device,
        seed=seed,
    )
    result = trainer.train_and_evaluate(max_epochs=max_epochs)
    return result


def load_config_path(model_name: str) -> Optional[str]:
    default_config = PROJECT_ROOT / 'configs' / f'{model_name}.yaml'
    return str(default_config) if default_config.exists() else None


def main():
    args = parse_args()
    sample_sizes = [int(x.strip()) for x in args.sample_sizes.split(',')]

    models_to_run = ['fre', 'patchcore', 'draem', 'padim'] if args.model == 'all' else [args.model]
    categories_to_run = (
        get_all_categories(args.data_path) if args.category == 'all'
        else [args.category]
    )

    if not categories_to_run:
        print("[ERROR] 未找到有效类别")
        raise SystemExit(1)

    temp_root = Path(args.output_dir) / '_temp_data'
    output_root = Path(args.output_dir)

    print()
    print("=" * 70)
    print("小样本鲁棒性分析")
    print("=" * 70)
    print(f"  模型: {', '.join([m.upper() for m in models_to_run])}")
    print(f"  类别: {', '.join(categories_to_run)}")
    print(f"  样本量: {sample_sizes}")
    print(f"  数据路径: {args.data_path}")
    print("=" * 70)

    all_results: Dict[str, dict] = {}

    total = len(models_to_run) * len(categories_to_run) * len(sample_sizes)
    task_idx = 0

    for category in categories_to_run:
        category_results: Dict[int, Dict[str, dict]] = {}

        for n_samples in sample_sizes:
            task_idx += 1
            print(f"\n{'=' * 70}")
            print(f"[{task_idx}/{total}] 准备: {category} @ N={n_samples}")
            print(f"{'=' * 70}")

            # 清理之前的临时数据（Windows 兼容：重试 + 错误处理）
            if temp_root.exists():
                import time
                for _ in range(3):
                    try:
                        shutil.rmtree(temp_root, ignore_errors=False)
                        break
                    except PermissionError:
                        time.sleep(0.5)
                else:
                    shutil.rmtree(temp_root, ignore_errors=True)
            temp_root.mkdir(parents=True, exist_ok=True)

            try:
                sample_training_data(
                    Path(args.data_path), category, n_samples, temp_root, args.seed
                )
                print(f"   已采样 {n_samples} 张训练图像")
            except Exception as e:
                print(f"   [ERROR] 采样失败: {e}")
                continue

            for model_name in models_to_run:
                print(f"\n   [{model_name.upper()}] 训练中 (N={n_samples})...")
                print(f"   {'-' * 60}")

                try:
                    config_path = load_config_path(model_name)
                    result = run_experiment(
                        model_name=model_name,
                        data_path=str(temp_root),
                        category=category,
                        output_dir=str(temp_root / '_results'),
                        config_path=config_path,
                        device=args.device,
                        seed=args.seed,
                        max_epochs=args.epochs,
                    )

                    metrics = {
                        'image_AUROC': round(result.get('image_AUROC', 0) * 100, 2),
                        'image_AUPR': round(result.get('image_AUPR', 0) * 100, 2),
                        'pixel_AUROC': round(result.get('pixel_AUROC', 0) * 100, 2),
                        'pixel_PRO': round(result.get('pixel_PRO', 0) * 100, 2),
                    }

                    if n_samples not in category_results:
                        category_results[n_samples] = {}
                    category_results[n_samples][model_name] = metrics

                    auroc = metrics['image_AUROC']
                    print(f"   [{model_name.upper()}] N={n_samples}: image_AUROC={auroc:.2f}%")

                except Exception as e:
                    print(f"   [{model_name.upper()}] 失败 (N={n_samples}): {e}")
                    import traceback
                    traceback.print_exc()
                    continue

        all_results[category] = category_results

    # 清理临时数据
    if temp_root.exists():
        shutil.rmtree(temp_root)

    # 保存结果
    output_root.mkdir(parents=True, exist_ok=True)
    results_path = output_root / RESULTS_FILENAME

    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n{'=' * 70}")
    print(f"结果已保存: {results_path}")
    print(f"{'=' * 70}")

    # 打印汇总表
    print_summary_table(all_results, sample_sizes, models_to_run)


def print_summary_table(
    all_results: dict,
    sample_sizes: list,
    models_to_run: list,
):
    """打印小样本性能汇总表"""
    print()
    print("=" * 90)
    print("小样本鲁棒性分析 - 汇总表 (image_AUROC %)")
    print("=" * 90)

    for category, cat_results in all_results.items():
        print(f"\n--- {category} ---")
        header = f"{'Model':<12}"
        for n in sample_sizes:
            header += f" | N={n:>4}"
        print(header)
        print("-" * (12 + len(sample_sizes) * 10))

        for model_name in models_to_run:
            row = f"{model_name.upper():<12}"
            for n in sample_sizes:
                if n in cat_results and model_name in cat_results[n]:
                    auroc = cat_results[n][model_name]['image_AUROC']
                    row += f" | {auroc:>6.2f}"
                else:
                    row += f" | {'-':>6}"
            print(row)

    print()
    print("=" * 90)

    # 生成 Markdown 报告
    generate_markdown_report(all_results, sample_sizes, models_to_run)


def generate_markdown_report(
    all_results: dict,
    sample_sizes: list,
    models_to_run: list,
):
    """生成 Markdown 格式的小样本分析报告"""
    output_root = Path('./results/small_sample')
    output_root.mkdir(parents=True, exist_ok=True)
    md_path = output_root / 'small_sample_report.md'

    lines = []
    lines.append("# 小样本鲁棒性分析报告\n")
    lines.append(f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"> 样本量: {', '.join([str(n) for n in sample_sizes])}\n")

    for category, cat_results in all_results.items():
        lines.append(f"## {category}\n")

        for metric_name, metric_label in [
            ('image_AUROC', '图像级 AUROC'),
            ('image_AUPR', '图像级 AUPR'),
            ('pixel_AUROC', '像素级 AUROC'),
            ('pixel_PRO', '像素级 PRO'),
        ]:
            lines.append(f"### {metric_label} (%)\n")
            lines.append(f"| Model | " + " | ".join([f"N={n}" for n in sample_sizes]) + " |")
            lines.append("|" + "|".join([":---:" for _ in range(len(sample_sizes) + 1)]) + "|")

            for model_name in models_to_run:
                row = f"| {model_name.upper()} |"
                for n in sample_sizes:
                    if n in cat_results and model_name in cat_results[n]:
                        val = cat_results[n][model_name].get(metric_name, '-')
                        row += f" {val:.2f} |"
                    else:
                        row += " - |"
                lines.append(row)

            lines.append("")

        lines.append("---\n")

    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"Markdown 报告: {md_path}")


if __name__ == '__main__':
    main()
