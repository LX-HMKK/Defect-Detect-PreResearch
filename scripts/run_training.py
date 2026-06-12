#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
入口脚本 2: 模型训练
用法:
    python scripts/run_training.py                    # 训练 PatchCore
    python scripts/run_training.py --model all        # 训练所有模型
    python scripts/run_training.py -m all -c all       # 训练所有模型+所有类别
"""

import io
import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 设置 Windows 终端编码为 UTF-8
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 添加项目根目录到路径
def _configure_runtime_temp() -> None:
    temp_dir = PROJECT_ROOT / "temp"
    pycache_dir = temp_dir / "pycache"
    temp_dir.mkdir(exist_ok=True)
    pycache_dir.mkdir(exist_ok=True)
    sys.pycache_prefix = str(pycache_dir)
    os.environ["PYTHONPYCACHEPREFIX"] = str(pycache_dir)


_configure_runtime_temp()

sys.path.insert(0, str(PROJECT_ROOT))


def print_banner():
    """打印欢迎横幅"""
    print()
    print("=" * 70)
    print("异常检测模型训练模块")
    print("基于 anomalib 2.x | 支持 3 种算法")
    print("=" * 70)
    print()
    print("支持的模型:")
    print("   FRE         - 基于特征重构")
    print("   PatchCore   - 基于特征建模 (工业最优)")
    print("   DRAEM       - 基于自监督学习")
    print()
    print("=" * 70)


def get_all_categories(data_path: str) -> list[str]:
    """自动发现数据目录中的所有类别（包含 train/ 子目录的文件夹）"""
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


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='异常检测模型训练 - 基于 anomalib 2.x',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/run_training.py -m patchcore -c bottle     # 训练 PatchCore
  python scripts/run_training.py -m all -c all              # 训练所有模型+所有类别
  python scripts/run_training.py -m fre -d ./data           # 指定数据路径
        """
    )

    parser.add_argument('--model', '-m', type=str, default='patchcore',
                        choices=['fre', 'patchcore', 'draem', 'padim', 'all'],
                        help='模型名称 (fre/patchcore/draem/padim/all)')
    parser.add_argument('--data_path', '-d', type=str, default='./data',
                        help='数据集路径（MVTec AD 格式）')
    parser.add_argument('--category', '-c', type=str, default='bottle',
                        help='产品类别名称 (或 all 自动发现所有类别)')
    parser.add_argument('--output_dir', '-o', type=str, default='./results',
                        help='结果输出目录')
    parser.add_argument('--eval_only', action='store_true',
                        help='仅评估模式（不训练）')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='评估时使用的权重路径')
    parser.add_argument('--device', type=str, default='auto',
                        help='计算设备 (auto/cpu/cuda)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--epochs', type=int, default=None,
                        help='最大训练轮次')
    parser.add_argument('--config', type=str, default=None,
                        help='YAML 配置文件路径（默认使用 configs/{model}.yaml）')

    return parser.parse_args()


def main():
    """主函数"""
    print_banner()

    # 解析参数
    args = parse_args()

    # 确定要运行的模型和类别
    models_to_run = ['fre', 'patchcore', 'draem', 'padim'] if args.model == 'all' else [args.model]
    categories_to_run = get_all_categories(args.data_path) if args.category == 'all' else [args.category]

    if args.category == 'all' and not categories_to_run:
        print("[ERROR] 未找到任何有效的数据类别（需包含 train/ 子目录）")
        raise SystemExit(1)

    # 打印配置信息
    print()
    print("配置信息")
    print("-" * 70)
    print(f"   模型:       {', '.join([m.upper() for m in models_to_run])}")
    print(f"   数据路径:   {args.data_path}")
    print(f"   产品类别:   {', '.join(categories_to_run)}")
    print(f"   计算设备:   {args.device}")
    print(f"   模式:       {'仅评估' if args.eval_only else '训练 + 评估'}")
    if args.epochs:
        print(f"   训练轮次:   {args.epochs}")
    print("-" * 70)
    print()

    # 执行训练
    from modules.algorithm import (
        AnomalyDetectionTrainer,
        SUPPORTED_MODELS,
        find_latest_checkpoint,
    )
    from modules.algorithm.trainer import compare_models

    total_tasks = len(models_to_run) * len(categories_to_run)
    task_idx = 0
    all_failed: list[dict] = []

    for cat_idx, category in enumerate(categories_to_run, 1):
        if len(categories_to_run) > 1:
            print(f"\n{'=' * 70}")
            print(f"[{cat_idx}/{len(categories_to_run)}] 类别: {category}")
            print(f"{'=' * 70}")

        for model_name in models_to_run:
            task_idx += 1
            print(f"\n[{task_idx}/{total_tasks}] {model_name.upper()} @ {category}")
            print("-" * 70)

            try:
                config_path = args.config
                if config_path is None:
                    default_config = PROJECT_ROOT / "configs" / f"{model_name}.yaml"
                    if default_config.exists():
                        config_path = str(default_config)

                trainer = AnomalyDetectionTrainer(
                    model_name=model_name,
                    data_path=args.data_path,
                    category=category,
                    output_dir=args.output_dir,
                    config_path=config_path,
                    device=args.device,
                    seed=args.seed
                )

                if args.eval_only:
                    resolved_checkpoint = args.checkpoint
                    if resolved_checkpoint is None:
                        latest_ckpt = find_latest_checkpoint(args.output_dir, model_name, category)
                        if latest_ckpt is None:
                            raise FileNotFoundError(
                                f"未找到可用 checkpoint: model={model_name}, category={category}。"
                                f"请先训练，或通过 --checkpoint 显式指定权重路径。"
                            )
                        resolved_checkpoint = str(latest_ckpt)
                        print(f"[INFO] 自动使用最新 checkpoint: {resolved_checkpoint}")
                    elif not Path(resolved_checkpoint).exists():
                        raise FileNotFoundError(f"checkpoint 不存在: {resolved_checkpoint}")

                    trainer.evaluate(resolved_checkpoint)
                else:
                    trainer.train_and_evaluate(max_epochs=args.epochs)

                print(f"   {model_name.upper()} @ {category} 完成")

            except Exception as e:
                print(f"   {model_name.upper()} @ {category} 失败: {e}")
                import traceback
                traceback.print_exc()
                all_failed.append({'model': model_name, 'category': category})
                continue

        # 单类别模型对比报告
        if len(models_to_run) > 1 and not args.eval_only:
            print(f"\n生成 {category} 对比报告...")
            compare_models(args.output_dir, category)

    # 最终总结
    print()
    print("=" * 70)
    print("训练任务总结")
    print("=" * 70)
    total_done = total_tasks - len(all_failed)
    print(f"   成功: {total_done}/{total_tasks}")
    if all_failed:
        print(f"   失败: {len(all_failed)}/{total_tasks}")
        for f in all_failed:
            print(f"      - {f['model'].upper()} @ {f['category']}")
        raise SystemExit(1)

    print()
    print("所有任务已完成!")
    print("=" * 70)
    print()


if __name__ == '__main__':
    main()
