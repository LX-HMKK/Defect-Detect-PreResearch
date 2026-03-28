#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
入口脚本 2: 模型训练
用法: 
    python run_training.py                    # 训练 Ganomaly
    python run_training.py --model all        # 训练所有模型
    python run_training.py -m patchcore       # 指定模型
"""

import sys
import io
import argparse
from pathlib import Path
from datetime import datetime

# 设置 Windows 终端编码为 UTF-8
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))


def print_banner():
    """打印欢迎横幅"""
    print()
    print("=" * 70)
    print("🚀 异常检测模型训练模块")
    print("📚 基于 anomalib 2.x | 支持 3 种算法")
    print("=" * 70)
    print()
    print("📋 支持的模型:")
    print("   🔹 Ganomaly    - 基于重构 (GAN)")
    print("   🔹 PatchCore   - 基于特征建模 (工业最优)")
    print("   🔹 DRAEM       - 基于自监督学习")
    print()
    print("=" * 70)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='异常检测模型训练 - 基于 anomalib 2.x',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_training.py -m patchcore -c bottle     # 训练 PatchCore
  python run_training.py -m all                     # 训练所有模型
  python run_training.py -m ganomaly -d ./data      # 指定数据路径
        """
    )
    
    parser.add_argument('--model', '-m', type=str, default='patchcore',
                        choices=['ganomaly', 'patchcore', 'draem', 'all'],
                        help='模型名称 (ganomaly/patchcore/draem/all)')
    parser.add_argument('--data_path', '-d', type=str, default='./data',
                        help='数据集路径（MVTec AD 格式）')
    parser.add_argument('--category', '-c', type=str, default='bottle',
                        help='产品类别名称')
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
    
    return parser.parse_args()


def main():
    """主函数"""
    print_banner()
    
    # 解析参数
    args = parse_args()
    
    # 确定要运行的模型
    models_to_run = ['ganomaly', 'patchcore', 'draem'] if args.model == 'all' else [args.model]
    
    # 打印配置信息
    print()
    print("⚙️  配置信息")
    print("-" * 70)
    print(f"   📦 模型:       {', '.join([m.upper() for m in models_to_run])}")
    print(f"   📁 数据路径:   {args.data_path}")
    print(f"   🏷️  产品类别:   {args.category}")
    print(f"   💻 计算设备:   {args.device}")
    print(f"   📝 模式:       {'仅评估' if args.eval_only else '训练 + 评估'}")
    if args.epochs:
        print(f"   🔄 训练轮次:   {args.epochs}")
    print("-" * 70)
    print()
    
    # 执行训练
    from modules.algorithm.trainer import (
        AnomalyDetectionTrainer,
        SUPPORTED_MODELS,
        compare_models,
    )
    
    results_summary = []
    
    for i, model_name in enumerate(models_to_run, 1):
        print(f"\n📌 [{i}/{len(models_to_run)}] 处理模型: {model_name.upper()}")
        print("=" * 70)
        
        try:
            trainer = AnomalyDetectionTrainer(
                model_name=model_name,
                data_path=args.data_path,
                category=args.category,
                output_dir=args.output_dir,
                device=args.device,
                seed=args.seed
            )
            
            if args.eval_only:
                trainer.evaluate(args.checkpoint)
            else:
                result = trainer.train_and_evaluate(max_epochs=args.epochs)
                results_summary.append({
                    'model': model_name.upper(),
                    'result': result
                })
            
            print(f"✅ {model_name.upper()} 完成")
                
        except Exception as e:
            print(f"❌ {model_name.upper()} 失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 生成对比报告
    if len(models_to_run) > 1:
        print("\n" + "=" * 70)
        print("📊 生成模型对比报告...")
        compare_models(args.output_dir, args.category)
    
    # 最终总结
    print()
    print("=" * 70)
    print("📋 训练任务总结")
    print("=" * 70)
    
    if results_summary:
        for r in results_summary:
            result = r['result']
            auroc = result.get('image_AUROC', 0) * 100
            status = "🌟 优秀" if auroc >= 95 else "👍 良好" if auroc >= 80 else "⚠️ 一般"
            print(f"   {r['model']:12s} | AUROC: {auroc:6.2f}% | {status}")
    
    print()
    print("🎉 所有任务已完成!")
    print("=" * 70)
    print()


if __name__ == '__main__':
    main()
