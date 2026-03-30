#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
入口脚本: 独立计算最优阈值
用法:
    python run_threshold.py -m patchcore -c region1              # 计算阈值
    python run_threshold.py -m all -c region1                    # 计算所有模型阈值
    python run_threshold.py -m patchcore -c region1 --save      # 计算并保存
"""

import sys
import io
import json
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
    print("🔧 阈值计算模块")
    print("=" * 70)


def get_all_categories(data_path: str) -> list:
    """扫描数据目录，获取所有可用的类别"""
    data_dir = Path(data_path)
    if not data_dir.exists():
        return []
    
    categories = []
    for item in data_dir.iterdir():
        if item.is_file() or item.name.startswith('.'):
            continue
        if (item / 'train').exists():
            categories.append(item.name)
    
    return sorted(categories)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='独立计算最优阈值脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_threshold.py -m patchcore -c bottle     # 计算 PatchCore bottle 阈值
  python run_threshold.py -m all -c bottle           # 计算所有模型 bottle 阈值
  python run_threshold.py -m all -c all -d ./data   # 计算所有模型所有类别阈值
  python run_threshold.py -m patchcore -c bottle --save  # 计算并保存到 JSON
        """
    )
    
    parser.add_argument('--model', '-m', type=str, default='patchcore',
                        choices=['patchcore', 'fre', 'draem', 'all'],
                        help='模型名称')
    parser.add_argument('--data_path', '-d', type=str, default='./data',
                        help='数据集路径')
    parser.add_argument('--category', '-c', type=str, default='bottle',
                        help='产品类别名称 (bottle/carpet/region1/all)')
    parser.add_argument('--output_dir', '-o', type=str, default='./results',
                        help='结果输出目录')
    parser.add_argument('--save', action='store_true',
                        help='保存阈值到结果 JSON 文件')
    parser.add_argument('--device', type=str, default='auto',
                        help='计算设备 (auto/cpu/cuda)')
    
    return parser.parse_args()


def compute_threshold(model_name: str, data_path: str, category: str, 
                      output_dir: str, device: str, save: bool):
    """计算单个模型的阈值"""
    from modules.algorithm.trainer import AnomalyDetectionTrainer
    
    print(f"\n{'='*70}")
    print(f"📌 计算阈值: {model_name.upper()} + {category}")
    print(f"{'='*70}")
    
    try:
        # 创建训练器
        trainer = AnomalyDetectionTrainer(
            model_name=model_name,
            data_path=data_path,
            category=category,
            output_dir=output_dir,
            device=device,
            seed=42
        )
        
        # 设置（加载数据和模型）
        trainer.setup()
        
        # 创建引擎
        from anomalib.engine import Engine
        trainer.engine = Engine(
            accelerator=device,
            devices=1,
            default_root_dir=str(Path(output_dir) / model_name),
            enable_progress_bar=False,
        )
        
        # 计算阈值
        threshold = trainer._compute_optimal_threshold()
        
        # 打印结果
        print(f"\n✅ 计算完成!")
        print(f"   模型: {model_name.upper()}")
        print(f"   类别: {category}")
        print(f"   最优阈值: {threshold:.3f}")
        
        # 如果需要保存
        if save:
            trainer._save_results()
            print(f"   已保存到: {output_dir}/comparison/{model_name}_{category}_results.json")
        
        return threshold
        
    except Exception as e:
        print(f"\n❌ 计算失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """主函数"""
    print_banner()
    
    args = parse_args()
    
    # 确定要运行的模型
    models = ['fre', 'patchcore', 'draem'] if args.model == 'all' else [args.model]
    
    # 确定要运行的数据类别
    if args.category == 'all':
        categories = get_all_categories(args.data_path)
        if not categories:
            print("❌ 错误: 数据目录中没有找到有效的类别")
            return
        print(f"📂 检测到 {len(categories)} 个数据类别: {', '.join(categories)}")
    else:
        categories = [args.category]
    
    # 打印配置
    print()
    print("⚙️  配置信息")
    print("-" * 70)
    print(f"   📦 模型:       {', '.join([m.upper() for m in models])}")
    print(f"   📁 数据路径:   {args.data_path}")
    print(f"   🏷️  类别:       {', '.join(categories)}")
    print(f"   💾 保存:        {'是' if args.save else '否'}")
    print(f"   💻 设备:       {args.device}")
    print("-" * 70)
    
    # 计算每个模型每个类别的阈值
    results = []
    for category in categories:
        for model in models:
            threshold = compute_threshold(
                model_name=model,
                data_path=args.data_path,
                category=category,
                output_dir=args.output_dir,
                device=args.device,
                save=args.save
            )
            if threshold is not None:
                results.append({
                    'model': model,
                    'category': category,
                    'threshold': threshold
                })
    
    # 打印汇总
    if results:
        print()
        print("=" * 70)
        print("📊 阈值计算结果汇总")
        print("=" * 70)
        print(f"{'模型':<12} {'类别':<10} {'阈值':<10}")
        print("-" * 70)
        for r in results:
            print(f"{r['model']:<12} {r['category']:<10} {r['threshold']:.3f}")
        print("=" * 70)


if __name__ == '__main__':
    main()
