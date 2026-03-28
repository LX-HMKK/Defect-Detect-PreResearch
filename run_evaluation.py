#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
入口脚本 3: 指标评测
用法: python run_evaluation.py --model all --category bottle
"""

import sys
import io
import argparse
from pathlib import Path

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
    print("📊 模型评估模块")
    print("🔍 查看已训练模型的 4 个硬性指标")
    print("=" * 70)
    print()
    print("📋 硬性指标:")
    print("   🖼️  图像级 | AUROC    - 区分正常/异常图片的能力")
    print("   🖼️  图像级 | AUPR     - 不平衡数据集上的稳定评估")
    print("   🎯 像素级 | Pixel AUROC - 像素级异常定位精度")
    print("   🎯 像素级 | PRO      - 连续异常区域检测能力")
    print()
    print("=" * 70)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='模型评估 - 查看4个硬性指标',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_evaluation.py -m patchcore -c bottle
  python run_evaluation.py -m all -c bottle
  python run_evaluation.py -m fre -c my_product -r ./results
        """
    )
    parser.add_argument('--results_dir', '-r', type=str, default='./results',
                        help='结果目录')
    parser.add_argument('--model', '-m', type=str, default='all',
                        choices=['efficientad', 'fre', 'patchcore', 'draem', 'all'],
                        help='模型名称')
    parser.add_argument('--category', '-c', type=str, default='bottle',
                        help='产品类别')
    
    return parser.parse_args()


def main():
    """主函数"""
    print_banner()
    
    args = parse_args()
    
    # 确定要评估的模型
    models_to_eval = ['efficientad', 'fre', 'patchcore', 'draem'] if args.model == 'all' else [args.model]
    
    # 打印配置信息
    print()
    print("⚙️  配置信息")
    print("-" * 70)
    print(f"   📦 模型:       {', '.join([m.upper() for m in models_to_eval])}")
    print(f"   🏷️  产品类别:   {args.category}")
    print(f"   📁 结果目录:   {args.results_dir}")
    print("-" * 70)
    print()
    
    from modules.evaluation.metrics import load_and_evaluate
    
    for i, model_name in enumerate(models_to_eval, 1):
        print(f"\n📌 [{i}/{len(models_to_eval)}] 评估模型: {model_name.upper()}")
        print("=" * 70)
        
        try:
            load_and_evaluate(args.results_dir, model_name, args.category)
            print(f"✅ {model_name.upper()} 评估完成")
        except Exception as e:
            print(f"❌ {model_name.upper()} 评估失败: {e}")
    
    print()
    print("=" * 70)
    print("🎉 评估任务完成!")
    print("=" * 70)
    print()


if __name__ == '__main__':
    main()
