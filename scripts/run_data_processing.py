#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
入口脚本 1: 数据集处理
用法: python scripts/run_data_processing.py --input_dir ./data/raw --output_dir ./data/processed/my_product
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
    print("📁 数据集处理模块")
    print("🔄 将原始图片转换为 MVTec AD 标准格式")
    print("=" * 70)
    print()
    print("📋 处理流程:")
    print("   1️⃣  扫描原始图片文件夹")
    print("   2️⃣  分离正常/异常样本")
    print("   3️⃣  限制训练集样本数量")
    print("   4️⃣  生成 MVTec AD 标准目录结构")
    print()
    print("=" * 70)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='数据集处理 - 转换为 MVTec AD 格式',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/run_data_processing.py -i ./data/raw -o ./data/processed/my_product
  python scripts/run_data_processing.py -i ./data/raw -o ./data/processed/product2 --max_train 200
        """
    )
    parser.add_argument('--input_dir', '-i', type=str, required=True,
                        help='原始数据文件夹路径')
    parser.add_argument('--output_dir', '-o', type=str, required=True,
                        help='输出目录（MVTec AD 格式）')
    parser.add_argument('--max_train', type=int, default=150,
                        help='训练集正常样本最大数量（默认150）')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    return parser.parse_args()


def main():
    """主函数"""
    print_banner()
    
    args = parse_args()
    
    # 打印配置信息
    print()
    print("⚙️  配置信息")
    print("-" * 70)
    print(f"   📥 输入目录:     {args.input_dir}")
    print(f"   📤 输出目录:     {args.output_dir}")
    print(f"   📊 最大训练样本: {args.max_train}")
    print(f"   🎲 随机种子:     {args.seed}")
    print("-" * 70)
    print()
    
    # 检查输入目录
    input_path = Path(args.input_dir)
    if not input_path.exists():
        print(f"❌ 输入目录不存在: {args.input_dir}")
        return
    
    # 执行转换
    from modules.data_processing.dataset_formatter import MVTecFormatter
    
    print("🚀 开始处理...")
    start_time = datetime.now()
    
    formatter = MVTecFormatter(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        max_train_samples=args.max_train,
        seed=args.seed
    )
    
    stats = formatter.convert()
    
    # 计算耗时
    elapsed = datetime.now() - start_time
    print(f"\n[TIME]  耗时: {elapsed.total_seconds():.1f} 秒")
    print()


if __name__ == '__main__':
    main()
