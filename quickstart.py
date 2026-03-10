"""
快速启动脚本
一键完成数据转换、模型训练和演示启动

使用方法:
    python quickstart.py --input_dir ./raw_data --category my_product

参数:
    --input_dir: 原始数据文件夹路径
    --category: 产品类别名称
    --skip_data_prep: 跳过数据准备步骤
    --skip_training: 跳过训练步骤
    --model: 指定训练单个模型 (patchcore/fastflow/draem/all)
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


def run_command(cmd: str, description: str) -> bool:
    """
    运行命令并显示进度
    
    Args:
        cmd: 要运行的命令
        description: 命令描述
    
    Returns:
        bool: 是否成功
    """
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"执行命令: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"❌ {description} 失败")
        return False
    
    print(f"✅ {description} 完成")
    return True


def main():
    parser = argparse.ArgumentParser(description='快速启动脚本')
    parser.add_argument('--input_dir', type=str, default=None,
                        help='原始数据文件夹路径')
    parser.add_argument('--category', type=str, required=True,
                        help='产品类别名称')
    parser.add_argument('--skip_data_prep', action='store_true',
                        help='跳过数据准备步骤')
    parser.add_argument('--skip_training', action='store_true',
                        help='跳过训练步骤')
    parser.add_argument('--model', type=str, default='all',
                        choices=['patchcore', 'autoencoder', 'draem', 'all'],
                        help='要训练的模型 (3个最易复现的算法)'
    
    args = parser.parse_args()
    
    print("="*60)
    print("🎯 无监督工业异常检测 - 快速启动")
    print("="*60)
    
    # 步骤 1: 数据准备
    if not args.skip_data_prep:
        if args.input_dir is None:
            print("\n⚠️  未指定 input_dir，跳过数据准备")
            print("   如果数据已准备好，请使用 --skip_data_prep 跳过此步骤")
        else:
            # 运行数据转换脚本
            data_cmd = (
                f"python data_formatter.py "
                f"--input_dir {args.input_dir} "
                f"--output_dir ./data/{args.category} "
                f"--category {args.category}"
            )
            
            if not run_command(data_cmd, "数据格式转换"):
                print("\n❌ 数据准备失败，请检查输入数据格式")
                sys.exit(1)
            
            # 更新配置文件中的类别名称
            for model in ['patchcore', 'fastflow', 'draem']:
                config_path = Path(f'./configs/{model}.yaml')
                if config_path.exists():
                    with open(config_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # 替换 category
                    content = content.replace('my_product', args.category)
                    content = content.replace('category: my_dataset', f'category: {args.category}')
                    
                    with open(config_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    print(f"   📝 已更新配置文件: {config_path}")
    
    # 步骤 2: 训练模型
    if not args.skip_training:
        train_cmd = (
            f"python train_evaluate.py "
            f"--model {args.model} "
            f"--category {args.category} "
            f"--data_path ./data/{args.category}"
        )
        
        if not run_command(train_cmd, "模型训练"):
            print("\n❌ 模型训练失败")
            sys.exit(1)
    
    # 步骤 3: 启动演示界面
    print(f"\n{'='*60}")
    print("🎉 所有步骤已完成!")
    print(f"{'='*60}")
    print("\n现在可以启动演示界面:")
    print("   python app.py")
    print("\n然后访问: http://127.0.0.1:7860")
    
    # 询问是否立即启动
    try:
        response = input("\n是否立即启动演示界面? (y/n): ").strip().lower()
        if response in ['y', 'yes', '是']:
            run_command("python app.py", "启动演示界面")
    except KeyboardInterrupt:
        print("\n\n👋 已取消")


if __name__ == '__main__':
    main()
