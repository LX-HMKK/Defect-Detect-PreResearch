"""
数据格式转换脚本
将普通文件夹结构转换为 MVTec AD 标准格式，以便 anomalib 使用

MVTec AD 标准格式:
my_dataset/
├── ground_truth/           # 像素级标注（测试集异常区域标注）
│   └── defect_type/
│       └── 000_mask.png
├── test/                   # 测试集图片
│   ├── good/              # 正常样本
│   └── defect_type/       # 异常样本（如 scratch, dent 等）
└── train/                  # 训练集图片（仅包含正常样本）
    └── good/              # 正常样本

使用方法:
    python data_formatter.py --input_dir ./raw_data --output_dir ./data/my_dataset [--mask_dir ./masks]
"""

import os
import shutil
import argparse
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
from tqdm import tqdm


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='将普通文件夹转换为 MVTec AD 格式')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='原始数据文件夹路径，应包含 train/ 和 test/ 子文件夹')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出目录路径（MVTec AD 格式）')
    parser.add_argument('--mask_dir', type=str, default=None,
                        help='掩膜文件夹路径（可选，包含异常区域的像素级标注）')
    parser.add_argument('--image_ext', type=str, default='.png',
                        help='图像文件扩展名（如 .png, .jpg, .bmp）')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='如果从单文件夹分割，训练集比例（默认 0.8）')
    
    return parser.parse_args()


def validate_input_structure(input_dir: str) -> dict:
    """
    验证并分析输入文件夹结构
    
    支持的结构:
    1. 已分割: input_dir/train/good/ 和 input_dir/test/{good,defect}/
    2. 未分割: input_dir/good/ 和 input_dir/defect/ （自动分割）
    3. 单类别: input_dir/*.png （全部视为正常样本，自动分割）
    
    Returns:
        dict: 包含结构类型和路径信息的字典
    """
    input_path = Path(input_dir)
    structure = {
        'type': None,
        'train_normal': None,
        'test_normal': None,
        'test_defect': None,
    }
    
    # 检查结构 1: 已分割
    if (input_path / 'train' / 'good').exists():
        structure['type'] = 'pre_split'
        structure['train_normal'] = input_path / 'train' / 'good'
        structure['test_normal'] = input_path / 'test' / 'good' if (input_path / 'test' / 'good').exists() else None
        # 查找测试集中的异常类别
        test_dir = input_path / 'test'
        if test_dir.exists():
            defect_classes = [d for d in test_dir.iterdir() 
                           if d.is_dir() and d.name != 'good']
            if defect_classes:
                structure['test_defect'] = {d.name: d for d in defect_classes}
    
    # 检查结构 2: 未分割但已分类
    elif (input_path / 'good').exists():
        structure['type'] = 'unsplit'
        structure['train_normal'] = input_path / 'good'
        defect_dirs = [d for d in input_path.iterdir() 
                      if d.is_dir() and d.name != 'good']
        if defect_dirs:
            structure['test_defect'] = {d.name: d for d in defect_dirs}
    
    # 结构 3: 单文件夹
    else:
        images = list(input_path.glob('*.png')) + list(input_path.glob('*.jpg')) + list(input_path.glob('*.bmp'))
        if images:
            structure['type'] = 'single_folder'
            structure['all_images'] = images
    
    return structure


def create_mvtec_structure(output_dir: str, defect_types: list):
    """
    创建 MVTec AD 标准目录结构
    
    Args:
        output_dir: 输出根目录
        defect_types: 异常类型列表（如 ['scratch', 'dent']）
    """
    output_path = Path(output_dir)
    
    # 创建训练集目录（仅正常样本）
    (output_path / 'train' / 'good').mkdir(parents=True, exist_ok=True)
    
    # 创建测试集目录
    (output_path / 'test' / 'good').mkdir(parents=True, exist_ok=True)
    for defect in defect_types:
        (output_path / 'test' / defect).mkdir(parents=True, exist_ok=True)
        (output_path / 'ground_truth' / defect).mkdir(parents=True, exist_ok=True)
    
    print(f"✅ 已创建 MVTec AD 目录结构: {output_dir}")


def copy_images(src_paths: list, dst_dir: Path, start_idx: int = 0) -> int:
    """
    复制图片到目标目录，并规范命名
    
    Args:
        src_paths: 源图片路径列表
        dst_dir: 目标目录
        start_idx: 起始编号
    
    Returns:
        int: 下一个编号
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    idx = start_idx
    
    for src_path in tqdm(src_paths, desc=f'复制到 {dst_dir.name}'):
        if src_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
            dst_name = f'{idx:03d}{src_path.suffix}'
            dst_path = dst_dir / dst_name
            shutil.copy2(src_path, dst_path)
            idx += 1
    
    return idx


def generate_dummy_masks(test_defect_dir: Path, ground_truth_dir: Path, image_ext: str = '.png'):
    """
    为异常样本生成空白掩膜（如果没有提供真实掩膜）
    
    Args:
        test_defect_dir: 测试异常样本目录
        ground_truth_dir: 输出掩膜目录
        image_ext: 图像扩展名
    """
    ground_truth_dir.mkdir(parents=True, exist_ok=True)
    
    for img_path in test_defect_dir.glob(f'*{image_ext}'):
        # 读取图片获取尺寸
        img = cv2.imread(str(img_path))
        if img is not None:
            h, w = img.shape[:2]
            # 创建空白掩膜（全黑）
            mask = np.zeros((h, w), dtype=np.uint8)
            mask_path = ground_truth_dir / f'{img_path.stem}_mask.png'
            cv2.imwrite(str(mask_path), mask)
    
    print(f"⚠️  已为 {test_defect_dir.name} 生成空白掩膜（请替换为真实标注）")


def convert_to_mvtec(args):
    """
    主转换函数
    """
    print("=" * 60)
    print("🔄 开始转换数据格式为 MVTec AD 标准格式")
    print("=" * 60)
    
    # 1. 验证输入结构
    print("\n📁 正在分析输入文件夹结构...")
    structure = validate_input_structure(args.input_dir)
    
    if structure['type'] is None:
        raise ValueError(f"无法识别输入文件夹结构: {args.input_dir}")
    
    print(f"   检测到结构类型: {structure['type']}")
    
    # 2. 确定异常类型
    defect_types = []
    if structure.get('test_defect'):
        defect_types = list(structure['test_defect'].keys())
    else:
        defect_types = ['defect']  # 默认异常类型名称
    
    print(f"   异常类型: {defect_types}")
    
    # 3. 创建输出目录结构
    print("\n📂 创建 MVTec AD 目录结构...")
    create_mvtec_structure(args.output_dir, defect_types)
    
    output_path = Path(args.output_dir)
    
    # 4. 处理数据复制
    print("\n📥 复制训练集图片（正常样本）...")
    
    if structure['type'] == 'pre_split':
        # 已分割的结构
        train_images = list(structure['train_normal'].glob(f'*{args.image_ext}'))
        copy_images(train_images, output_path / 'train' / 'good')
        
        # 复制测试集正常样本
        if structure['test_normal']:
            test_normal_images = list(structure['test_normal'].glob(f'*{args.image_ext}'))
            copy_images(test_normal_images, output_path / 'test' / 'good')
        
        # 复制测试集异常样本
        if structure.get('test_defect'):
            for defect_name, defect_dir in structure['test_defect'].items():
                defect_images = list(defect_dir.glob(f'*{args.image_ext}'))
                copy_images(defect_images, output_path / 'test' / defect_name)
    
    elif structure['type'] == 'unsplit':
        # 未分割，需要自动分割
        all_normal = list(structure['train_normal'].glob(f'*{args.image_ext}'))
        n_train = int(len(all_normal) * args.train_ratio)
        
        train_images = all_normal[:n_train]
        test_normal_images = all_normal[n_train:]
        
        copy_images(train_images, output_path / 'train' / 'good')
        copy_images(test_normal_images, output_path / 'test' / 'good')
        
        # 复制异常样本到测试集
        if structure.get('test_defect'):
            for defect_name, defect_dir in structure['test_defect'].items():
                defect_images = list(defect_dir.glob(f'*{args.image_ext}'))
                copy_images(defect_images, output_path / 'test' / defect_name)
    
    elif structure['type'] == 'single_folder':
        # 单文件夹，自动分割
        all_images = structure['all_images']
        n_train = int(len(all_images) * args.train_ratio)
        
        train_images = all_images[:n_train]
        test_images = all_images[n_train:]
        
        copy_images(train_images, output_path / 'train' / 'good')
        copy_images(test_images, output_path / 'test' / 'good')
        print("⚠️  单文件夹模式：所有图片视为正常样本，无异常样本")
    
    # 5. 处理掩膜
    print("\n🎭 处理像素级标注掩膜...")
    
    if args.mask_dir and Path(args.mask_dir).exists():
        # 复制提供的掩膜
        mask_path = Path(args.mask_dir)
        for defect in defect_types:
            src_mask_dir = mask_path / defect
            if src_mask_dir.exists():
                dst_mask_dir = output_path / 'ground_truth' / defect
                mask_files = list(src_mask_dir.glob('*.png'))
                copy_images(mask_files, dst_mask_dir)
    else:
        # 生成空白掩膜作为占位符
        for defect in defect_types:
            test_defect_dir = output_path / 'test' / defect
            ground_truth_dir = output_path / 'ground_truth' / defect
            if test_defect_dir.exists():
                generate_dummy_masks(test_defect_dir, ground_truth_dir)
    
    # 6. 输出统计信息
    print("\n" + "=" * 60)
    print("✅ 数据格式转换完成!")
    print("=" * 60)
    print(f"\n📊 数据集统计:")
    
    train_count = len(list((output_path / 'train' / 'good').glob('*')))
    test_good_count = len(list((output_path / 'test' / 'good').glob('*')))
    
    print(f"   训练集（正常）: {train_count} 张")
    print(f"   测试集（正常）: {test_good_count} 张")
    
    for defect in defect_types:
        defect_count = len(list((output_path / 'test' / defect).glob('*')))
        print(f"   测试集（{defect}）: {defect_count} 张")
    
    print(f"\n📂 输出目录: {args.output_dir}")
    print("\n💡 提示:")
    print("   - 请检查 ground_truth/ 下的掩膜是否为真实标注")
    print("   - 训练时请在配置文件中将 category 设为文件夹名称")


def main():
    args = parse_args()
    convert_to_mvtec(args)


if __name__ == '__main__':
    main()
