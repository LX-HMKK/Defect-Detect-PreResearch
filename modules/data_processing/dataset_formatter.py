"""
================================================================================
模块 1: 数据集处理模块 (Data Processing Module)
================================================================================

功能: 将企业提供的原始图片整理成 MVTec AD 标准格式

处理流程:
    1. 读取原始图片文件夹
    2. 分离正常样本和异常样本
    3. 限制训练集正常样本数量 ≤ 150 张
    4. 生成 MVTec AD 格式的目录结构
    5. 处理像素级标注掩膜（如果有）

输出格式 (MVTec AD 标准):
    dataset/
    ├── train/good/           # 训练集：仅正常样本 (≤150张)
    ├── test/good/            # 测试集：正常样本
    ├── test/defect/          # 测试集：异常样本
    └── ground_truth/defect/  # 掩膜：像素级标注

使用示例:
    from modules.data_processing.dataset_formatter import MVTecFormatter
    
    formatter = MVTecFormatter(
        input_dir="./data/raw/enterprise_data",
        output_dir="./data/processed/my_product",
        max_train_samples=150
    )
    formatter.convert()
================================================================================
"""

import os
import shutil
import random
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import argparse

import cv2
import numpy as np
from tqdm import tqdm


class MVTecFormatter:
    """
    MVTec AD 格式数据转换器
    
    将企业原始数据转换为学术界标准的 MVTec AD 格式，
    并确保训练集正常样本数量不超过指定上限（默认150张）
    """
    
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        max_train_samples: int = 150,
        image_extensions: Tuple[str, ...] = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'),
        seed: int = 42
    ):
        """
        初始化转换器
        
        Args:
            input_dir: 原始数据文件夹路径
            output_dir: 输出目录（MVTec AD 格式）
            max_train_samples: 训练集正常样本最大数量（默认150）
            image_extensions: 支持的图片格式
            seed: 随机种子（用于采样）
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.max_train_samples = max_train_samples
        self.image_extensions = image_extensions
        self.seed = seed
        
        # 设置随机种子
        random.seed(seed)
        np.random.seed(seed)
        
        # 统计信息
        self.stats = {
            'train_normal': 0,
            'test_normal': 0,
            'test_defect': 0,
            'truncated': 0  # 因超过上限而被截断的样本数
        }
    
    def _find_images(self, directory: Path) -> List[Path]:
        """
        查找目录下的所有图片文件
        
        Args:
            directory: 目标目录
        
        Returns:
            List[Path]: 图片文件路径列表
        """
        images = []
        for ext in self.image_extensions:
            images.extend(directory.glob(f'*{ext}'))
            images.extend(directory.glob(f'*{ext.upper()}'))
        return sorted(images)
    
    def _detect_structure(self) -> Dict:
        """
        自动检测输入文件夹结构
        
        支持的输入结构:
        1. 已分割: raw/train/good/ 和 raw/test/{good,defect}/
        2. 半分割: raw/good/ 和 raw/defect/ （自动按比例分割）
        3. 单文件夹: raw/*.png （所有图片视为正常，按比例分割）
        
        Returns:
            Dict: 包含检测到的结构信息
        """
        structure = {
            'type': None,
            'train_normal': [],
            'test_normal': [],
            'test_defect': [],
            'defect_types': []
        }
        
        # 尝试检测结构 1: 已分割
        train_good_dir = self.input_dir / 'train' / 'good'
        test_dir = self.input_dir / 'test'
        
        if train_good_dir.exists():
            structure['type'] = 'pre_split'
            structure['train_normal'] = self._find_images(train_good_dir)
            
            # 检测测试集
            if test_dir.exists():
                test_good_dir = test_dir / 'good'
                if test_good_dir.exists():
                    structure['test_normal'] = self._find_images(test_good_dir)
                
                # 检测异常类别
                for subdir in test_dir.iterdir():
                    if subdir.is_dir() and subdir.name != 'good':
                        structure['defect_types'].append(subdir.name)
                        structure['test_defect'].extend(self._find_images(subdir))
        
        # 尝试检测结构 2: 半分割
        elif (self.input_dir / 'good').exists():
            structure['type'] = 'semi_split'
            all_normal = self._find_images(self.input_dir / 'good')
            
            # 按比例分割：80% 训练，20% 测试，但训练不超过 max_train_samples
            n_train = min(int(len(all_normal) * 0.8), self.max_train_samples)
            structure['train_normal'] = all_normal[:n_train]
            structure['test_normal'] = all_normal[n_train:]
            
            # 检测异常类别
            for subdir in self.input_dir.iterdir():
                if subdir.is_dir() and subdir.name != 'good':
                    structure['defect_types'].append(subdir.name)
                    structure['test_defect'].extend(self._find_images(subdir))
        
        # 结构 3: 单文件夹
        else:
            all_images = self._find_images(self.input_dir)
            if all_images:
                structure['type'] = 'single_folder'
                # 按比例分割，训练不超过 max_train_samples
                n_train = min(int(len(all_images) * 0.8), self.max_train_samples)
                structure['train_normal'] = all_images[:n_train]
                structure['test_normal'] = all_images[n_train:]
        
        return structure
    
    def _limit_train_samples(self, images: List[Path]) -> List[Path]:
        """
        限制训练集样本数量
        
        Args:
            images: 原始图片列表
        
        Returns:
            List[Path]: 限制后的图片列表
        """
        if len(images) > self.max_train_samples:
            self.stats['truncated'] = len(images) - self.max_train_samples
            print(f"⚠️  训练集样本过多 ({len(images)} 张)，将随机采样保留 {self.max_train_samples} 张")
            return random.sample(images, self.max_train_samples)
        return images
    
    def _copy_images(self, images: List[Path], dst_dir: Path, desc: str = "复制") -> int:
        """
        复制图片到目标目录
        
        Args:
            images: 源图片路径列表
            dst_dir: 目标目录
            desc: 进度条描述
        
        Returns:
            int: 成功复制的数量
        """
        dst_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        
        for idx, src_path in enumerate(tqdm(images, desc=desc)):
            dst_name = f"{idx:03d}{src_path.suffix.lower()}"
            dst_path = dst_dir / dst_name
            try:
                shutil.copy2(src_path, dst_path)
                count += 1
            except Exception as e:
                print(f"❌ 复制失败 {src_path}: {e}")
        
        return count
    
    def _generate_dummy_masks(self, test_defect_dir: Path, mask_output_dir: Path):
        """
        生成空白掩膜作为占位符（如果没有提供真实掩膜）
        
        Args:
            test_defect_dir: 测试异常样本目录
            mask_output_dir: 掩膜输出目录
        """
        mask_output_dir.mkdir(parents=True, exist_ok=True)
        
        for img_path in self._find_images(test_defect_dir):
            img = cv2.imread(str(img_path))
            if img is not None:
                h, w = img.shape[:2]
                mask = np.zeros((h, w), dtype=np.uint8)
                mask_path = mask_output_dir / f"{img_path.stem}_mask.png"
                cv2.imwrite(str(mask_path), mask)
        
        print(f"⚠️  已为 {test_defect_dir.name} 生成空白掩膜（请替换为真实标注）")
    
    def convert(self, defect_types: Optional[List[str]] = None) -> Dict:
        """
        执行数据格式转换
        
        Args:
            defect_types: 指定异常类型名称列表（可选）
        
        Returns:
            Dict: 转换统计信息
        """
        print("="*70)
        print("🔄 数据集处理模块 (Data Processing Module)")
        print("="*70)
        print(f"\n📂 输入目录: {self.input_dir}")
        print(f"📂 输出目录: {self.output_dir}")
        print(f"📊 训练集上限: {self.max_train_samples} 张")
        
        # 1. 检测输入结构
        print("\n🔍 检测输入文件夹结构...")
        structure = self._detect_structure()
        
        if structure['type'] is None:
            raise ValueError(f"无法识别输入文件夹结构: {self.input_dir}")
        
        print(f"   检测到结构类型: {structure['type']}")
        print(f"   原始训练集正常样本: {len(structure['train_normal'])} 张")
        print(f"   原始测试集正常样本: {len(structure['test_normal'])} 张")
        print(f"   原始测试集异常样本: {len(structure['test_defect'])} 张")
        
        # 2. 限制训练集样本数量
        structure['train_normal'] = self._limit_train_samples(structure['train_normal'])
        
        # 3. 确定异常类型
        if defect_types:
            structure['defect_types'] = defect_types
        elif not structure['defect_types']:
            structure['defect_types'] = ['defect']
        
        print(f"   异常类型: {structure['defect_types']}")
        
        # 4. 创建输出目录结构
        print("\n📂 创建 MVTec AD 目录结构...")
        (self.output_dir / 'train' / 'good').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'test' / 'good').mkdir(parents=True, exist_ok=True)
        for defect in structure['defect_types']:
            (self.output_dir / 'test' / defect).mkdir(parents=True, exist_ok=True)
            (self.output_dir / 'ground_truth' / defect).mkdir(parents=True, exist_ok=True)
        
        # 5. 复制训练集
        print("\n📥 处理训练集...")
        self.stats['train_normal'] = self._copy_images(
            structure['train_normal'],
            self.output_dir / 'train' / 'good',
            "训练集"
        )
        
        # 6. 复制测试集
        print("\n📥 处理测试集...")
        self.stats['test_normal'] = self._copy_images(
            structure['test_normal'],
            self.output_dir / 'test' / 'good',
            "测试集-正常"
        )
        
        # 复制异常样本
        if structure['type'] == 'pre_split' and structure['defect_types']:
            # 已分割结构：按类别复制
            for defect_type in structure['defect_types']:
                src_dir = self.input_dir / 'test' / defect_type
                if src_dir.exists():
                    count = self._copy_images(
                        self._find_images(src_dir),
                        self.output_dir / 'test' / defect_type,
                        f"测试集-{defect_type}"
                    )
                    self.stats['test_defect'] += count
        else:
            # 其他结构：统一放入第一个异常类别
            self.stats['test_defect'] = self._copy_images(
                structure['test_defect'],
                self.output_dir / 'test' / structure['defect_types'][0],
                "测试集-异常"
            )
        
        # 7. 处理掩膜
        print("\n🎭 处理像素级标注掩膜...")
        for defect_type in structure['defect_types']:
            test_defect_dir = self.output_dir / 'test' / defect_type
            mask_dir = self.output_dir / 'ground_truth' / defect_type
            if test_defect_dir.exists():
                self._generate_dummy_masks(test_defect_dir, mask_dir)
        
        # 8. 输出统计信息
        print("\n" + "="*70)
        print("✅ 数据格式转换完成!")
        print("="*70)
        print(f"\n📊 最终数据集统计:")
        print(f"   训练集（正常）: {self.stats['train_normal']} 张")
        if self.stats['truncated'] > 0:
            print(f"   ⚠️  被截断的样本: {self.stats['truncated']} 张")
        print(f"   测试集（正常）: {self.stats['test_normal']} 张")
        print(f"   测试集（异常）: {self.stats['test_defect']} 张")
        print(f"\n📂 输出目录: {self.output_dir}")
        print("\n💡 提示:")
        print("   - 请检查 ground_truth/ 下的掩膜是否为真实标注")
        print("   - 训练时模型将只使用 train/good/ 下的样本")
        
        return self.stats


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description='数据集处理模块 - 将原始图片转换为 MVTec AD 格式'
    )
    parser.add_argument('--input_dir', '-i', type=str, required=True,
                        help='原始数据文件夹路径')
    parser.add_argument('--output_dir', '-o', type=str, required=True,
                        help='输出目录（MVTec AD 格式）')
    parser.add_argument('--max_train', type=int, default=150,
                        help='训练集正常样本最大数量（默认150）')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    
    args = parser.parse_args()
    
    # 执行转换
    formatter = MVTecFormatter(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        max_train_samples=args.max_train,
        seed=args.seed
    )
    formatter.convert()


if __name__ == '__main__':
    main()
