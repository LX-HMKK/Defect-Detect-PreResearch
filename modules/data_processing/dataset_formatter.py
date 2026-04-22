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

    def _resize_with_letterbox(
        self,
        image: np.ndarray,
        target_size: Tuple[int, int],
        interpolation: int,
        border_value: int | Tuple[int, int, int] = 0,
    ) -> np.ndarray:
        """将图像缩放并居中填充到目标尺寸。"""
        target_h, target_w = target_size
        h, w = image.shape[:2]

        scale = min(target_h / h, target_w / w)
        new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
        resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

        if image.ndim == 2:
            canvas = np.full((target_h, target_w), border_value, dtype=image.dtype)
        else:
            canvas = np.full((target_h, target_w, image.shape[2]), border_value, dtype=image.dtype)

        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        return canvas
    
    def _find_images(self, directory: Path) -> List[Path]:
        """
        查找目录下的所有图片文件
        
        Args:
            directory: 目标目录
        
        Returns:
            List[Path]: 图片文件路径列表
        """
        unique_images: Dict[str, Path] = {}
        for ext in self.image_extensions:
            for image_path in directory.glob(f'*{ext}'):
                unique_images[str(image_path.resolve()).lower()] = image_path
            for image_path in directory.glob(f'*{ext.upper()}'):
                unique_images[str(image_path.resolve()).lower()] = image_path
        return sorted(unique_images.values())
    
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
            'defect_types': [],
            'test_defect_by_type': {},
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
                        defect_images = self._find_images(subdir)
                        structure['defect_types'].append(subdir.name)
                        structure['test_defect'].extend(defect_images)
                        structure['test_defect_by_type'][subdir.name] = defect_images
        
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
                    defect_images = self._find_images(subdir)
                    structure['defect_types'].append(subdir.name)
                    structure['test_defect'].extend(defect_images)
                    structure['test_defect_by_type'][subdir.name] = defect_images
        
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
            print(f"[WARN] 训练集样本过多 ({len(images)} 张)，将随机采样保留 {self.max_train_samples} 张")
            return random.sample(images, self.max_train_samples)
        return images
    
    def _copy_images(self, images: List[Path], dst_dir: Path, desc: str = "复制", target_size: Tuple[int, int] = (256, 256)) -> int:
        """
        复制图片到目标目录（并统一尺寸）
        
        Args:
            images: 源图片路径列表
            dst_dir: 目标目录
            desc: 进度条描述
            target_size: 目标尺寸 (height, width)
        
        Returns:
            int: 成功复制的数量
        """
        dst_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        target_h, target_w = target_size
        
        for idx, src_path in enumerate(tqdm(images, desc=desc)):
            dst_name = f"{idx:03d}{src_path.suffix.lower()}"
            dst_path = dst_dir / dst_name
            try:
                img = cv2.imread(str(src_path))
                if img is not None:
                    canvas = self._resize_with_letterbox(
                        img,
                        (target_h, target_w),
                        interpolation=cv2.INTER_AREA,
                        border_value=(0, 0, 0),
                    )
                    cv2.imwrite(str(dst_path), canvas)
                    count += 1
                else:
                    shutil.copy2(src_path, dst_path)
                    count += 1
            except Exception as e:
                print(f"[FAIL] 处理失败 {src_path}: {e}")
        
        return count

    def _find_mask_for_image(self, mask_dir: Path, image_stem: str) -> Optional[Path]:
        """根据图片 stem 查找对应 mask 文件。"""
        for ext in self.image_extensions:
            candidate = mask_dir / f"{image_stem}_mask{ext.lower()}"
            if candidate.exists():
                return candidate
            candidate_upper = mask_dir / f"{image_stem}_mask{ext.upper()}"
            if candidate_upper.exists():
                return candidate_upper
        for ext in self.image_extensions:
            candidate = mask_dir / f"{image_stem}{ext.lower()}"
            if candidate.exists():
                return candidate
            candidate_upper = mask_dir / f"{image_stem}{ext.upper()}"
            if candidate_upper.exists():
                return candidate_upper
        return None

    def _copy_defect_images_with_masks(
        self,
        src_images: List[Path],
        src_mask_dir: Path,
        dst_test_dir: Path,
        dst_mask_dir: Path,
        desc: str,
        target_size: Tuple[int, int] = (256, 256),
    ) -> int:
        """复制异常样本及其真实掩膜；掩膜与图像使用相同 letterbox 对齐。"""
        if not src_mask_dir.exists():
            raise ValueError(f"缺少真实掩膜目录: {src_mask_dir}")

        dst_test_dir.mkdir(parents=True, exist_ok=True)
        dst_mask_dir.mkdir(parents=True, exist_ok=True)
        copied = 0

        for idx, src_image in enumerate(tqdm(src_images, desc=desc)):
            src_mask = self._find_mask_for_image(src_mask_dir, src_image.stem)
            if src_mask is None:
                raise ValueError(f"缺少真实掩膜: {src_mask_dir} 中未找到 {src_image.stem} 的 mask")

            image = cv2.imread(str(src_image))
            if image is None:
                raise ValueError(f"无法读取异常图像: {src_image}")

            mask = cv2.imread(str(src_mask), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"无法读取真实掩膜: {src_mask}")

            dst_image_name = f"{idx:03d}{src_image.suffix.lower()}"
            dst_image_path = dst_test_dir / dst_image_name
            dst_mask_path = dst_mask_dir / f"{Path(dst_image_name).stem}_mask.png"

            resized_image = self._resize_with_letterbox(
                image,
                target_size,
                interpolation=cv2.INTER_AREA,
                border_value=(0, 0, 0),
            )
            resized_mask = self._resize_with_letterbox(
                mask,
                target_size,
                interpolation=cv2.INTER_NEAREST,
                border_value=0,
            )

            cv2.imwrite(str(dst_image_path), resized_image)
            cv2.imwrite(str(dst_mask_path), resized_mask)
            copied += 1

        return copied
    
    def convert(self, defect_types: Optional[List[str]] = None) -> Dict:
        """
        执行数据格式转换
        
        Args:
            defect_types: 指定异常类型名称列表（可选）
        
        Returns:
            Dict: 转换统计信息
        """
        print("="*70)
        print("[WAIT] 数据集处理模块 (Data Processing Module)")
        print("="*70)
        print(f"\n[DIR] 输入目录: {self.input_dir}")
        print(f"[DIR] 输出目录: {self.output_dir}")
        print(f"[STAT] 训练集上限: {self.max_train_samples} 张")
        
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
            if structure['test_defect'] and not any(
                defect in structure['test_defect_by_type'] for defect in defect_types
            ):
                structure['test_defect_by_type'] = {defect_types[0]: structure['test_defect']}
        elif not structure['defect_types']:
            structure['defect_types'] = ['defect']
        
        print(f"   异常类型: {structure['defect_types']}")
        
        # 4. 创建输出目录结构
        print("\n[DIR] 创建 MVTec AD 目录结构...")
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
        
        # 复制异常样本 + 真实掩膜（严格模式：缺掩膜即失败）
        print("\n🎭 处理异常样本与像素级标注掩膜...")
        for defect_type in structure['defect_types']:
            src_images = structure['test_defect_by_type'].get(defect_type, [])
            if not src_images:
                continue
            src_mask_dir = self.input_dir / 'ground_truth' / defect_type
            count = self._copy_defect_images_with_masks(
                src_images=src_images,
                src_mask_dir=src_mask_dir,
                dst_test_dir=self.output_dir / 'test' / defect_type,
                dst_mask_dir=self.output_dir / 'ground_truth' / defect_type,
                desc=f"测试集-{defect_type}",
            )
            self.stats['test_defect'] += count
        
        # 8. 输出统计信息
        print("\n" + "="*70)
        print("[OK] 数据格式转换完成!")
        print("="*70)
        print(f"\n[STAT] 最终数据集统计:")
        print(f"   训练集（正常）: {self.stats['train_normal']} 张")
        if self.stats['truncated'] > 0:
            print(f"   [WARN] 被截断的样本: {self.stats['truncated']} 张")
        print(f"   测试集（正常）: {self.stats['test_normal']} 张")
        print(f"   测试集（异常）: {self.stats['test_defect']} 张")
        print(f"\n[DIR] 输出目录: {self.output_dir}")
        print("\n💡 提示:")
        print("   - 已严格校验 ground_truth/ 下存在真实掩膜")
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
