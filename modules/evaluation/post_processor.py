"""
============================================================================
异常热力图后处理模块 (Anomaly Map Post-Processor)
============================================================================

功能: 对模型输出的异常热力图进行后处理，提升像素级定位指标（PRO/Pixel AUROC）。

后处理策略（按应用顺序）:
    1. 高斯平滑 — 抑制热力图噪声，减少孤立假阳性像素
    2. 形态学闭运算 — 填充缺陷区域内部的小空洞，提升区域连续性
    3. 小区域过滤 — 移除面积过小的连通分量（噪声假阳性）
    4. 引导滤波（可选）— 用原图边缘引导热力图平滑，保边去噪

原理:
    PRO 指标对区域连续性敏感——如果模型对同一个缺陷区域输出碎片化的高分数
    （中间有低分空洞），阈值化后会出现多个不连通的预测区域，降低 Overlap。
    后处理通过平滑+闭运算使预测区域更连续、更接近真实标注的连通区域形态。

使用示例:
    from modules.evaluation.post_processor import AnomalyMapProcessor

    processor = AnomalyMapProcessor(
        gaussian_sigma=1.0,
        closing_radius=3,
        min_area=10,
    )
    processed_maps = processor.process(anomaly_maps)  # (N, H, W) ndarray
============================================================================
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import cv2


@dataclass
class PostProcessConfig:
    """
    后处理参数配置

    Args:
        gaussian_sigma: 高斯模糊 sigma 值（0=禁用）
        closing_radius: 形态学闭运算核半径（0=禁用）
        min_area: 最小连通区域面积（像素），低于此值的区域视为噪声
        dilate_radius: 膨胀核半径（0=禁用），适度膨胀可提升 PRO Overlap
        apply_per_image_norm: 是否逐图独立归一化（默认 True，与 PRO 计算一致）
    """
    gaussian_sigma: float = 1.0
    closing_radius: int = 3
    min_area: int = 10
    dilate_radius: int = 0
    apply_per_image_norm: bool = True

    def to_dict(self) -> Dict:
        return {
            'gaussian_sigma': self.gaussian_sigma,
            'closing_radius': self.closing_radius,
            'min_area': self.min_area,
            'dilate_radius': self.dilate_radius,
            'apply_per_image_norm': self.apply_per_image_norm,
        }

    def label(self) -> str:
        """生成人类可读的配置标签"""
        parts = []
        if self.gaussian_sigma > 0:
            parts.append(f"Gσ={self.gaussian_sigma:.1f}")
        if self.closing_radius > 0:
            parts.append(f"C_r={self.closing_radius}")
        if self.min_area > 0:
            parts.append(f"minA={self.min_area}")
        if self.dilate_radius > 0:
            parts.append(f"D_r={self.dilate_radius}")
        return "_".join(parts) if parts else "raw"


# 预设配置方案（覆盖常见工业场景）
PRESET_CONFIGS: Dict[str, PostProcessConfig] = {
    'light': PostProcessConfig(
        gaussian_sigma=0.5,
        closing_radius=2,
        min_area=5,
        dilate_radius=0,
    ),
    'medium': PostProcessConfig(
        gaussian_sigma=1.0,
        closing_radius=3,
        min_area=10,
        dilate_radius=2,
    ),
    'strong': PostProcessConfig(
        gaussian_sigma=2.0,
        closing_radius=5,
        min_area=20,
        dilate_radius=3,
    ),
    'aggressive': PostProcessConfig(
        gaussian_sigma=3.0,
        closing_radius=7,
        min_area=30,
        dilate_radius=5,
    ),
    'smooth_only': PostProcessConfig(
        gaussian_sigma=1.5,
        closing_radius=0,
        min_area=0,
        dilate_radius=0,
    ),
    'morph_only': PostProcessConfig(
        gaussian_sigma=0.0,
        closing_radius=3,
        min_area=10,
        dilate_radius=2,
    ),
    'off': PostProcessConfig(
        gaussian_sigma=0.0,
        closing_radius=0,
        min_area=0,
        dilate_radius=0,
    ),
}


class AnomalyMapProcessor:
    """
    异常热力图后处理器

    对模型输出的 anomaly maps 应用平滑和形态学后处理，
    提升像素级定位指标的连续性。
    """

    def __init__(self, config: PostProcessConfig = None):
        """
        初始化后处理器

        Args:
            config: 后处理参数配置（默认使用 'medium' 预设）
        """
        self.config = config or PRESET_CONFIGS['medium']

    def process(self, anomaly_maps: np.ndarray) -> np.ndarray:
        """
        对一批异常热力图执行后处理

        Args:
            anomaly_maps: 异常热力图 (N, H, W) float32，值范围任意

        Returns:
            处理后的热力图 (N, H, W) float32
        """
        maps = np.asarray(anomaly_maps, dtype=np.float32)
        processed = np.zeros_like(maps)

        for i in range(len(maps)):
            processed[i] = self._process_single(maps[i])

        return processed

    def _process_single(self, anomaly_map: np.ndarray) -> np.ndarray:
        """
        对单张热力图执行后处理流水线

        Args:
            anomaly_map: (H, W) float32 异常热力图

        Returns:
            处理后的热力图 (H, W) float32
        """
        cfg = self.config
        h, w = anomaly_map.shape[:2]
        result = anomaly_map.copy()

        # 步骤 1: 逐图归一化到 [0, 1]（与 PRO 计算中的归一化一致）
        if cfg.apply_per_image_norm:
            min_v = float(result.min())
            max_v = float(result.max())
            if max_v - min_v > 1e-8:
                result = (result - min_v) / (max_v - min_v)
            else:
                result = np.zeros_like(result)

        # 步骤 2: 高斯平滑 — 抑制热力图噪声
        if cfg.gaussian_sigma > 0:
            # ksize 自动根据 sigma 计算
            ksize = max(3, int(4 * cfg.gaussian_sigma + 1) | 1)
            result = cv2.GaussianBlur(result, (ksize, ksize), cfg.gaussian_sigma)

        # 步骤 3: 形态学闭运算 — 先膨胀后腐蚀，填充缺陷内部小空洞
        # 注意：此操作作用于连续值热力图，而非二值图
        if cfg.closing_radius > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * cfg.closing_radius + 1, 2 * cfg.closing_radius + 1),
            )
            # 先膨胀（让高值区域扩展相邻低值区域）
            result = cv2.dilate(result, kernel, iterations=1)
            # 再腐蚀（收缩回原边界，但空洞已被填充）
            result = cv2.erode(result, kernel, iterations=1)

        # 步骤 4: 膨胀 — 适度扩展高值区域边界，提升 Overlap
        if cfg.dilate_radius > 0:
            kernel_d = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * cfg.dilate_radius + 1, 2 * cfg.dilate_radius + 1),
            )
            result = cv2.dilate(result, kernel_d, iterations=1)

        # 步骤 5: 小区域过滤 — 移除面积过小的连通分量
        if cfg.min_area > 0:
            # 对每个阈值分别处理太昂贵，这里采用自适应策略：
            # 找到面积小于 min_area 的局部极大值区域，将其值设为其邻域中值
            # 使用连通分量分析
            binary_high = (result > result.mean() + result.std()).astype(np.uint8)
            # 确保单通道 uint8
            if binary_high.ndim == 3:
                binary_high = binary_high[:, :, 0]
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                binary_high, connectivity=8
            )
            small_mask = np.zeros((h, w), dtype=bool)
            for label_id in range(1, num_labels):
                area = stats[label_id, cv2.CC_STAT_AREA]
                if area < cfg.min_area:
                    small_mask |= (labels == label_id)

            if small_mask.any():
                # 用局部中值替换小区域
                from scipy.ndimage import median_filter
                replacement = median_filter(result, size=7)
                result = result.copy()
                result[small_mask] = replacement[small_mask]

        # 确保输出值在有效范围内
        result = np.clip(result, 0.0, 1.0)

        return result.astype(np.float32)


def process_anomaly_maps(
    anomaly_maps: np.ndarray,
    gaussian_sigma: float = 1.0,
    closing_radius: int = 3,
    min_area: int = 10,
    dilate_radius: int = 2,
) -> np.ndarray:
    """
    便捷函数：对异常热力图应用后处理

    Args:
        anomaly_maps: (N, H, W) float32
        gaussian_sigma: 高斯模糊 sigma
        closing_radius: 形态学闭运算核半径
        min_area: 最小连通区域面积
        dilate_radius: 膨胀半径

    Returns:
        处理后的 (N, H, W) float32
    """
    config = PostProcessConfig(
        gaussian_sigma=gaussian_sigma,
        closing_radius=closing_radius,
        min_area=min_area,
        dilate_radius=dilate_radius,
    )
    processor = AnomalyMapProcessor(config)
    return processor.process(anomaly_maps)


def grid_search_configs() -> List[PostProcessConfig]:
    """
    生成网格搜索配置列表

    在合理范围内生成参数组合，用于寻找最优后处理配置。

    Returns:
        PostProcessConfig 列表
    """
    configs = []
    # 核心参数网格
    for gs in [0.0, 0.5, 1.0, 1.5, 2.0]:
        for cr in [0, 2, 3, 5]:
            for ma in [0, 5, 10, 20]:
                for dr in [0, 1, 2, 3]:
                    # 跳过全零配置（= raw）
                    if gs == 0.0 and cr == 0 and ma == 0 and dr == 0:
                        continue
                    # 跳过极端组合（时间换空间，减少搜索量）
                    if gs > 1.5 and cr > 3:
                        continue  # 过度平滑可能导致关键信息丢失
                    if dr > 2 and gs > 1.5:
                        continue
                    configs.append(PostProcessConfig(
                        gaussian_sigma=gs,
                        closing_radius=cr,
                        min_area=ma,
                        dilate_radius=dr,
                    ))
    return configs
