"""
指标评测模块

提供论文/综述要求的 4 个硬性指标计算功能：
图像级 AUROC、AUPR，像素级 Pixel AUROC、PRO。
"""

from .metrics import (
    MetricsEvaluator,
    AnomalyMetrics,
    load_and_evaluate,
)

__all__ = [
    'MetricsEvaluator',
    'AnomalyMetrics',
    'load_and_evaluate',
]
