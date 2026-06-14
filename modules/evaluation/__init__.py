"""
指标评测模块

提供论文/综述要求的 4 个硬性指标计算功能：
图像级 AUROC、AUPR，像素级 Pixel AUROC、PRO。
以及异常热力图后处理工具。
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

# 后处理模块依赖 cv2，CI 环境可能不提供，做惰性导入
try:
    from .post_processor import (  # noqa: F401
        AnomalyMapProcessor,
        PostProcessConfig,
        PRESET_CONFIGS,
        process_anomaly_maps,
    )
    __all__.extend([
        'AnomalyMapProcessor',
        'PostProcessConfig',
        'PRESET_CONFIGS',
        'process_anomaly_maps',
    ])
except ImportError:
    pass
