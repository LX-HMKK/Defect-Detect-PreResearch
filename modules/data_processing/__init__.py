"""
数据集处理模块

将原始图片转换为 MVTec AD 标准格式。
"""

from .dataset_formatter import MVTecFormatter

__all__ = ['MVTecFormatter']
