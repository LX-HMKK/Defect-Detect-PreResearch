"""
配置管理模块

提供统一的配置管理功能
"""

from .manager import ConfigManager, get_config, get, get_threshold, get_model_config, get_data_config, reset_config

# 注意: 此处的 get() 函数名会遮蔽 Python 内置的 dict.get()。
# 通过 Python 的 LEGB 解析规则正确处理 — 调用此 get() 时传入字符串键；
# 字典的 .get() 方法调用不受影响。

__all__ = [
    'ConfigManager',
    'get_config',
    'get',
    'get_threshold',
    'get_model_config',
    'get_data_config',
    'reset_config',
]
