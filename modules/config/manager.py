"""
================================================================================
配置管理模块 (Configuration Manager)
================================================================================

功能: 统一管理应用配置，从 YAML 文件读取，提供配置访问接口

使用示例:
    from modules.config.manager import ConfigManager
    
    # 获取配置实例
    config = ConfigManager()
    
    # 访问配置
    image_size = config.get('data.image_size')  # [256, 256]
    threshold = config.get_threshold('patchcore', 'bottle')  # 从训练结果读取
    
    # 更新配置
    config.set('data.batch_size', 64)
================================================================================
"""

import os
import yaml
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union
from functools import lru_cache


class ConfigManager:
    """
    配置管理器
    
    管理应用的 YAML 配置文件，支持:
    - 读取 YAML 配置
    - 层级配置访问 (如 'data.image_size')
    - 从训练结果自动读取最优阈值
    - 配置热更新
    """
    
    def __init__(self, config_path: str = None):
        """
        初始化配置管理器
        
        Args:
            config_path: 主配置文件路径（默认从 config/config.yaml 加载）
        """
        # 默认路径：config/config.yaml
        if config_path is None:
            # 相对于项目根目录
            config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = {}
        self._load_config()
    
    def _load_config(self) -> None:
        """加载配置文件"""
        if not self.config_path.exists():
            print(f"[WARN] 配置文件不存在: {self.config_path}，使用默认配置")
            self._config = self._get_default_config()
            return
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f)
            print(f"[OK] 配置文件已加载: {self.config_path}")
        except Exception as e:
            print(f"[WARN] 加载配置文件失败: {e}，使用默认配置")
            self._config = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        获取默认配置
        
        Returns:
            默认配置字典
        """
        return {
            'data': {
                'image_size': [256, 256],
                'train_batch_size': 32,
                'eval_batch_size': 32,
                'num_workers': 0,
            },
            'threshold': {
                'default': 0.5,
                'dataset_defaults': {
                    'bottle': 0.5,
                    'carpet': 0.5,
                }
            },
            'training': {
                'epochs': {
                    'patchcore': 1,
                    'fre': 50,
                    'draem': 200,
                },
                'seed': 42,
                'accelerator': 'auto',
            },
            'paths': {
                'data_root': './data',
                'results_root': './results',
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值
        
        支持层级访问，如:
        - 'data.image_size' -> [256, 256]
        - 'models.patchcore.backbone' -> 'wide_resnet50_2'
        
        Args:
            key: 配置键（支持点号分隔的层级访问）
            default: 默认值
            
        Returns:
            配置值或默认值
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """
        设置配置值
        
        Args:
            key: 配置键
            value: 配置值
        """
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def get_threshold(self, model: str, dataset: str) -> float:
        """
        获取数据集的最优阈值
        
        优先级:
        1. 从训练结果文件读取
        2. 配置文件中的 dataset_defaults
        3. 默认阈值 0.5
        
        Args:
            model: 模型名称 (patchcore/fre/draem)
            dataset: 数据集名称
            
        Returns:
            最优阈值
        """
        # 1. 尝试从训练结果文件读取
        result_file_template = self.get(
            'threshold.result_file_template',
            './results/comparison/{model}_{dataset}_results.json'
        )
        result_file = Path(result_file_template.format(model=model, dataset=dataset))
        
        if result_file.exists():
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    threshold_key = self.get('threshold.threshold_key', 'optimal_threshold')
                    threshold = data.get('metrics', {}).get(threshold_key)
                    if threshold is not None:
                        return float(threshold)
            except Exception:
                pass
        
        # 2. 从配置读取数据集默认值
        dataset_defaults = self.get('threshold.dataset_defaults', {})
        if dataset in dataset_defaults:
            return float(dataset_defaults[dataset])
        
        # 3. 返回默认阈值
        return self.get('threshold.default', 0.5)
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """
        获取模型配置
        
        Args:
            model_name: 模型名称 (patchcore/fre/draem)
            
        Returns:
            模型配置字典
        """
        return self.get(f'models.{model_name}', {})
    
    def get_data_config(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        获取数据配置
        
        Args:
            model_name: 模型名称（用于获取特定模型的 batch size）
            
        Returns:
            数据配置字典
        """
        config = {
            'image_size': self.get('data.image_size', [256, 256]),
            'train_batch_size': self.get('data.train_batch_size', 32),
            'eval_batch_size': self.get('data.eval_batch_size', 32),
            'num_workers': self.get('data.num_workers', 0),
        }
        
        # DRAEM 使用特殊 batch size
        if model_name == 'draem':
            config['train_batch_size'] = self.get('data.draem_batch_size', 4)
            config['eval_batch_size'] = self.get('data.draem_batch_size', 4)
        
        return config
    
    def get_epochs(self, model_name: str) -> int:
        """
        获取模型的默认训练 epoch
        
        Args:
            model_name: 模型名称
            
        Returns:
            训练 epoch 数
        """
        return self.get(f'training.epochs.{model_name}', 50)
    
    def reload(self) -> None:
        """重新加载配置文件"""
        self._load_config()
        print("[OK] 配置已重新加载")
    
    def save(self, path: Optional[str] = None) -> None:
        """
        保存配置到文件
        
        Args:
            path: 保存路径（默认覆盖原文件）
        """
        save_path = Path(path) if path else self.config_path
        
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                yaml.dump(self._config, f, allow_unicode=True, sort_keys=False)
            print(f"[OK] 配置已保存: {save_path}")
        except Exception as e:
            print(f"[FAIL] 保存配置失败: {e}")


# 全局配置实例（单例模式）
_config_instance: Optional[ConfigManager] = None


def get_config(config_path: str = None) -> ConfigManager:
    """
    获取全局配置实例
    
    Args:
        config_path: 配置文件路径（默认从 config/config.yaml 加载）
        
    Returns:
        ConfigManager 实例
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = ConfigManager(config_path)  # None triggers default path
    return _config_instance


def reset_config() -> None:
    """重置全局配置实例"""
    global _config_instance
    _config_instance = None


# 便捷函数
def get(key: str, default: Any = None) -> Any:
    """便捷获取配置值"""
    return get_config().get(key, default)


def get_threshold(model: str, dataset: str) -> float:
    """便捷获取阈值"""
    return get_config().get_threshold(model, dataset)


def get_model_config(model_name: str) -> Dict[str, Any]:
    """便捷获取模型配置"""
    return get_config().get_model_config(model_name)


def get_data_config(model_name: Optional[str] = None) -> Dict[str, Any]:
    """便捷获取数据配置"""
    return get_config().get_data_config(model_name)
