"""
================================================================================
模块 2: 核心算法复现模块 (Algorithm Implementation Module) - Anomalib 2.x
================================================================================

功能: 调用 anomalib 2.x 训练和测试 4 种异常检测算法

复现的算法（4个）:
    1. PatchCore (基于特征建模) - 工业界效果最好
    2. PaDiM (基于特征建模) - 概率建模，无需训练
    3. FRE (基于特征重构) - 重构法改进版
    4. DRAEM (基于自监督学习) - 无需真实异常样本训练

约束条件:
    - 只用正常样本训练（无监督设定）
    - 基于 anomalib 库，不手写神经网络底层

使用示例:
    from modules.algorithm.trainer import AnomalyDetectionTrainer

    trainer = AnomalyDetectionTrainer(
        model_name='patchcore',
        data_path='./data',
        category='bottle'
    )
    results = trainer.train()  # 训练
    metrics = trainer.evaluate()  # 测试并输出4个硬性指标
================================================================================
"""

import os
import shutil
import sys
import json
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any, Mapping

import torch
import numpy as np
import cv2  # Import cv2 first to avoid DLL loading issues with anomalib
import pandas as pd
from tqdm import tqdm

# anomalib 2.x 导入
from anomalib.data import MVTec, Folder
from anomalib.engine import Engine
from anomalib.models import (
    Patchcore,
    Draem,
    Fre,
    Padim,
)
from anomalib.metrics import Evaluator, AUPR, PRO, AUROC, F1Score
from pytorch_lightning.callbacks import Callback

# monkey-patch 兼容层 — anomalib 2.3.0 与 PyTorch Lightning 1.9.5
from . import _anomalib_compat

# 配置管理
from modules._runtime import resolve_project_path
from modules.config import get_model_config, get_data_config, get

# 忽略警告
warnings.filterwarnings('ignore')

# ================================================================================
# 预训练模型缓存配置
# ================================================================================

# 预训练权重缓存目录
PRETRAINED_CACHE_DIR = resolve_project_path(
    get('paths.pre_trained_dir', './.cache/pretrained')
)
PRETRAINED_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 设置 Torch Hub 缓存目录
TORCH_HUB_CACHE_DIR = resolve_project_path(
    get('paths.cache.torch_hub', './.cache/pretrained/torch_hub')
)
TORCH_HUB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
torch.hub.set_dir(str(TORCH_HUB_CACHE_DIR))

# 设置 HuggingFace 缓存目录
HF_CACHE_DIR = resolve_project_path(
    get('paths.cache.huggingface', './.cache/pretrained/huggingface')
)
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(HF_CACHE_DIR)
os.environ["HF_HUB_CACHE"] = str(HF_CACHE_DIR / "hub")


# ================================================================================
# 支持的模型配置（3个算法）
# ================================================================================

SUPPORTED_MODELS = ['fre', 'patchcore', 'draem', 'padim']

MODEL_INFO = {
    'fre': {
        '方向': '基于特征重构',
        '原理': '预训练CNN提取特征，线性自编码器重构特征，重构误差作为异常分数',
        '特点': '重构法改进版，支持像素级定位',
        '复现难度': '* (简单)',
        '训练时间': '~5分钟',
    },
    'patchcore': {
        '方向': '基于特征建模',
        '原理': '预训练CNN提取局部特征，构建记忆库，最近邻搜索检测异常',
        '特点': '工业界效果最好，无需训练，推理最快',
        '复现难度': '* (easiest)',
        '训练时间': '~1分钟 (仅构建记忆库)',
    },
    'draem': {
        '方向': '基于自监督学习',
        '原理': '生成合成异常样本，训练判别网络区分正常/异常区域',
        '特点': '无需真实异常样本即可训练，对小缺陷敏感',
        '复现难度': '*** (hard)',
        '训练时间': '~30分钟 (200 epochs)',
    },
    'padim': {
        '方向': '基于特征建模',
        '原理': '对每个patch位置建立多元高斯分布，通过马氏距离度量异常',
        '特点': '无需训练，概率建模比记忆库更轻量',
        '复现难度': '* (简单)',
        '训练时间': '~1分钟 (仅构建高斯模型)',
    }
}


def find_latest_checkpoint(
    output_dir: str | Path,
    model_name: str,
    category: Optional[str] = None,
    source: Optional[str] = None,
) -> Optional[Path]:
    """
    在输出目录中查找指定模型（可选指定类别、来源）的最新 checkpoint。

    Args:
        output_dir: 结果根目录（通常为 ./results）
        model_name: 模型名称（fre/patchcore/draem）
        category: 数据类别（可选）
        source: 数据集来源子目录（default / user，可选）。
                传入后只在 results/{model}/{ModelName}/{source}/{category} 中查找。

    Returns:
        最新 checkpoint 路径；未找到则返回 None。
    """
    model_root = Path(output_dir) / model_name
    if not model_root.exists():
        return None

    if category:
        if source:
            # 精确到 default/user 子目录
            patterns: List[str] = [
                f"**/{source}/{category}/**/weights/lightning/model.ckpt",
                f"**/{source}/{category}/**/model.ckpt",
                f"**/{source}/{category}/**/*.ckpt",
            ]
        else:
            # 兼容旧结构：未指定 source 时全目录搜索（包含历史 MVTec/ 路径）
            patterns = [
                f"**/{category}/**/weights/lightning/model.ckpt",
                f"**/{category}/**/model.ckpt",
                f"**/{category}/**/*.ckpt",
                f"**/MVTec/{category}/**/weights/lightning/model.ckpt",
                f"**/MVTec/{category}/**/model.ckpt",
                f"**/MVTec/{category}/**/*.ckpt",
            ]
    else:
        patterns = [
            "**/weights/lightning/model.ckpt",
            "**/model.ckpt",
            "**/*.ckpt",
        ]

    candidates: List[Path] = []
    for pattern in patterns:
        for path in model_root.glob(pattern):
            if path.is_file():
                candidates.append(path)

    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _get_required_data_config(config: Optional[Dict[str, Any]], 
                              key: str, 
                              model_name: str) -> Any:
    """
    从配置中获取必需的数据参数，如果缺失则报错
    
    Args:
        config: 传入的配置
        key: 配置键名
        model_name: 模型名称（用于错误信息）
        
    Returns:
        配置值
        
    Raises:
        ValueError: 配置缺失时抛出
    """
    # 1. 尝试从传入的 config 读取（Anomalib 2.x 格式）
    if config:
        if 'data' in config and 'init_args' in config['data']:
            data_config = config['data']['init_args']
            if key in data_config:
                return data_config[key]
        elif key in config:
            return config[key]
    
    # 2. 尝试从 configs/config.yaml 读取
    value = get(f'data.{key}', None)
    if value is not None:
        return value
    
    # 3. 如果仍然缺失，报错
    raise ValueError(
        f"数据配置缺失: 请在 configs/{model_name}.yaml 的 data.init_args 部分或 "
        f"configs/config.yaml 的 data 部分设置 {key}"
    )


def get_datamodule_from_config(
    data_path: str,
    category: str,
    model_name: str,
    config: Optional[Dict[str, Any]] = None
) -> Union[MVTec, Folder]:
    """
    根据配置创建数据模块 - 严格从 YAML 读取，缺配置直接报错
    
    Args:
        data_path: 数据目录路径
        category: 类别名称
        model_name: 模型名称
        config: 额外配置参数（可以是完整 YAML config 或 data.init_args 部分）
    
    Returns:
        MVTec 或 Folder 数据模块
        
    Raises:
        ValueError: 配置缺失时抛出
    """
    data_path = Path(data_path)
    
    # 严格从配置读取，缺失则报错
    train_batch_size = _get_required_data_config(config, 'train_batch_size', model_name)
    eval_batch_size = _get_required_data_config(config, 'eval_batch_size', model_name)
    num_workers = _get_required_data_config(config, 'num_workers', model_name)
    
    # 检测数据集格式
    category_path = data_path / category

    # 如果是 MVTec AD 格式（有 train、test、ground_truth 目录）
    if (
        (category_path / 'train').exists()
        and (category_path / 'test').exists()
        and (category_path / 'ground_truth').exists()
    ):
        return MVTec(
            root=str(data_path),
            category=category,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
        )
    else:
        # 使用 Folder 格式
        # 上传数据集仅含正常样本：train/good 为训练集，test/good 为测试集。
        # 不设置 abnormal_dir，避免 anomalib 将 test/good 同时当作异常样本。
        return Folder(
            name=category,
            root=str(category_path),
            normal_dir='train/good',
            normal_test_dir='test/good',
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
        )


def _require_config(config: Optional[Dict[str, Any]], model_defaults: Dict[str, Any], 
                    key: str, model_name: str) -> Any:
    """
    从配置中获取必需参数，如果缺失则报错
    
    Args:
        config: 传入的配置
        model_defaults: 默认配置
        key: 配置键名
        model_name: 模型名称（用于错误信息）
        
    Returns:
        配置值
        
    Raises:
        ValueError: 配置缺失时抛出
    """
    # 优先从传入的 config 读取
    if config and key in config:
        return config[key]
    
    # 其次从 model_defaults 读取
    if key in model_defaults:
        return model_defaults[key]
    
    # 如果都没有，报错
    raise ValueError(
        f"配置缺失: 请在 configs/config.yaml 的 models.{model_name} 部分或 "
        f"configs/{model_name}.yaml 的 model.init_args 部分设置 {key}"
    )


class _LearningRateSetter(Callback):
    """
    Lightning 回调：在训练开始时覆盖优化器学习率。

    用于 Training Studio 前端传入自定义 learning_rate 的场景。
    不 patch 模型的 configure_optimizers，避免 checkpoint 保存时无法 pickle。
    """

    def __init__(self, lr: float):
        self.lr = lr

    def on_train_start(self, trainer, pl_module):
        for optimizer in trainer.optimizers:
            for group in optimizer.param_groups:
                group['lr'] = self.lr
                group.setdefault('initial_lr', self.lr)
        print(f"   [INFO] 优化器学习率已设置为: {self.lr}")


def get_model_from_config(model_name: str, config: Optional[Dict[str, Any]] = None, enable_pixel_metrics: bool = True):
    """
    根据配置创建模型 - 严格从 YAML 读取，缺配置直接报错

    Args:
        model_name: 模型名称
        config: 模型配置参数（来自 YAML 的 model.init_args）
        enable_pixel_metrics: 是否启用像素级指标（上传数据集无 ground_truth 时关闭）

    Returns:
        模型实例

    Raises:
        ValueError: 配置缺失时抛出
    """
    # 创建 evaluator
    test_metrics = [
        AUROC(fields=["pred_score", "gt_label"]),
        AUPR(fields=["pred_score", "gt_label"]),
        F1Score(fields=["pred_label", "gt_label"]),
    ]
    if enable_pixel_metrics:
        test_metrics.extend([
            AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel_"),
            PRO(fields=["anomaly_map", "gt_mask"], prefix="pixel_"),
            F1Score(fields=["pred_mask", "gt_mask"], prefix="pixel_"),
        ])
    evaluator = Evaluator(test_metrics=test_metrics)
    
    # 从配置管理系统获取模型默认配置
    model_defaults = get_model_config(model_name)
    
    if model_name == 'patchcore':
        # 严格从配置读取，缺失则报错
        backbone = _require_config(config, model_defaults, 'backbone', 'patchcore')
        layers = _require_config(config, model_defaults, 'layers', 'patchcore')
        coreset_sampling_ratio = _require_config(config, model_defaults, 'coreset_sampling_ratio', 'patchcore')
        num_neighbors = _require_config(config, model_defaults, 'num_neighbors', 'patchcore')
        pre_trained = _require_config(config, model_defaults, 'pre_trained', 'patchcore')
        
        return Patchcore(
            backbone=backbone,
            layers=layers,
            coreset_sampling_ratio=coreset_sampling_ratio,
            num_neighbors=num_neighbors,
            pre_trained=pre_trained,
            evaluator=evaluator,
        )
    
    elif model_name == 'fre':
        # 严格从配置读取，缺失则报错
        backbone = _require_config(config, model_defaults, 'backbone', 'fre')
        layer = _require_config(config, model_defaults, 'layer', 'fre')
        pre_trained = _require_config(config, model_defaults, 'pre_trained', 'fre')
        pooling_kernel_size = _require_config(config, model_defaults, 'pooling_kernel_size', 'fre')
        input_dim = _require_config(config, model_defaults, 'input_dim', 'fre')
        latent_dim = _require_config(config, model_defaults, 'latent_dim', 'fre')
        
        return Fre(
            backbone=backbone,
            layer=layer,
            pre_trained=pre_trained,
            pooling_kernel_size=pooling_kernel_size,
            input_dim=input_dim,
            latent_dim=latent_dim,
            evaluator=evaluator,
        )
    
    elif model_name == 'draem':
        # DRAEM 使用默认参数，可选配置
        beta = [0.1, 1.0]
        enable_sspcab = False

        if config:
            beta = config.get('beta', beta)
            enable_sspcab = config.get('enable_sspcab', enable_sspcab)
        elif model_defaults:
            beta = model_defaults.get('beta', beta)
            enable_sspcab = model_defaults.get('enable_sspcab', enable_sspcab)

        return Draem(
            beta=tuple(beta) if isinstance(beta, list) else beta,
            enable_sspcab=enable_sspcab,
            evaluator=evaluator,
        )

    elif model_name == 'padim':
        backbone = _require_config(config, model_defaults, 'backbone', 'padim')
        layers = _require_config(config, model_defaults, 'layers', 'padim')
        pre_trained = _require_config(config, model_defaults, 'pre_trained', 'padim')

        return Padim(
            backbone=backbone,
            layers=layers,
            pre_trained=pre_trained,
            evaluator=evaluator,
        )

    else:
        raise ValueError(f"不支持的模型: {model_name}")


class AnomalyDetectionTrainer:
    """
    异常检测算法训练器 (Anomalib 2.x)
    
    封装 anomalib 2.x 的训练和评估流程，支持3种算法：
    - PatchCore: 特征建模方法
    - FRE: 特征重构方法
    - DRAEM: 自监督学习方法
    """
    
    def __init__(
        self,
        model_name: str,
        data_path: str,
        category: str,
        output_dir: str = './results',
        config_path: Optional[str] = None,
        device: str = 'auto',
        seed: int = 42,
        extra_callbacks: Optional[List] = None,
        enable_pixel_metrics: bool = True,
        learning_rate: Optional[float] = None,
        source: Optional[str] = None,
    ):
        """
        初始化训练器

        Args:
            model_name: 模型名称 (fre/patchcore/draem/padim)
            data_path: 数据集路径（MVTec AD 格式）
            category: 产品类别名称
            output_dir: 结果输出目录
            config_path: 配置文件路径（可选，保留参数兼容性）
            device: 计算设备 (auto/cpu/cuda)
            seed: 随机种子
            extra_callbacks: 额外的 PyTorch Lightning 回调列表（可选，默认 []）
            enable_pixel_metrics: 是否启用像素级指标（上传数据集无 ground_truth 时关闭）
            learning_rate: 覆盖模型默认学习率（仅 DRAEM/FRE 生效；PatchCore/PaDiM 忽略）
            source: 数据集来源（default / user），训练结束后把结果移动到对应子目录。
                    不指定则保持 anomalib 默认路径。
        """
        if model_name not in SUPPORTED_MODELS:
            raise ValueError(f"不支持的模型: {model_name}。请选择: {SUPPORTED_MODELS}")

        self.model_name = model_name
        self.data_path = Path(data_path)
        self.category = category
        self.output_dir = Path(output_dir)
        self.device = device
        self.seed = seed
        self.extra_callbacks = extra_callbacks or []
        self.enable_pixel_metrics = enable_pixel_metrics
        self.learning_rate = learning_rate
        self.source = source

        # 加载 YAML 配置（如果提供）
        self.config = None
        if config_path:
            config_path = Path(config_path)
            if config_path.exists():
                import yaml
                with open(config_path, 'r', encoding='utf-8') as f:
                    self.config = yaml.safe_load(f)
                print(f"[CONFIG] 已加载配置文件: {config_path}")
            else:
                print(f"[WARN] 配置文件不存在: {config_path}")
        
        # 数据模块和模型
        self.datamodule = None
        self.model = None
        self.engine = None
        
        # 结果
        self.results: Optional[Dict[str, Any]] = None
    
    def _print_model_info(self):
        """打印模型信息"""
        info = MODEL_INFO[self.model_name]
        print("="*70)
        print(f"[MODEL] 核心算法复现模块 - {self.model_name.upper()}")
        print("="*70)
        print(f"[INFO] 方向: {info['方向']}")
        print(f"[PRINCIPLE] 原理: {info['原理']}")
        print(f"[FEATURE] 特点: {info['特点']}")
        print(f"[STAT] 复现难度: {info['复现难度']}")
        print(f"[TIME]  预估训练时间: {info['训练时间']}")
        print("="*70)
    
    def setup(self):
        """设置数据模块和模型"""
        print("\n[STAT] 加载数据集...")
        
        # 从配置中提取模型参数（位于 model.init_args）
        model_config = None
        if self.config and 'model' in self.config and 'init_args' in self.config['model']:
            model_config = self.config['model']['init_args']
        
        # 创建数据模块
        self.datamodule = get_datamodule_from_config(
            str(self.data_path),
            self.category,
            self.model_name,
            self.config
        )
        self.datamodule.setup()
        
        print(f"   训练集样本数: {len(self.datamodule.train_data)}")
        print(f"   测试集样本数: {len(self.datamodule.test_data)}")
        
        print(f"\n[BUILD] 创建 {self.model_name} 模型...")
        self.model = get_model_from_config(self.model_name, model_config, self.enable_pixel_metrics)
    
    def _load_required_config(self, config_key: str, config_section: str = None, error_msg: str = None) -> Any:
        """
        从配置中加载必需参数，如果缺失则报错
        
        Args:
            config_key: 配置键名
            config_section: 配置所在 section（如 'trainer', 'model'）
            error_msg: 自定义错误信息
            
        Returns:
            配置值
            
        Raises:
            ValueError: 配置缺失时抛出
        """
        value = None
        
        # 1. 尝试从传入的 YAML 配置读取
        if self.config:
            if config_section and config_section in self.config:
                value = self.config[config_section].get(config_key)
            else:
                value = self.config.get(config_key)
        
        # 2. 尝试从 configs/config.yaml 读取
        if value is None:
            value = get(config_key, None)
        
        # 3. 如果仍然缺失，报错
        if value is None:
            if error_msg is None:
                section_str = f"{config_section}." if config_section else ""
                error_msg = f"配置缺失: 请在 YAML 配置文件中设置 {section_str}{config_key}"
            raise ValueError(error_msg)
        
        return value
    
    def train(self, max_epochs: Optional[int] = None) -> Dict[str, Any]:
        """
        训练模型
        
        Args:
            max_epochs: 最大训练轮次（可选，优先从 YAML 配置读取）
        
        Returns:
            Dict: 训练结果
            
        Raises:
            ValueError: 配置缺失时抛出
        """
        self._print_model_info()
        self.setup()
        
        # 从 YAML 配置严格读取 max_epochs
        if max_epochs is None:
            max_epochs = self._load_required_config(
                'max_epochs',
                config_section='trainer',
                error_msg=f"训练配置缺失: 请在 configs/{self.model_name}.yaml 的 trainer 部分设置 max_epochs"
            )
        
        # 读取早停配置
        early_stopping_callback = None
        if self.config and 'early_stopping' in self.config:
            early_stopping_config = self.config['early_stopping']
            if early_stopping_config.get('enabled', False):
                from pytorch_lightning.callbacks import EarlyStopping
                
                # 读取早停参数，使用 _load_required_config 模式
                es_monitor = early_stopping_config.get('monitor', 'image_AUROC')
                es_mode = early_stopping_config.get('mode', 'max')
                es_patience = early_stopping_config.get('patience', 10)
                es_min_delta = early_stopping_config.get('min_delta', 0.001)
                
                early_stopping_callback = EarlyStopping(
                    monitor=es_monitor,
                    mode=es_mode,
                    patience=es_patience,
                    min_delta=es_min_delta,
                    verbose=True,
                )
                print(f"   [INFO] 启用早停机制: monitor={es_monitor}, patience={es_patience}, mode={es_mode}")
        
        # 创建 Engine (禁用 rich 进度条避免 Windows GBK 编码问题)
        print("\n[WAIT] 开始训练...")
        if self.model_name in ('patchcore', 'padim'):
            print("   [TIP] PatchCore 无需训练 epoch，正在构建特征记忆库...")

        # 创建 Engine
        callbacks = list(self.extra_callbacks or [])
        if early_stopping_callback:
            callbacks.append(early_stopping_callback)

        # 若显式传入 learning_rate，对需要优化器的模型添加回调覆盖学习率
        if self.learning_rate is not None and self.model_name in ('draem', 'fre'):
            print(f"   [INFO] 使用自定义学习率: {self.learning_rate}")
            callbacks.append(_LearningRateSetter(self.learning_rate))

        self.engine = Engine(
            max_epochs=max_epochs,
            accelerator=self.device,
            devices=1,
            default_root_dir=str(self.output_dir / self.model_name),
            logger=False,
            enable_progress_bar=False,  # 禁用 rich 进度条
            # 显式传入 None 而非空列表，避免 anomalib Engine 对空列表的解析行为差异
            callbacks=callbacks if callbacks else None,
        )

        # 训练
        self.engine.fit(
            datamodule=self.datamodule,
            model=self.model,
        )
        
        print("[OK] 训练完成")
        return {'status': 'success', 'epochs': max_epochs}
    
    def evaluate(self, checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        """
        评估模型并输出4个硬性指标
        
        硬性指标:
            - 图像级: AUROC, AUPR
            - 像素级: Pixel-level AUROC, PRO
        
        Args:
            checkpoint_path: 模型权重路径（可选，默认使用训练后的模型）
        
        Returns:
            Dict: 包含4个硬性指标的结果字典
        """
        print("\n" + "="*70)
        print("[TEST] 模型评估 - 输出4个硬性指标")
        print("="*70)
        
        # 如果没有训练过，先设置
        if self.engine is None:
            self.setup()
            self.engine = Engine(
                accelerator=self.device,
                devices=1,
                default_root_dir=str(self.output_dir / self.model_name),
                logger=False,
                enable_progress_bar=False,
            )
        
        # 测试
        print("\n[WAIT] 开始测试...")
        test_results = self.engine.test(
            datamodule=self.datamodule,
            model=self.model,
            ckpt_path=checkpoint_path,
        )
        
        if test_results and len(test_results) > 0:
            results = test_results[0]
        else:
            results = {}
        
        # 提取4个硬性指标（兼容不同模型返回的字段名）
        # FRE 返回: AUROC, AUPR, pixel_AUROC, pixel_PRO
        # PatchCore/DRAEM 返回: image_AUROC, image_AUPR, pixel_AUROC, pixel_PRO
        self.results = {
            'image_AUROC': results.get('image_AUROC', results.get('AUROC', 0.0)),
            'image_AUPR': results.get('image_AUPR', results.get('AUPR', 0.0)),
            'pixel_AUROC': results.get('pixel_AUROC', 0.0),
            'pixel_PRO': results.get('pixel_PRO', 0.0),
        }
        
        # 打印4个硬性指标
        print("\n" + "-"*70)
        print("[STAT] 4个硬性指标评估结果")
        print("-"*70)
        
        # 图像级指标
        image_auroc = self.results.get('image_AUROC', 0) * 100
        image_aupr = self.results.get('image_AUPR', 0) * 100
        
        print("\n【图像级指标】- 判断图片是否有缺陷")
        print(f"   [OK] AUROC: {image_auroc:.2f}%")
        print(f"   [OK] AUPR:  {image_aupr:.2f}%")
        
        # 像素级指标
        pixel_auroc = self.results.get('pixel_AUROC', 0) * 100
        pixel_pro = self.results.get('pixel_PRO', 0) * 100
        
        print("\n【像素级指标】- 判断缺陷具体位置")
        print(f"   [OK] Pixel AUROC: {pixel_auroc:.2f}%")
        print(f"   [OK] PRO:          {pixel_pro:.2f}%")
        
        print("-"*70)
        
        # 计算最优阈值 (Youden's J)
        print("\n[WAIT] 计算最优阈值...")
        optimal_threshold = self._compute_optimal_threshold(checkpoint_path=checkpoint_path)
        self.results['optimal_threshold'] = optimal_threshold
        print(f"   [OK] 最优阈值: {optimal_threshold:.3f} (Youden's J)")
        
        # 保存结果
        self._save_results()

        # 如果指定了 source，将 anomalib 生成的结果目录移动到 default/user 下
        self._reorganize_result_dir()

        return self.results

    def _reorganize_result_dir(self) -> None:
        """
        将 anomalib 生成的结果目录移动到 {source}/{category} 下。

        anomalib 默认会生成 MVTec/{category} 或 {category}（取决于数据模块类型）。
        调用方通过 source='default' / 'user' 指定目标子目录。
        """
        if not self.source:
            return

        model_subdir_map = {
            'fre': 'Fre',
            'patchcore': 'Patchcore',
            'draem': 'Draem',
            'padim': 'Padim',
        }
        model_subdir = model_subdir_map.get(self.model_name, self.model_name.capitalize())
        base = self.output_dir / self.model_name / model_subdir

        # 可能的历史源目录
        possible_sources = [
            base / 'MVTec' / self.category,
            base / self.category,
        ]
        dst = base / self.source / self.category

        for src in possible_sources:
            if not src.exists() or src == dst:
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                shutil.rmtree(dst)
            shutil.move(str(src), str(dst))
            print(f"\n[REORG] 结果目录已整理: {src} -> {dst}")
            break
    
    def _compute_optimal_threshold(self, checkpoint_path: Optional[str] = None) -> float:
        """
        使用 Youden's J 统计量计算最优阈值
        
        Youden's J = Sensitivity + Specificity - 1
        = TP/(TP+FN) + TN/(TN+FP) - 1
        
        在 0-1 范围内搜索使 J 最大的阈值
        """
        # 获取默认阈值（从配置文件）
        default_threshold = get('threshold.default', 0.5)
        
        if self.engine is None or self.datamodule is None:
            return default_threshold
        
        try:
            # 获取阈值搜索配置
            search_config = get('evaluation.threshold_search', {})
            search_steps = search_config.get('steps', 100)
            search_min = search_config.get('min', 0.0)
            search_max = search_config.get('max', 1.0)
            
            # 获取预测结果
            predictions = self.engine.predict(
                datamodule=self.datamodule,
                model=self.model,
                ckpt_path=checkpoint_path,
            )
            
            # 收集得分和标签
            good_scores = []
            bad_scores = []
            
            for pred in predictions:
                # pred.pred_score 可能是多元素 tensor（如 DRAEM 返回向量），取最大值作为图像级得分
                score = float(pred.pred_score.cpu().max().item())
                # gt_label 可能是多元素 tensor，统一转为标量（取第一个元素）
                gt_label_tensor = pred.gt_label.cpu()
                if gt_label_tensor.numel() == 1:
                    gt_label_val = bool(gt_label_tensor.item())
                else:
                    # 多元素时取第一个元素
                    gt_label_val = bool(gt_label_tensor.flatten()[0].item())
                # 检查是否为 GOOD 样本 (gt_label = False/0 表示正常)
                is_good = not gt_label_val
                
                if is_good:
                    good_scores.append(score)
                else:
                    bad_scores.append(score)
            
            if not good_scores or not bad_scores:
                return default_threshold
            # Diagnostic: 输出分数分布信息，帮助理解阈值搜索的行为
            try:
                if len(good_scores) > 0 and len(bad_scores) > 0:
                    good_arr = np.array(good_scores, dtype=float)
                    bad_arr = np.array(bad_scores, dtype=float)
                    print(
                        f"[DIAG] score distribution - good: min={good_arr.min():.3f} max={good_arr.max():.3f} mean={good_arr.mean():.3f} | "
                        f"bad:  min={bad_arr.min():.3f} max={bad_arr.max():.3f} mean={bad_arr.mean():.3f}"
                    )
            except Exception:
                pass
            
            # 搜索最优阈值
            best_threshold = default_threshold
            best_j = -1
            
            # 在得分范围内搜索
            all_scores = good_scores + bad_scores
            # 使用固定搜索区间 [search_min, search_max]，而不是受实际分数范围限制
            min_score = search_min
            max_score = search_max
            
            # 在范围内均匀采样 search_steps 个点
            step_size = (max_score - min_score) / search_steps
            for i in range(search_steps + 1):
                threshold = min_score + i * step_size
                
                # True Positive: BAD 正确分类为异常
                tp = sum(1 for s in bad_scores if s > threshold)
                # True Negative: GOOD 正确分类为正常
                tn = sum(1 for s in good_scores if s <= threshold)
                # False Positive: GOOD 错误分类为异常
                fp = sum(1 for s in good_scores if s > threshold)
                # False Negative: BAD 错误分类为正常
                fn = sum(1 for s in bad_scores if s <= threshold)
                
                # 计算 Youden's J
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                j = sensitivity + specificity - 1
                
                if j > best_j:
                    best_j = j
                    best_threshold = threshold
            
            # 更新当前数据集的最优阈值到结果 JSON（仅对当前 category 的 JSON 生效）
            self._update_results_json_threshold(best_threshold)
            return round(best_threshold, 3)
            
        except Exception as e:
            print(f"   [WARN] 阈值计算失败: {e}，使用默认值 {default_threshold}")
            return default_threshold
    
    def _save_results(self):
        """保存评估结果"""
        result_dir = self.output_dir / 'comparison'
        result_dir.mkdir(parents=True, exist_ok=True)
        
        # 准备保存的数据
        # optimal_threshold 在 self.results 中（顶层），与 metrics 平级保存，便于 UI 读取
        save_data = {
            'model': self.model_name,
            'category': self.category,
            'timestamp': datetime.now().isoformat(),
            'metrics': self.results,
            'optimal_threshold': self.results.get('optimal_threshold'),
        }
        
        # 保存为 JSON
        json_path = result_dir / f'{self.model_name}_{self.category}_results.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n[FILE] 结果已保存: {json_path}")

    def _update_results_json_threshold(self, threshold_value: float) -> None:
        """将当前数据集的最优阈值写入对应的 results JSON 文件。

        目标文件形如: results/comparison/<model>_<category>_results.json
        其中 metrics.optimal_threshold 和顶层 optimal_threshold 将被更新。
        """
        try:
            json_path = self.output_dir / 'comparison' / f"{self.model_name}_{self.category}_results.json"
            if not json_path.exists():
                return
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # 兼容历史结构：metrics 下有 optimistic field
            if isinstance(data, dict):
                metrics = data.get('metrics', {})
                if isinstance(metrics, dict):
                    metrics['optimal_threshold'] = threshold_value
                data['metrics'] = metrics
                data['optimal_threshold'] = threshold_value
            # 写回文件
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"[FILE] 更新阈值到: {json_path} (threshold={threshold_value:.3f})")
        except Exception as e:
            print(f"   [WARN] 无法更新阈值 JSON: {e}")
    
    def train_and_evaluate(self, max_epochs: Optional[int] = None) -> Dict:
        """
        完整流程：训练 + 评估
        
        Args:
            max_epochs: 最大训练轮次
        
        Returns:
            Dict: 评估结果（4个硬性指标）
        """
        self.train(max_epochs=max_epochs)
        return self.evaluate()


def compare_models(results_dir: str, category: str):
    """
    对比多个模型的结果
    
    Args:
        results_dir: 结果目录
        category: 产品类别
    """
    result_dir = Path(results_dir) / 'comparison'
    
    if not result_dir.exists():
        print(f"[FAIL] 结果目录不存在: {result_dir}")
        return
    
    # 收集所有结果
    all_results = []
    for model_name in SUPPORTED_MODELS:
        json_path = result_dir / f'{model_name}_{category}_results.json'
        if json_path.exists():
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                metrics = data.get('metrics', {})
                info = MODEL_INFO.get(model_name, {})
                
                all_results.append({
                    'Model': model_name.upper(),
                    '方向': info.get('方向', 'N/A'),
                    'AUROC(%)': metrics.get('image_AUROC', 0) * 100,
                    'AUPR(%)': metrics.get('image_AUPR', 0) * 100,
                    'Pixel AUROC(%)': metrics.get('pixel_AUROC', 0) * 100,
                    'PRO(%)': metrics.get('pixel_PRO', 0) * 100
                })
    
    if not all_results:
        print("[FAIL] 未找到任何结果文件")
        return
    
    # 创建 DataFrame
    df = pd.DataFrame(all_results)
    
    # 打印表格
    print("\n" + "="*70)
    print("[STAT] 四算法对比结果（4个硬性指标）")
    print("="*70)
    print("\n" + df.to_string(index=False))
    
    # 保存为 CSV
    csv_path = result_dir / f'comparison_{category}.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n[STAT] 对比表格已保存: {csv_path}")
    
    # 生成 Markdown 报告
    md_path = result_dir / f'report_{category}.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# 异常检测算法对比报告\n\n")
        f.write(f"**产品类别**: {category}\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 算法说明
        f.write("## 算法说明\n\n")
        for model_name, info in MODEL_INFO.items():
            f.write(f"### {model_name.upper()}\n")
            f.write(f"- **方向**: {info['方向']}\n")
            f.write(f"- **原理**: {info['原理']}\n")
            f.write(f"- **复现难度**: {info['复现难度']}\n\n")
        
        # 对比表格
        f.write("## 4个硬性指标对比\n\n")
        f.write("| 算法 | 方向 | AUROC | AUPR | Pixel AUROC | PRO |\n")
        f.write("|:---:|:---:|:---:|:---:|:---:|:---:|")
        for r in all_results:
            f.write(f"| {r['Model']} | {r['方向']} | {r['AUROC(%)']:.2f}% | {r['AUPR(%)']:.2f}% | {r['Pixel AUROC(%)']:.2f}% | {r['PRO(%)']:.2f}% |\n")
        
        f.write("\n## 指标说明\n\n")
        f.write("### 图像级指标（判断图片是否有缺陷）\n")
        f.write("- **AUROC**: 接收者操作特征曲线下面积，越接近100%越好\n")
        f.write("- **AUPR**: 精确率-召回率曲线下面积，在不平衡数据中更稳定\n\n")
        f.write("### 像素级指标（判断缺陷具体位置）\n")
        f.write("- **Pixel AUROC**: 像素级ROC曲线下面积，评估异常定位精度\n")
        f.write("- **PRO**: Per-Region Overlap，评估连续异常区域的检测能力\n")
    
    print(f"[FILE] 报告已保存: {md_path}")


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description='核心算法复现模块 - 训练和评估3种异常检测算法 (Anomalib 2.x)'
    )
    parser.add_argument('--model', '-m', type=str, default='patchcore',
                        choices=SUPPORTED_MODELS + ['all'],
                        help='模型名称 (fre/patchcore/draem/padim/all)')
    parser.add_argument('--data_path', '-d', type=str, default='./data',
                        help='数据集路径（MVTec AD 格式）')
    parser.add_argument('--category', '-c', type=str, default='bottle',
                        help='产品类别名称')
    parser.add_argument('--output_dir', '-o', type=str, default='./results',
                        help='结果输出目录')
    parser.add_argument('--eval_only', action='store_true',
                        help='仅评估模式（不训练）')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='评估时使用的权重路径')
    parser.add_argument('--device', type=str, default='auto',
                        help='计算设备 (auto/cpu/cuda)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--epochs', type=int, default=None,
                        help='最大训练轮次')
    
    args = parser.parse_args()
    
    print("="*70)
    print("[Algorithm] Core Algorithm Module (Anomalib 2.x)")
    print("="*70)
    print(f"\n[PATH] 数据集路径: {args.data_path}")
    print(f"[CATEGORY] 产品类别: {args.category}")
    print(f"[CONFIG]  计算设备: {args.device}")
    
    # 确定要运行的模型
    models_to_run = SUPPORTED_MODELS if args.model == 'all' else [args.model]
    
    # 训练和评估
    for model_name in models_to_run:
        try:
            trainer = AnomalyDetectionTrainer(
                model_name=model_name,
                data_path=args.data_path,
                category=args.category,
                output_dir=args.output_dir,
                device=args.device,
                seed=args.seed
            )
            
            if args.eval_only:
                trainer.evaluate(args.checkpoint)
            else:
                trainer.train_and_evaluate(max_epochs=args.epochs)
                
        except Exception as e:
            print(f"\n[FAIL] 模型 {model_name} 运行失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 生成对比报告
    if len(models_to_run) > 1:
        compare_models(args.output_dir, args.category)
    
    print("\n" + "="*70)
    print("[OK] 所有任务已完成!")
    print("="*70)


if __name__ == '__main__':
    main()
