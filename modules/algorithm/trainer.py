"""
================================================================================
模块 2: 核心算法复现模块 (Algorithm Implementation Module) - Anomalib 2.x
================================================================================

功能: 调用 anomalib 2.x 训练和测试 3 种异常检测算法

复现的算法（3个）:
    1. Ganomaly (基于重构/GAN)
    2. PatchCore (基于特征建模) - 工业界效果最好
    3. DRAEM (基于自监督学习) - 无需真实异常样本训练

约束条件:
    - 只用正常样本训练（无监督设定）
    - 基于 anomalib 库，不手写神经网络底层

使用示例:
    from modules.algorithm.trainer import AnomalyDetectionTrainer
    
    trainer = AnomalyDetectionTrainer(
        model_name='patchcore',
        data_path='./data/processed/my_product',
        category='my_product'
    )
    results = trainer.train()  # 训练
    metrics = trainer.evaluate()  # 测试并输出4个硬性指标
================================================================================
"""

import os
import sys
import json
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any, Mapping
import argparse
from io import StringIO

import torch
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
)

# 配置管理
from modules.config import get_model_config, get_data_config, get

# 忽略警告
warnings.filterwarnings('ignore')

# ================================================================================
# 预训练模型缓存配置
# ================================================================================

import os
from pathlib import Path

# 预训练权重缓存目录
PRETRAINED_CACHE_DIR = Path(__file__).parent.parent.parent / "pre_trained"
PRETRAINED_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 设置 Torch Hub 缓存目录
torch.hub.set_dir(str(PRETRAINED_CACHE_DIR / "torch_hub"))

# 设置 HuggingFace 缓存目录
os.environ["HF_HOME"] = str(PRETRAINED_CACHE_DIR / "huggingface")
os.environ["HF_HUB_CACHE"] = str(PRETRAINED_CACHE_DIR / "huggingface" / "hub")


# ================================================================================
# 支持的模型配置（3个算法）
# ================================================================================

SUPPORTED_MODELS = ['fre', 'patchcore', 'draem']

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
    }
}


def get_datamodule_from_config(
    data_path: str,
    category: str,
    model_name: str,
    config: Optional[Dict[str, Any]] = None
) -> Union[MVTec, Folder]:
    """
    根据配置创建数据模块
    
    Args:
        data_path: 数据目录路径
        category: 类别名称
        model_name: 模型名称
        config: 额外配置参数（可以是完整 YAML config 或 data.init_args 部分）
    
    Returns:
        MVTec 或 Folder 数据模块
    """
    data_path = Path(data_path)
    
    # 从配置管理系统获取默认配置
    default_config = get_data_config(model_name)
    train_batch_size = default_config['train_batch_size']
    eval_batch_size = default_config['eval_batch_size']
    num_workers = default_config['num_workers']
    
    # 从传入的 config 覆盖（Anomalib 2.x 格式）
    if config:
        if 'data' in config and 'init_args' in config['data']:
            # 完整 config，包含 data.init_args
            data_config = config['data']['init_args']
            train_batch_size = data_config.get('train_batch_size', train_batch_size)
            eval_batch_size = data_config.get('eval_batch_size', eval_batch_size)
            num_workers = data_config.get('num_workers', num_workers)
        else:
            # 直接是 data.init_args 或其他格式
            train_batch_size = config.get('train_batch_size', train_batch_size)
            eval_batch_size = config.get('eval_batch_size', eval_batch_size)
            num_workers = config.get('num_workers', num_workers)
    
    # 检测数据集格式
    category_path = data_path / category
    
    # 如果是 MVTec AD 格式（有 train, test, ground_truth 目录）
    if (category_path / 'train').exists() and (category_path / 'test').exists():
        return MVTec(
            root=str(data_path),
            category=category,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
        )
    else:
        # 使用 Folder 格式
        return Folder(
            root=str(category_path),
            normal_dir='train/good',
            abnormal_dir='test',
            normal_test_dir='test/good',
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            task='segmentation',
        )


def get_model_from_config(model_name: str, config: Optional[Dict[str, Any]] = None):
    """
    根据配置创建模型
    
    Args:
        model_name: 模型名称
        config: 模型配置参数
    
    Returns:
        模型实例
    """
    # 创建 evaluator，启用 AUPR 和 PRO 指标（PatchCore 和 Draem 支持像素级指标）
    from anomalib.metrics import Evaluator, AUPR, PRO, AUROC, F1Score
    evaluator = Evaluator(
        test_metrics=[
            AUROC(fields=["pred_score", "gt_label"]),
            AUPR(fields=["pred_score", "gt_label"]),
            F1Score(fields=["pred_label", "gt_label"]),
            AUROC(fields=["anomaly_map", "gt_mask"], prefix="pixel_"),
            PRO(fields=["anomaly_map", "gt_mask"], prefix="pixel_"),
            F1Score(fields=["pred_mask", "gt_mask"], prefix="pixel_"),
        ]
    )
    
    # 从配置管理系统获取模型默认配置
    model_defaults = get_model_config(model_name)
    
    if model_name == 'patchcore':
        # 从配置读取默认值，允许传入 config 覆盖
        backbone = model_defaults.get('backbone', 'wide_resnet50_2')
        layers = model_defaults.get('layers', ['layer2', 'layer3'])
        coreset_sampling_ratio = model_defaults.get('coreset_sampling_ratio', 0.1)
        num_neighbors = model_defaults.get('num_neighbors', 9)
        pre_trained = model_defaults.get('pre_trained', True)
        
        if config:
            backbone = config.get('backbone', backbone)
            layers = config.get('layers', layers)
            coreset_sampling_ratio = config.get('coreset_sampling_ratio', coreset_sampling_ratio)
            num_neighbors = config.get('num_neighbors', num_neighbors)
            pre_trained = config.get('pre_trained', pre_trained)
        
        return Patchcore(
            backbone=backbone,
            layers=layers,
            coreset_sampling_ratio=coreset_sampling_ratio,
            num_neighbors=num_neighbors,
            pre_trained=pre_trained,
            evaluator=evaluator,
        )
    
    elif model_name == 'fre':
        # FRE (Feature Reconstruction Error) - 重构法改进版
        backbone = model_defaults.get('backbone', 'resnet50')
        layer = model_defaults.get('layer', 'layer3')
        pre_trained = model_defaults.get('pre_trained', True)
        pooling_kernel_size = model_defaults.get('pooling_kernel_size', 2)
        input_dim = model_defaults.get('input_dim', 65536)
        latent_dim = model_defaults.get('latent_dim', 220)
        
        # 允许传入 config 覆盖
        if config:
            backbone = config.get('backbone', backbone)
            layer = config.get('layer', layer)
            pre_trained = config.get('pre_trained', pre_trained)
            pooling_kernel_size = config.get('pooling_kernel_size', pooling_kernel_size)
            input_dim = config.get('input_dim', input_dim)
            latent_dim = config.get('latent_dim', latent_dim)
        
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
        # 从配置读取 beta 范围和 SSPCAB 设置
        beta = model_defaults.get('beta', [0.1, 1.0])
        enable_sspcab = model_defaults.get('enable_sspcab', False)
        
        # 允许传入 config 覆盖
        if config:
            beta = config.get('beta', beta)
            enable_sspcab = config.get('enable_sspcab', enable_sspcab)
        
        return Draem(
            beta=tuple(beta) if isinstance(beta, list) else beta,
            enable_sspcab=enable_sspcab,
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
        seed: int = 42
    ):
        """
        初始化训练器
        
        Args:
            model_name: 模型名称 (efficientad/patchcore/draem)
            data_path: 数据集路径（MVTec AD 格式）
            category: 产品类别名称
            output_dir: 结果输出目录
            config_path: 配置文件路径（可选，保留参数兼容性）
            device: 计算设备 (auto/cpu/cuda)
            seed: 随机种子
        """
        if model_name not in SUPPORTED_MODELS:
            raise ValueError(f"不支持的模型: {model_name}。请选择: {SUPPORTED_MODELS}")
        
        self.model_name = model_name
        self.data_path = Path(data_path)
        self.category = category
        self.output_dir = Path(output_dir)
        self.device = device
        self.seed = seed
        
        # 加载 YAML 配置（如果提供）
        self.config = None
        if config_path:
            config_path = Path(config_path)
            if config_path.exists():
                from anomalib.deploy.config import load_config
                self.config = load_config(str(config_path))
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
        self.model = get_model_from_config(self.model_name, model_config)
    
    def train(self, max_epochs: Optional[int] = None) -> Dict[str, Any]:
        """
        训练模型
        
        Args:
            max_epochs: 最大训练轮次（可选，默认为模型推荐值）
        
        Returns:
            Dict: 训练结果
        """
        self._print_model_info()
        self.setup()
        
        # 设置默认 epoch（优先级：传入参数 > YAML 配置 > config.yaml > 代码默认值）
        if max_epochs is None:
            # 尝试从 YAML 配置读取
            if self.config and 'trainer' in self.config:
                max_epochs = self.config['trainer'].get('max_epochs')
            
            # 从主配置文件 config.yaml 读取
            if max_epochs is None:
                max_epochs = get(f'training.epochs.{self.model_name}', None)
            
            # 如果都没有，使用硬编码默认值
            if max_epochs is None:
                if self.model_name == 'patchcore':
                    max_epochs = 1
                elif self.model_name == 'fre':
                    max_epochs = 50
                elif self.model_name == 'draem':
                    max_epochs = 200
        
        # 创建 Engine (禁用 rich 进度条避免 Windows GBK 编码问题)
        print("\n[WAIT] 开始训练...")
        if self.model_name == 'patchcore':
            print("   [TIP] PatchCore 无需训练 epoch，正在构建特征记忆库...")
        
        self.engine = Engine(
            max_epochs=max_epochs,
            accelerator=self.device,
            devices=1,
            default_root_dir=str(self.output_dir / self.model_name),
            enable_progress_bar=False,  # 禁用 rich 进度条
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
        optimal_threshold = self._compute_optimal_threshold()
        self.results['optimal_threshold'] = optimal_threshold
        print(f"   [OK] 最优阈值: {optimal_threshold:.3f} (Youden's J)")
        
        # 保存结果
        self._save_results()
        
        return self.results
    
    def _compute_optimal_threshold(self) -> float:
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
            
            # 搜索最优阈值
            best_threshold = default_threshold
            best_j = -1
            
            # 在得分范围内搜索
            all_scores = good_scores + bad_scores
            min_score = max(search_min, min(all_scores))
            max_score = min(search_max, max(all_scores))
            
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
    print("[STAT] 三算法对比结果（4个硬性指标）")
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
                        help='模型名称 (ganomaly/patchcore/draem/all)')
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
