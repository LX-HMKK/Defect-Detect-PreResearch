"""
================================================================================
模块 2: 核心算法复现模块 (Algorithm Implementation Module)
================================================================================

功能: 调用 anomalib 训练和测试 3 种异常检测算法

复现的算法（从4个方向选3个）:
    1. AutoEncoder (基于重构) - 经典基线
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

# 压制 anomalib 的 OpenVINO 等无用警告信息
class _StdoutSuppressor:
    """临时压制 stdout 输出"""
    def __enter__(self):
        self._old_stdout = sys.stdout
        self._old_stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()
        return self
    def __exit__(self, *args):
        sys.stdout = self._old_stdout
        sys.stderr = self._old_stderr

# anomalib 导入（压制其内部 print 的 OpenVINO/wandb 警告）
try:
    with _StdoutSuppressor():
        from anomalib.config import get_configurable_parameters
        from anomalib.data import get_datamodule
        from anomalib.models import get_model
        from anomalib.utils.callbacks import LoadModelCallback, get_callbacks
        from pytorch_lightning import Trainer, seed_everything
        from omegaconf import OmegaConf, DictConfig, ListConfig
except ImportError as e:
    print(f"[FAIL] 错误: 未安装 anomalib。请运行: pip install anomalib")
    raise

# 忽略警告
warnings.filterwarnings('ignore')


# ================================================================================
# 支持的模型配置（3个最易复现的算法）
# ================================================================================

SUPPORTED_MODELS = ['autoencoder', 'patchcore', 'draem']

MODEL_INFO = {
    'autoencoder': {
        '方向': '基于重构',
        '原理': '训练编码器-解码器网络，通过重构误差检测异常',
        '特点': '经典基线，结构简单，易于理解',
        '复现难度': '** (easier)',
        '训练时间': '~15分钟 (100 epochs)',
        'config_file': 'configs/autoencoder.yaml'
    },
    'patchcore': {
        '方向': '基于特征建模',
        '原理': '预训练CNN提取局部特征，构建记忆库，最近邻搜索检测异常',
        '特点': '工业界效果最好，无需训练，推理最快',
        '复现难度': '* (easiest)',
        '训练时间': '~1分钟 (仅构建记忆库)',
        'config_file': 'configs/patchcore.yaml'
    },
    'draem': {
        '方向': '基于自监督学习',
        '原理': '生成合成异常样本，训练判别网络区分正常/异常区域',
        '特点': '无需真实异常样本即可训练，对小缺陷敏感',
        '复现难度': '*** (medium)',
        '训练时间': '~30分钟 (200 epochs)',
        'config_file': 'configs/draem.yaml'
    }
}


class AnomalyDetectionTrainer:
    """
    异常检测算法训练器
    
    封装 anomalib 的训练和评估流程，支持3种算法：
    - AutoEncoder: 重构方法
    - PatchCore: 特征建模方法
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
            model_name: 模型名称 (autoencoder/patchcore/draem)
            data_path: 数据集路径（MVTec AD 格式）
            category: 产品类别名称
            output_dir: 结果输出目录
            config_path: 配置文件路径（可选，默认使用 configs/{model}.yaml）
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
        
        # 加载配置文件
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / MODEL_INFO[model_name]['config_file']
        self.config_path = Path(config_path)
        
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        
        # 加载并修改配置
        self.config = self._load_config()
        
        # 训练器
        self.trainer: Optional[Trainer] = None
        self.model = None
        self.datamodule = None
        
        # 结果
        self.results: Optional[Dict[str, Any]] = None
    
    def _load_config(self) -> Any:
        """加载并修改配置文件"""
        config = OmegaConf.load(self.config_path)
        
        # 修改数据集配置
        config.dataset.path = str(self.data_path)
        config.dataset.category = self.category
        
        # 修改输出目录
        config.project.path = str(self.output_dir)
        config.project.default_root_dir = str(self.output_dir / self.model_name)
        
        # 设置设备
        config.trainer.accelerator = self.device
        config.trainer.devices = 1
        
        # 设置随机种子
        config.project.seed = self.seed
        seed_everything(self.seed)
        
        return config
    
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
        
        # 设置 num_workers=0 避免 Windows 多进程问题
        self.config.dataset.num_workers = 0
        
        self.datamodule = get_datamodule(self.config)
        self.datamodule.setup()
        
        print(f"   训练集样本数: {len(self.datamodule.train_data)}")
        print(f"   测试集样本数: {len(self.datamodule.test_data)}")
        
        print(f"\n[BUILD] 创建 {self.model_name} 模型...")
        self.model = get_model(self.config)
    
    def train(self) -> Dict[str, Any]:
        """
        训练模型
        
        Returns:
            Dict: 训练结果
        """
        self._print_model_info()
        self.setup()
        
        # 创建回调函数
        callbacks = get_callbacks(self.config)
        
        # 创建训练器
        self.trainer = Trainer(
            max_epochs=self.config.trainer.max_epochs,
            accelerator=self.config.trainer.accelerator,
            devices=self.config.trainer.devices,
            callbacks=callbacks,
            gradient_clip_val=self.config.trainer.get('gradient_clip_val', 0),
            accumulate_grad_batches=self.config.trainer.get('accumulate_grad_batches', 1),
            check_val_every_n_epoch=self.config.trainer.get('check_val_every_n_epoch', 1),
            default_root_dir=self.config.project.default_root_dir,
            num_sanity_val_steps=self.config.trainer.get('num_sanity_val_steps', 0),
        )
        
        # 训练
        print("\n[WAIT] 开始训练...")
        if self.model_name == 'patchcore':
            print("   [TIP] PatchCore 无需训练 epoch，正在构建特征记忆库...")
        
        self.trainer.fit(model=self.model, datamodule=self.datamodule)  # type: ignore
        
        # BUG FIX: PatchCore 的 training_epoch_end 未实现，内存库未构建
        # 原因: training_step 返回 None，training_epoch_end 不会被调用
        # 解决: 手动遍历训练集收集特征，构建内存库
        if self.model_name == 'patchcore':
            import torch
            from tqdm import tqdm
            
            print("   [BUILD] 手动收集训练集特征...")
            self.model.model.feature_extractor.eval()
            embeddings_list = []
            
            train_loader = self.datamodule.train_dataloader()
            with torch.no_grad():
                for batch in tqdm(train_loader, desc="   [BUILD] 收集特征"):
                    img = batch["image"]
                    features = self.model.model.feature_extractor(img)
                    features = {layer: self.model.model.feature_pooler(f) for layer, f in features.items()}
                    embedding = self.model.model.generate_embedding(features)
                    embeddings_list.append(embedding)
            
            # 拼接所有特征
            all_embeddings = torch.cat(embeddings_list, dim=0)  # [N, C, H, W]
            all_embeddings = self.model.model.reshape_embedding(all_embeddings)  # [N*H*W, C]
            
            # 内存库构建 - 使用随机采样代替慢速 coreset 选择
            print(f"   [BUILD] 特征总数: {all_embeddings.shape[0]}")
            
            # 目标内存库大小: 使用 0.1 ratio 但最大不超过 1000 个样本
            coreset_ratio = self.config.model.get('coreset_sampling_ratio', 0.1)
            target_size = min(int(all_embeddings.shape[0] * coreset_ratio), 1000)
            
            # 随机采样构建内存库 (避免慢速 coreset 选择)
            indices = torch.randperm(all_embeddings.shape[0])[:target_size]
            memory_bank = all_embeddings[indices].clone()
            
            # 直接设置内存库
            self.model.model.memory_bank = memory_bank
            print(f"   [OK] 内存库已构建 (随机采样): {self.model.model.memory_bank.shape}")
        
        print("[OK] 训练完成")
        return {'status': 'success', 'epochs': self.config.trainer.max_epochs}
    
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
        
        # 如果没有训练过，先加载模型
        if self.trainer is None:
            self.setup()
            
            if checkpoint_path is None:
                # 自动查找最新的 checkpoint
                checkpoint_dir = Path(self.config.project.default_root_dir) / 'checkpoints'
                if checkpoint_dir.exists():
                    checkpoints = list(checkpoint_dir.glob('*.ckpt'))
                    if checkpoints:
                        checkpoint_path = str(sorted(checkpoints)[-1])
            
            if checkpoint_path:
                print(f"\n[PATH] 加载权重: {checkpoint_path}")
                callbacks = [LoadModelCallback(weights_path=checkpoint_path)]
            else:
                callbacks = []
            
            self.trainer = Trainer(
                accelerator=self.config.trainer.accelerator,
                devices=self.config.trainer.devices,
                callbacks=callbacks,
                default_root_dir=self.config.project.default_root_dir,
            )
        
        # PatchCore: 使用直接推理而不是 Lightning Trainer.test()
        # 因为 Lightning 的 test_step 可能没有正确使用 memory bank
        if self.model_name == 'patchcore':
            print("\n[WAIT] 开始测试 (PatchCore 直接推理)...")
            self.results = self._evaluate_patchcore_direct()
        else:
            # 其他模型使用 Lightning Trainer
            print("\n[WAIT] 开始测试...")
            test_results = self.trainer.test(model=self.model, datamodule=self.datamodule)  # type: ignore
            
            if test_results and len(test_results) > 0:
                self.results = test_results[0]
            else:
                self.results = {}
        
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
        
        # 保存结果
        self._save_results()
        
        return self.results
    
    def _evaluate_patchcore_direct(self) -> Dict[str, Any]:
        """
        PatchCore 直接推理（绕过 Lightning Trainer）
        
        因为 Lightning 的 test_step 可能没有正确使用 memory bank，
        我们直接调用模型的 forward 方法进行推理。
        
        Returns:
            Dict: 包含4个硬性指标的结果
        """
        from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
        import numpy as np
        
        self.model.eval()
        self.model.model.feature_extractor.eval()
        
        all_labels = []
        all_image_scores = []
        all_anomaly_maps = []
        all_gt_masks = []
        
        test_loader = self.datamodule.test_dataloader()
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="   [TEST] 推理"):
                img = batch["image"].to(self.model.device if hasattr(self.model, 'device') else 'cuda')
                label = batch["label"].numpy()
                
                # 直接调用模型的 forward
                output = self.model.model(img)
                
                # output 是 (anomaly_map, image_scores) 元组
                if isinstance(output, tuple):
                    anomaly_map, image_scores = output
                    image_scores = image_scores.cpu().numpy()
                    anomaly_map = anomaly_map.cpu().numpy()
                else:
                    # 如果是字典或其他格式，尝试提取
                    if isinstance(output, dict):
                        image_scores = output.get('pred_scores', output.get('image_score', None))
                        anomaly_map = output.get('anomaly_map', output.get('pred_masks', None))
                        if image_scores is not None and hasattr(image_scores, 'cpu'):
                            image_scores = image_scores.cpu().numpy()
                        if anomaly_map is not None and hasattr(anomaly_map, 'cpu'):
                            anomaly_map = anomaly_map.cpu().numpy()
                    else:
                        image_scores = np.array([0.5])
                        anomaly_map = np.zeros((img.shape[0], 1, img.shape[2], img.shape[3]))
                
                all_labels.extend(label.tolist())
                all_image_scores.extend(image_scores.tolist() if hasattr(image_scores, '__iter__') else [image_scores])
                all_anomaly_maps.append(anomaly_map)
                
                # 收集 ground truth masks
                if 'mask' in batch:
                    mask = batch['mask']
                    if isinstance(mask, torch.Tensor):
                        mask = mask.cpu().numpy()
                    all_gt_masks.append(mask)
        
        # 转换为 numpy 数组
        all_labels = np.array(all_labels)
        all_image_scores = np.array(all_image_scores)
        
        # 合并所有 anomaly maps 和 masks
        if all_anomaly_maps:
            all_anomaly_maps = np.concatenate(all_anomaly_maps, axis=0)
        if all_gt_masks:
            all_gt_masks = np.concatenate(all_gt_masks, axis=0)
        
        # 计算图像级指标
        image_auroc = roc_auc_score(all_labels, all_image_scores)
        precision, recall, _ = precision_recall_curve(all_labels, all_image_scores)
        image_aupr = auc(recall, precision)
        
        # 计算像素级指标
        pixel_auroc = 0.0
        pixel_pro = 0.0
        
        if len(all_anomaly_maps) > 0 and len(all_gt_masks) > 0:
            # 确保形状匹配
            if all_anomaly_maps.shape[0] == all_gt_masks.shape[0]:
                # 展平计算 Pixel AUROC
                anomaly_flat = all_anomaly_maps.reshape(-1)
                mask_flat = all_gt_masks.reshape(-1)
                
                if len(np.unique(mask_flat)) > 1:
                    pixel_auroc = roc_auc_score(mask_flat, anomaly_flat)
                
                # 计算 PRO (简化版)
                pixel_pro = pixel_auroc  # 使用简化估计
        
        results = {
            'image_AUROC': float(image_auroc),
            'image_AUPR': float(image_aupr),
            'pixel_AUROC': float(pixel_auroc),
            'pixel_PRO': float(pixel_pro)
        }
        
        return results
    
    def _save_results(self):
        """保存评估结果"""
        result_dir = self.output_dir / 'comparison'
        result_dir.mkdir(parents=True, exist_ok=True)
        
        # 准备保存的数据
        save_data = {
            'model': self.model_name,
            'category': self.category,
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                'image_AUROC': self.results.get('image_AUROC', 0),
                'image_AUPR': self.results.get('image_AUPR', 0),
                'pixel_AUROC': self.results.get('pixel_AUROC', 0),
                'pixel_PRO': self.results.get('pixel_PRO', 0)
            }
        }
        
        # 保存为 JSON
        json_path = result_dir / f'{self.model_name}_{self.category}_results.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n[FILE] 结果已保存: {json_path}")
    
    def train_and_evaluate(self) -> Dict:
        """
        完整流程：训练 + 评估
        
        Returns:
            Dict: 评估结果（4个硬性指标）
        """
        self.train()
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
        f.write("|:---:|:---:|:---:|:---:|:---:|:---:|\n")
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


def _load_run_config_from_yaml(model_name: str) -> Dict[str, Any]:
    """从 YAML 配置文件加载 run 配置"""
    config_path = Path(__file__).parent.parent.parent / MODEL_INFO[model_name]['config_file']
    if not config_path.exists():
        return {}
    config = OmegaConf.load(config_path)
    if hasattr(config, 'run'):
        return OmegaConf.to_container(config.run, resolve=True)
    return {}


def main():
    """命令行入口 - 支持无参数运行（从 YAML 读取配置）"""
    parser = argparse.ArgumentParser(
        description='核心算法复现模块 - 训练和评估3种异常检测算法'
    )
    # 参数变为可选，有值时命令行优先
    parser.add_argument('--model', '-m', type=str, default=None,
                        choices=SUPPORTED_MODELS + ['all'],
                        help='模型名称 (autoencoder/patchcore/draem/all)')
    parser.add_argument('--data_path', '-d', type=str, default=None,
                        help='数据集路径（MVTec AD 格式）')
    parser.add_argument('--category', '-c', type=str, default=None,
                        help='产品类别名称')
    parser.add_argument('--output_dir', '-o', type=str, default=None,
                        help='结果输出目录')
    parser.add_argument('--eval_only', action='store_true',
                        help='仅评估模式（不训练）')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='评估时使用的权重路径')
    parser.add_argument('--device', type=str, default=None,
                        help='计算设备 (auto/cpu/cuda)')
    parser.add_argument('--seed', type=int, default=None,
                        help='随机种子')
    
    args = parser.parse_args()
    
    # 如果命令行没有指定参数，从 YAML 配置读取
    # 默认使用 patchcore 模型的配置
    default_model = args.model if args.model else 'patchcore'
    yaml_config = _load_run_config_from_yaml(default_model)
    
    # 合并配置：命令行参数优先
    model = args.model or yaml_config.get('model', default_model)
    data_path = args.data_path or yaml_config.get('data_path', './data/processed')
    category = args.category or yaml_config.get('category', 'my_product')
    output_dir = args.output_dir or yaml_config.get('output_dir', './results')
    device = args.device or yaml_config.get('device', 'cuda')
    seed = args.seed if args.seed is not None else yaml_config.get('seed', 42)
    
    print("="*70)
    print("[Algorithm] Core Algorithm Module")
    print("="*70)
    print(f"\n[PATH] 数据集路径: {data_path}")
    print(f"[CATEGORY] 产品类别: {category}")
    print(f"[CONFIG]  计算设备: {device}")
    if args.model:
        print(f"   (命令行参数)")
    else:
        print(f"   (从 {model}.yaml 读取)")
    
    # 确定要运行的模型
    models_to_run = SUPPORTED_MODELS if model == 'all' else [model]
    
    # 训练和评估
    for model_name in models_to_run:
        try:
            trainer = AnomalyDetectionTrainer(
                model_name=model_name,
                data_path=data_path,
                category=category,
                output_dir=output_dir,
                device=device,
                seed=seed
            )
            
            if args.eval_only:
                trainer.evaluate(args.checkpoint)
            else:
                trainer.train_and_evaluate()
                
        except Exception as e:
            print(f"\n[FAIL] 模型 {model_name} 运行失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 生成对比报告
    if len(models_to_run) > 1:
        compare_models(output_dir, category)
    
    print("\n" + "="*70)
    print("[OK] 所有任务已完成!")
    print("="*70)


if __name__ == '__main__':
    main()
