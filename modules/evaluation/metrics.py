"""
================================================================================
模块 3: 指标评测模块 (Evaluation Metrics Module)
================================================================================

功能: 计算和输出论文/综述要求的 4 个硬性指标

硬性指标 (PPT要求):
    【图像级】
    1. AUROC  (Area Under the Receiver Operating Characteristic curve)
       用途: 评估模型区分正常/异常图片的能力
       范围: 0-1，越接近1越好
    
    2. AUPR   (Area Under the Precision-Recall curve)
       用途: 在不平衡的测试集上更稳定的评估指标
       范围: 0-1，越接近1越好
    
    【像素级】
    3. Pixel-level AUROC
       用途: 评估模型定位异常像素区域的精度
       范围: 0-1，越接近1越好
    
    4. PRO    (Per-Region Overlap)
       用途: 评估连续异常区域的检测能力，对大面积异常更友好
       范围: 0-1，越接近1越好

使用示例:
    from modules.evaluation.metrics import MetricsEvaluator
    
    evaluator = MetricsEvaluator(predictions, ground_truths)
    metrics = evaluator.compute_all()  # 返回4个指标
================================================================================
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import argparse

import numpy as np
import torch
from scipy import ndimage
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc


@dataclass
class AnomalyMetrics:
    """异常检测指标数据类"""
    # 图像级指标
    image_auroc: float = 0.0  # 图像级 AUROC
    image_aupr: float = 0.0   # 图像级 AUPR
    
    # 像素级指标
    pixel_auroc: float = 0.0  # 像素级 AUROC
    pro: float = 0.0          # Per-Region Overlap
    
    def to_dict(self) -> Dict[str, float]:
        """转换为字典"""
        return {
            'image_AUROC': self.image_auroc,
            'image_AUPR': self.image_aupr,
            'pixel_AUROC': self.pixel_auroc,
            'pixel_PRO': self.pro
        }
    
    def to_percent_dict(self) -> Dict[str, float]:
        """转换为百分比字典（用于展示）"""
        return {
            'image_AUROC(%)': self.image_auroc * 100,
            'image_AUPR(%)': self.image_aupr * 100,
            'pixel_AUROC(%)': self.pixel_auroc * 100,
            'pixel_PRO(%)': self.pro * 100
        }
    
    def __str__(self) -> str:
        """字符串表示"""
        return (
            f"图像级指标:\n"
            f"  - AUROC: {self.image_auroc*100:.2f}%\n"
            f"  - AUPR:  {self.image_aupr*100:.2f}%\n"
            f"像素级指标:\n"
            f"  - Pixel AUROC: {self.pixel_auroc*100:.2f}%\n"
            f"  - PRO:         {self.pro*100:.2f}%"
        )


class MetricsEvaluator:
    """
    异常检测指标评测器
    
    计算论文要求的4个硬性指标：
    - 图像级: AUROC, AUPR
    - 像素级: Pixel AUROC, PRO
    """
    
    def __init__(self, pro_integration_limit: float = 0.3):
        """
        初始化评测器
        
        Args:
            pro_integration_limit: PRO积分的假阳性率上限（默认0.3）
        """
        self.pro_integration_limit = pro_integration_limit
    
    def compute_image_auroc(
        self,
        anomaly_scores: np.ndarray,
        ground_truth_labels: np.ndarray
    ) -> float:
        """
        计算图像级 AUROC
        
        Args:
            anomaly_scores: 异常分数 (N,) - 每张图片的异常得分
            ground_truth_labels: 真实标签 (N,) - 0=正常, 1=异常
        
        Returns:
            float: AUROC 值
        """
        if len(np.unique(ground_truth_labels)) < 2:
            # 只有一种类别，无法计算AUROC
            return 0.5
        
        return roc_auc_score(ground_truth_labels, anomaly_scores)
    
    def compute_image_aupr(
        self,
        anomaly_scores: np.ndarray,
        ground_truth_labels: np.ndarray
    ) -> float:
        """
        计算图像级 AUPR
        
        Args:
            anomaly_scores: 异常分数 (N,)
            ground_truth_labels: 真实标签 (N,) - 0=正常, 1=异常
        
        Returns:
            float: AUPR 值
        """
        if len(np.unique(ground_truth_labels)) < 2:
            return 0.0
        
        precision, recall, _ = precision_recall_curve(ground_truth_labels, anomaly_scores)
        return auc(recall, precision)
    
    def compute_pixel_auroc(
        self,
        anomaly_maps: np.ndarray,
        ground_truth_masks: np.ndarray
    ) -> float:
        """
        计算像素级 AUROC
        
        Args:
            anomaly_maps: 异常热力图 (N, H, W) - 每张图片的像素级异常分数
            ground_truth_masks: 真实掩膜 (N, H, W) - 0=正常像素, 1=异常像素
        
        Returns:
            float: Pixel AUROC 值
        """
        # 展平为1D数组
        scores_flat = anomaly_maps.reshape(-1)
        masks_flat = ground_truth_masks.reshape(-1)
        
        if len(np.unique(masks_flat)) < 2:
            return 0.5
        
        return roc_auc_score(masks_flat, scores_flat)
    
    def compute_pro(
        self,
        anomaly_maps: np.ndarray,
        ground_truth_masks: np.ndarray
    ) -> float:
        """
        计算 PRO (Per-Region Overlap)
        
        PRO 是评估连续异常区域检测能力的指标，计算方式：
        1. 将 ground_truth_masks 中的每个连通区域作为一个独立区域
        2. 对每个区域计算 Overlap = (预测为异常的像素数 ∩ 真实异常像素数) / 真实异常像素数
        3. 在不同阈值下计算平均 Overlap，并对假阳性率积分
        
        Args:
            anomaly_maps: 异常热力图 (N, H, W)
            ground_truth_masks: 真实掩膜 (N, H, W)
        
        Returns:
            float: PRO 值
        """
        anomaly_maps = np.asarray(anomaly_maps, dtype=np.float32)
        ground_truth_masks = np.asarray(ground_truth_masks)

        # 标准化异常图到 [0, 1]
        min_score = float(anomaly_maps.min())
        max_score = float(anomaly_maps.max())
        if max_score - min_score < 1e-8:
            anomaly_maps_norm = np.zeros_like(anomaly_maps, dtype=np.float32)
        else:
            anomaly_maps_norm = (anomaly_maps - min_score) / (max_score - min_score)
        
        # 生成不同阈值
        thresholds = np.linspace(0, 1, 100)
        
        pro_values = []
        fpr_values = []
        
        for threshold in thresholds:
            # 二值化预测
            pred_masks = (anomaly_maps_norm >= threshold).astype(int)
            
            # 计算每个连通区域的 Overlap
            weighted_overlap = 0.0
            total_gt_pixels = 0
            fp = 0
            tn = 0
            
            for i in range(len(ground_truth_masks)):
                gt_mask = (ground_truth_masks[i] > 0).astype(np.uint8)
                pred_mask = pred_masks[i]
                
                # 找到所有连通区域
                labeled_array, num_features = ndimage.label(gt_mask)
                
                for region_id in range(1, num_features + 1):
                    region_mask = (labeled_array == region_id).astype(int)
                    region_pixels = region_mask.sum()
                    total_gt_pixels += region_pixels
                    
                    if region_pixels > 0:
                        overlap = (pred_mask * region_mask).sum() / region_pixels
                        weighted_overlap += overlap * region_pixels

                fp += int(((pred_mask == 1) & (gt_mask == 0)).sum())
                tn += int(((pred_mask == 0) & (gt_mask == 0)).sum())
            
            # 计算平均 PRO
            if total_gt_pixels > 0:
                avg_pro = weighted_overlap / total_gt_pixels
            else:
                avg_pro = 0
            
            # 计算假阳性率 (FPR)
            fpr = fp / (fp + tn + 1e-8)
            
            pro_values.append(avg_pro)
            fpr_values.append(fpr)
        
        # 对 FPR 在 [0, pro_integration_limit] 范围内积分
        pro_values = np.array(pro_values)
        fpr_values = np.array(fpr_values)
        
        # 按 FPR 升序聚合，处理重复值
        fpr_to_pro: Dict[float, float] = {}
        for fpr, pro in zip(fpr_values.tolist(), pro_values.tolist()):
            if fpr in fpr_to_pro:
                fpr_to_pro[fpr] = max(fpr_to_pro[fpr], pro)
            else:
                fpr_to_pro[fpr] = pro
        sorted_fpr = np.array(sorted(fpr_to_pro.keys()), dtype=np.float64)
        sorted_pro = np.array([fpr_to_pro[fpr] for fpr in sorted_fpr], dtype=np.float64)

        if sorted_fpr.size == 0:
            return 0.0

        # 插值到均匀的 FPR 点
        fpr_grid = np.linspace(0, self.pro_integration_limit, 100)
        pro_interp = np.interp(fpr_grid, sorted_fpr, sorted_pro)
        
        # 计算积分 (梯形法则)
        pro_score = np.trapz(pro_interp, fpr_grid) / self.pro_integration_limit
        
        return float(np.clip(pro_score, 0.0, 1.0))
    
    def compute_all(
        self,
        anomaly_scores: np.ndarray,
        ground_truth_labels: np.ndarray,
        anomaly_maps: Optional[np.ndarray] = None,
        ground_truth_masks: Optional[np.ndarray] = None
    ) -> AnomalyMetrics:
        """
        计算所有4个硬性指标
        
        Args:
            anomaly_scores: 图像级异常分数 (N,)
            ground_truth_labels: 图像级真实标签 (N,)
            anomaly_maps: 像素级异常热力图 (N, H, W)，可选
            ground_truth_masks: 像素级真实掩膜 (N, H, W)，可选
        
        Returns:
            AnomalyMetrics: 包含4个指标的结果
        """
        metrics = AnomalyMetrics()
        
        # 图像级指标
        metrics.image_auroc = self.compute_image_auroc(anomaly_scores, ground_truth_labels)
        metrics.image_aupr = self.compute_image_aupr(anomaly_scores, ground_truth_labels)
        
        # 像素级指标
        if anomaly_maps is not None and ground_truth_masks is not None:
            metrics.pixel_auroc = self.compute_pixel_auroc(anomaly_maps, ground_truth_masks)
            metrics.pro = self.compute_pro(anomaly_maps, ground_truth_masks)
        
        return metrics
    
    def print_metrics(self, metrics: AnomalyMetrics, title: str = "评估结果"):
        """
        美观地打印指标
        
        Args:
            metrics: 指标对象
            title: 标题
        """
        print("="*70)
        print(f"[STAT] {title}")
        print("="*70)
        
        print("\n【图像级指标】- 判断图片是否有缺陷")
        print(f"   1. AUROC: {metrics.image_auroc*100:.2f}%")
        print(f"      └─ 评估模型区分正常/异常图片的能力")
        print(f"   2. AUPR:  {metrics.image_aupr*100:.2f}%")
        print(f"      └─ 在不平衡数据集上的稳定评估指标")
        
        print("\n【像素级指标】- 判断缺陷具体位置")
        print(f"   3. Pixel AUROC: {metrics.pixel_auroc*100:.2f}%")
        print(f"      └─ 评估像素级异常定位精度")
        print(f"   4. PRO:         {metrics.pro*100:.2f}%")
        print(f"      └─ 评估连续异常区域的检测能力")
        
        print("="*70)


def load_and_evaluate(results_dir: str, model_name: str, category: str) -> bool:
    """
    加载已有结果并打印指标
    
    Args:
        results_dir: 结果目录
        model_name: 模型名称
        category: 产品类别
    """
    json_path = Path(results_dir) / 'comparison' / f'{model_name}_{category}_results.json'
    
    if not json_path.exists():
        print(f"[FAIL] 结果文件不存在: {json_path}")
        return False
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    metrics_data = data.get('metrics', {})
    
    metrics = AnomalyMetrics(
        image_auroc=metrics_data.get('image_AUROC', 0),
        image_aupr=metrics_data.get('image_AUPR', 0),
        pixel_auroc=metrics_data.get('pixel_AUROC', 0),
        pro=metrics_data.get('pixel_PRO', 0)
    )
    
    evaluator = MetricsEvaluator()
    evaluator.print_metrics(metrics, f"{model_name.upper()} - {category}")
    return True


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description='指标评测模块 - 查看已保存的4个硬性指标'
    )
    parser.add_argument('--results_dir', '-r', type=str, default='./results',
                        help='结果目录')
    parser.add_argument('--model', '-m', type=str, required=True,
                        choices=['fre', 'patchcore', 'draem', 'all'],
                        help='模型名称')
    parser.add_argument('--category', '-c', type=str, required=True,
                        help='产品类别')
    
    args = parser.parse_args()
    
    print("="*70)
    print("[STAT] 指标评测模块 (Evaluation Metrics Module)")
    print("="*70)
    print("\n4个硬性指标:")
    print("  【图像级】AUROC, AUPR")
    print("  【像素级】Pixel AUROC, PRO")
    print("="*70)
    
    if args.model == 'all':
        for model in ['fre', 'patchcore', 'draem']:
            load_and_evaluate(args.results_dir, model, args.category)
            print()
    else:
        load_and_evaluate(args.results_dir, args.model, args.category)


if __name__ == '__main__':
    main()
