#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
混淆矩阵生成工具

对每个模型/数据集组合生成混淆矩阵，用于答辩论证。
基于 Youden's J 最优阈值计算 TP/FP/TN/FN，生成 matplotlib 可视化。

用法:
    python tools/run_confusion_matrix.py -m all -c all -d ./data
"""

import io
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from anomalib.engine import Engine
from anomalib.data import PredictDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(PROJECT_ROOT))

from modules.algorithm.trainer import (
    AnomalyDetectionTrainer, get_model_from_config, get_datamodule_from_config
)
from modules.algorithm import SUPPORTED_MODELS
from modules.config import get_threshold, get_model_config, get_data_config
from modules.ui.demo import AnomalyDetector

# ============================================================================
# 中文字体配置
# ============================================================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = PROJECT_ROOT / 'results' / 'confusion_matrices'


def collect_predictions(detector: AnomalyDetector, model_key: str, dataset: str,
                        data_path: str) -> Tuple[List[float], List[int]]:
    """遍历测试集收集所有图像的异常分数和真实标签"""
    data_dir = Path(data_path) / dataset
    test_dir = data_dir / 'test'
    scores = []
    labels = []

    # 加载模型
    success, _ = detector.load_model(model_key, dataset)
    if not success:
        print(f"  [WARN] 模型加载失败: {model_key}/{dataset}")
        return [], []

    # 正常样本 (label=0)
    good_dir = test_dir / 'good'
    if good_dir.exists():
        for img_path in sorted(good_dir.glob('*')):
            if img_path.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp'):
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                _, _, result_html = detector.predict(img)
                score = _extract_score(result_html)
                if score is not None:
                    scores.append(score)
                    labels.append(0)  # normal

    # 异常样本 (label=1) — 遍历所有缺陷子目录
    for defect_dir in sorted(test_dir.iterdir()):
        if not defect_dir.is_dir() or defect_dir.name == 'good':
            continue
        for img_path in sorted(defect_dir.glob('*')):
            if img_path.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp'):
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                _, _, result_html = detector.predict(img)
                score = _extract_score(result_html)
                if score is not None:
                    scores.append(score)
                    labels.append(1)  # anomaly

    return scores, labels


def _extract_score(result_html: str) -> Optional[float]:
    """从结果 HTML 中提取异常分数"""
    import re
    match = re.search(r'得分\s*<b[^>]*>\s*([\d.]+)\s*</b>', result_html)
    if match:
        return float(match.group(1))
    # 尝试备选模式：英文标签 "Score"
    match = re.search(r'Score\s*<b[^>]*>\s*([\d.]+)\s*</b>', result_html)
    if match:
        return float(match.group(1))
    return None


def compute_confusion_matrix(scores: List[float], labels: List[int],
                             threshold: float) -> Dict[str, int]:
    """在给定阈值下计算混淆矩阵"""
    tp = fp = tn = fn = 0
    for score, label in zip(scores, labels):
        pred = 1 if score > threshold else 0
        if pred == 1 and label == 1:
            tp += 1
        elif pred == 1 and label == 0:
            fp += 1
        elif pred == 0 and label == 0:
            tn += 1
        elif pred == 0 and label == 1:
            fn += 1
    return {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn}


def plot_confusion_matrix(cm: Dict[str, int], model_name: str, dataset: str,
                          threshold: float, metrics: Dict[str, float],
                          save_path: Path):
    """绘制混淆矩阵可视化"""
    tp, fp, tn, fn = cm['TP'], cm['FP'], cm['TN'], cm['FN']
    total = tp + fp + tn + fn
    if total == 0:
        print(f"  [WARN] 无有效预测: {model_name}/{dataset}")
        return

    matrix = np.array([[tn, fp], [fn, tp]])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5),
                             gridspec_kw={'width_ratios': [1, 1.2]})

    # 左侧：混淆矩阵热力图
    ax1 = axes[0]
    im = ax1.imshow(matrix, cmap='Blues', vmin=0, vmax=max(total // 2, 1))

    # 标注数值和百分比
    labels_text = [
        [f'TN={tn}\n({tn/total*100:.1f}%)', f'FP={fp}\n({fp/total*100:.1f}%)'],
        [f'FN={fn}\n({fn/total*100:.1f}%)', f'TP={tp}\n({tp/total*100:.1f}%)']
    ]
    for i in range(2):
        for j in range(2):
            color = 'white' if matrix[i, j] > total * 0.3 else 'black'
            ax1.text(j, i, labels_text[i][j], ha='center', va='center',
                     fontsize=11, fontweight='bold', color=color)

    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(['预测正常', '预测异常'], fontsize=10)
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(['实际正常', '实际异常'], fontsize=10)
    ax1.set_title(f'{model_name.upper()} @ {dataset}\n混淆矩阵 (阈值={threshold:.3f})',
                  fontsize=12, fontweight='bold', pad=10)
    plt.colorbar(im, ax=ax1, shrink=0.8, label='样本数')

    # 右侧：指标汇总
    ax2 = axes[1]
    ax2.axis('off')

    accuracy = (tp + tn) / total * 100
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    metrics_text = [
        ('准确率 Accuracy', f'{accuracy:.1f}%'),
        ('精确率 Precision', f'{precision:.1f}%'),
        ('召回率 Recall', f'{recall:.1f}%'),
        ('特异度 Specificity', f'{specificity:.1f}%'),
        ('F1 Score', f'{f1:.1f}%'),
        ('', ''),
        ('AUROC', f'{metrics.get("image_AUROC", 0)*100:.1f}%'),
        ('AUPR', f'{metrics.get("image_AUPR", 0)*100:.1f}%'),
        ('Pixel AUROC', f'{metrics.get("pixel_AUROC", 0)*100:.1f}%'),
        ('PRO', f'{metrics.get("pixel_PRO", 0)*100:.1f}%'),
        ('', ''),
        ('总样本数', str(total)),
        ('正常/异常', f'{tn+fp} / {fn+tp}'),
    ]

    y = 1.0
    for name, value in metrics_text:
        if name:
            ax2.text(0.05, y, name, fontsize=10, color='#555', va='center')
            ax2.text(0.55, y, value, fontsize=10, fontweight='bold',
                     color='#1e40af', va='center')
        y -= 0.07

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1.05)
    ax2.set_title('性能指标汇总', fontsize=12, fontweight='bold', pad=10)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  [OK] 已保存: {save_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='混淆矩阵生成工具')
    parser.add_argument('--model', '-m', type=str, default='all',
                        choices=['fre', 'patchcore', 'draem', 'padim', 'all'])
    parser.add_argument('--data_path', '-d', type=str, default='./data')
    parser.add_argument('--category', '-c', type=str, default='all')
    parser.add_argument('--threshold', '-t', type=str, default='optimal',
                        help='阈值: "optimal"(Youden J) 或具体数值')
    args = parser.parse_args()

    data_path = str(Path(args.data_path).resolve())

    # 确定数据集列表
    if args.category == 'all':
        data_dir = Path(data_path)
        categories = [d.name for d in sorted(data_dir.iterdir())
                      if d.is_dir() and (d / 'train').exists()]
    else:
        categories = args.category.split(',')

    # 确定模型列表
    models = SUPPORTED_MODELS if args.model == 'all' else [args.model]

    detector = AnomalyDetector()
    total = len(models) * len(categories)

    print("=" * 70)
    print("混淆矩阵生成工具")
    print(f"  模型: {[m.upper() for m in models]}")
    print(f"  数据集: {categories}")
    print(f"  任务数: {total}")
    print("=" * 70)

    count = 0
    for model_name in models:
        for dataset in categories:
            count += 1
            print(f"\n[{count}/{total}] {model_name.upper()} @ {dataset}")

            # 加载最优阈值
            threshold = get_threshold(model_name, dataset)
            if threshold is None or threshold == 0.5:
                # 从结果 JSON 读取
                result_file = PROJECT_ROOT / 'results' / 'comparison' / f'{model_name}_{dataset}_results.json'
                if result_file.exists():
                    try:
                        data = json.loads(result_file.read_text())
                        threshold = data.get('optimal_threshold',
                                             data.get('metrics', {}).get('optimal_threshold', 0.5))
                    except Exception:
                        threshold = 0.5

            # 加载已有指标
            result_file = PROJECT_ROOT / 'results' / 'comparison' / f'{model_name}_{dataset}_results.json'
            metrics = {}
            if result_file.exists():
                try:
                    data = json.loads(result_file.read_text())
                    metrics = data.get('metrics', data)
                except Exception:
                    pass

            # 收集预测分数
            scores, labels = collect_predictions(detector, model_name, dataset, data_path)

            if not scores:
                print(f"  [SKIP] 无有效预测数据")
                continue

            # 计算混淆矩阵
            cm = compute_confusion_matrix(scores, labels, threshold)

            # 保存 JSON
            cm_data = {
                'model': model_name,
                'category': dataset,
                'threshold': threshold,
                'confusion_matrix': cm,
                'total': len(scores),
                'accuracy': (cm['TP'] + cm['TN']) / len(scores),
                'precision': cm['TP'] / (cm['TP'] + cm['FP']) if (cm['TP'] + cm['FP']) > 0 else 0,
                'recall': cm['TP'] / (cm['TP'] + cm['FN']) if (cm['TP'] + cm['FN']) > 0 else 0,
            }
            json_path = OUTPUT_DIR / f'{model_name}_{dataset}_confusion.json'
            json_path.parent.mkdir(parents=True, exist_ok=True)
            json_path.write_text(json.dumps(cm_data, indent=2, ensure_ascii=False),
                                 encoding='utf-8')

            # 生成可视化
            png_path = OUTPUT_DIR / f'{model_name}_{dataset}_confusion.png'
            plot_confusion_matrix(cm, model_name, dataset, threshold, metrics, png_path)

    print(f"\n{'=' * 70}")
    print(f"完成! 输出目录: {OUTPUT_DIR}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
