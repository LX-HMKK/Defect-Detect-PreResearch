#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
异常热力图后处理优化评测脚本

对已有模型的 anomaly maps 应用不同的后处理参数组合，
比较 PRO/Pixel AUROC 的改善幅度，生成最优配置推荐。

用法:
    # 对所有模型/所有数据集评测默认预设
    python tools/run_post_process_eval.py -m all -c all -d ./data

    # 网格搜索最优后处理参数（耗时较长）
    python tools/run_post_process_eval.py -m patchcore -c bottle -d ./data --grid_search

    # 仅评测几个预设配置
    python tools/run_post_process_eval.py -m all -c bottle -d ./data --presets

    # 对特定模型/数据集评测并保存结果
    python tools/run_post_process_eval.py -m patchcore -c region2 -d ./data --save
"""

import io
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(PROJECT_ROOT))
from modules._runtime import configure_runtime_temp, resolve_project_path
configure_runtime_temp()

from modules.algorithm import SUPPORTED_MODELS, find_latest_checkpoint
from modules.algorithm.trainer import AnomalyDetectionTrainer
from modules.config import get
from modules.evaluation.metrics import MetricsEvaluator
from modules.evaluation.post_processor import (
    AnomalyMapProcessor,
    PostProcessConfig,
    PRESET_CONFIGS,
    process_anomaly_maps,
)
from anomalib.engine import Engine


def get_all_categories(data_path: str) -> List[str]:
    """自动发现数据目录中的所有类别"""
    data_dir = Path(data_path)
    if not data_dir.exists():
        return []
    categories: List[str] = []
    for item in sorted(data_dir.iterdir()):
        if item.is_file() or item.name.startswith('.'):
            continue
        if (item / 'train').exists():
            categories.append(item.name)
    return categories


def collect_anomaly_maps(
    model_name: str,
    category: str,
    data_path: str,
    output_dir: str,
    device: str = 'auto',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    加载已训练模型，对测试集执行推理，收集异常热力图和真实标注。

    Returns:
        (anomaly_maps, gt_masks, anomaly_scores, gt_labels)
        其中 anomaly_maps: (N, H, W), gt_masks: (N, H, W)
    """
    trainer = AnomalyDetectionTrainer(
        model_name=model_name,
        data_path=data_path,
        category=category,
        output_dir=output_dir,
        device=device,
        seed=42,
    )
    trainer.setup()

    ckpt_path = find_latest_checkpoint(output_dir, model_name, category)
    if ckpt_path is None:
        raise FileNotFoundError(
            f"未找到 checkpoint: model={model_name}, category={category}"
        )

    engine = Engine(
        accelerator=device,
        devices=1,
        default_root_dir=str(
            resolve_project_path(get('paths.temp_dir', './.cache'))
            / "lightning_logs"
            / model_name
        ),
        logger=False,
        enable_progress_bar=False,
    )

    predictions = engine.predict(
        datamodule=trainer.datamodule,
        model=trainer.model,
        ckpt_path=str(ckpt_path),
    )

    anomaly_maps_list = []
    gt_masks_list = []
    anomaly_scores_list = []
    gt_labels_list = []

    for pred in predictions:
        # 异常热力图 — PatchCore 返回多尺度特征图 (C, H, W)，聚合为单通道
        amap = pred.anomaly_map
        if isinstance(amap, torch.Tensor):
            amap = amap.cpu().numpy()
        if amap.ndim == 3 and amap.shape[0] > 1:
            # 多通道：沿通道取均值 → 单通道 (H, W)
            amap = amap.mean(axis=0)
        elif amap.ndim == 3 and amap.shape[0] == 1:
            amap = amap[0]
        anomaly_maps_list.append(amap.astype(np.float32))

        # 真实掩膜
        mask = pred.gt_mask
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        if mask.ndim == 3 and mask.shape[0] > 1:
            mask = mask.mean(axis=0)
        elif mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask[0]
        gt_masks_list.append(mask.astype(np.float32))

        # 图像级得分
        score = float(pred.pred_score.cpu().max().item())
        anomaly_scores_list.append(score)

        # 图像级标签
        gt_label = pred.gt_label.cpu()
        if gt_label.numel() == 1:
            label = bool(gt_label.item())
        else:
            label = bool(gt_label.flatten()[0].item())
        gt_labels_list.append(label)

    # 统一尺寸（anomalib 输出可能不一致）
    target_h = 256
    target_w = 256
    resized_maps = []
    resized_masks = []
    for amap, mask in zip(anomaly_maps_list, gt_masks_list):
        if amap.shape != (target_h, target_w):
            amap_resized = cv2.resize(amap, (target_w, target_h))
        else:
            amap_resized = amap
        if mask.shape != (target_h, target_w):
            mask_resized = cv2.resize(mask, (target_w, target_h),
                                       interpolation=cv2.INTER_NEAREST)
        else:
            mask_resized = mask
        resized_maps.append(amap_resized)
        resized_masks.append(mask_resized)

    return (
        np.array(resized_maps, dtype=np.float32),
        np.array(resized_masks, dtype=np.float32),
        np.array(anomaly_scores_list, dtype=np.float64),
        np.array(gt_labels_list, dtype=np.int32),
    )


def evaluate_post_processing(
    anomaly_maps: np.ndarray,
    gt_masks: np.ndarray,
    anomaly_scores: np.ndarray,
    gt_labels: np.ndarray,
    processor: AnomalyMapProcessor,
    evaluator: MetricsEvaluator,
) -> Dict:
    """对给定后处理器评测全部指标"""
    processed = processor.process(anomaly_maps)

    pixel_auroc = evaluator.compute_pixel_auroc(processed, gt_masks)
    pro = evaluator.compute_pro(processed, gt_masks)

    return {
        'pixel_AUROC': float(pixel_auroc),
        'pixel_PRO': float(pro),
    }


def run_presets_evaluation(
    model_name: str,
    category: str,
    data_path: str,
    output_dir: str,
    device: str = 'auto',
) -> List[Dict]:
    """运行预设配置评测"""
    evaluator = MetricsEvaluator()

    print(f"\n  加载模型并推理: {model_name.upper()} @ {category}")
    anomaly_maps, gt_masks, anomaly_scores, gt_labels = collect_anomaly_maps(
        model_name, category, data_path, output_dir, device
    )

    # 基准（无后处理）
    results = []
    raw_pro = evaluator.compute_pro(anomaly_maps, gt_masks)
    raw_pixel_auroc = evaluator.compute_pixel_auroc(anomaly_maps, gt_masks)

    results.append({
        'config': 'raw (无后处理)',
        'pixel_AUROC': float(raw_pixel_auroc),
        'pixel_PRO': float(raw_pro),
        'pro_delta': 0.0,
    })
    print(f"    {'raw':<20}  Pixel AUROC={raw_pixel_auroc*100:.2f}%  PRO={raw_pro*100:.2f}%")

    # 评测所有预设
    for preset_name in ['light', 'medium', 'strong', 'smooth_only', 'morph_only']:
        config = PRESET_CONFIGS[preset_name]
        processor = AnomalyMapProcessor(config)
        metrics = evaluate_post_processing(
            anomaly_maps, gt_masks, anomaly_scores, gt_labels,
            processor, evaluator,
        )
        delta = metrics['pixel_PRO'] - raw_pro
        results.append({
            'config': preset_name,
            'pixel_AUROC': metrics['pixel_AUROC'],
            'pixel_PRO': metrics['pixel_PRO'],
            'pro_delta': delta,
            'params': config.to_dict(),
        })
        delta_str = f"(+{delta*100:.2f}%)" if delta > 0 else f"({delta*100:.2f}%)"
        print(f"    {preset_name:<20}  Pixel AUROC={metrics['pixel_AUROC']*100:.2f}%  "
              f"PRO={metrics['pixel_PRO']*100:.2f}%  {delta_str}")

    return results


def run_grid_search(
    model_name: str,
    category: str,
    data_path: str,
    output_dir: str,
    device: str = 'auto',
    max_configs: int = 80,
) -> List[Dict]:
    """
    网格搜索最优后处理参数

    在合理参数范围内搜索，但限制组合数避免过度耗时的搜索。
    跳过已知无效的大范围组合。
    """
    from modules.evaluation.post_processor import grid_search_configs

    evaluator = MetricsEvaluator()

    print(f"\n  加载模型并推理: {model_name.upper()} @ {category}")
    anomaly_maps, gt_masks, anomaly_scores, gt_labels = collect_anomaly_maps(
        model_name, category, data_path, output_dir, device
    )

    raw_pro = evaluator.compute_pro(anomaly_maps, gt_masks)
    raw_pixel_auroc = evaluator.compute_pixel_auroc(anomaly_maps, gt_masks)
    print(f"    基准 (raw): Pixel AUROC={raw_pixel_auroc*100:.2f}%  PRO={raw_pro*100:.2f}%")

    configs = grid_search_configs()
    print(f"    待评测配置数: {len(configs)}")

    results = [{
        'config': 'raw',
        'pixel_AUROC': float(raw_pixel_auroc),
        'pixel_PRO': float(raw_pro),
        'pro_delta': 0.0,
    }]

    # 网格搜索
    for i, config in enumerate(configs):
        processor = AnomalyMapProcessor(config)
        metrics = evaluate_post_processing(
            anomaly_maps, gt_masks, anomaly_scores, gt_labels,
            processor, evaluator,
        )
        delta = metrics['pixel_PRO'] - raw_pro
        results.append({
            'config': config.label(),
            'pixel_AUROC': metrics['pixel_AUROC'],
            'pixel_PRO': metrics['pixel_PRO'],
            'pro_delta': delta,
            'params': config.to_dict(),
        })

        if (i + 1) % 20 == 0:
            best_so_far = max(results[1:], key=lambda r: r['pixel_PRO'])
            print(f"    [{i+1}/{len(configs)}] 当前最优: {best_so_far['config']} "
                  f"PRO={best_so_far['pixel_PRO']*100:.2f}% "
                  f"(+{(best_so_far['pixel_PRO']-raw_pro)*100:.2f}%)")

    # 排序：按 PRO 降序
    grid_results = sorted(results, key=lambda r: r['pixel_PRO'], reverse=True)
    return grid_results


def generate_report(all_results: Dict[str, List[Dict]], output_path: Path):
    """生成 Markdown 对比报告"""
    lines = []
    lines.append("# 异常热力图后处理优化报告\n")
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    lines.append("## 后处理策略\n")
    lines.append("对模型输出的异常热力图依次应用以下处理：\n")
    lines.append("1. **高斯平滑** — 抑制热力图噪声，减少孤立假阳性像素")
    lines.append("2. **形态学闭运算** — 填充缺陷区域内部的小空洞")
    lines.append("3. **适度膨胀** — 扩展高值区域边界，提升区域 Overlap")
    lines.append("4. **小区域过滤** — 移除面积过小的噪声连通分量\n")

    lines.append("## 预设配置\n")
    lines.append("| 预设 | Gσ | 闭运算 r | min_area | 膨胀 r |")
    lines.append("|------|:---:|:---:|:---:|:---:|")
    for name in ['light', 'medium', 'strong', 'smooth_only', 'morph_only']:
        cfg = PRESET_CONFIGS[name]
        lines.append(f"| {name} | {cfg.gaussian_sigma} | {cfg.closing_radius} | "
                     f"{cfg.min_area} | {cfg.dilate_radius} |")
    lines.append("")

    # 按数据集分组
    for combo_key, results in sorted(all_results.items()):
        if not results:
            continue
        model, cat = combo_key.split('@')
        lines.append(f"### {model.upper()} @ {cat}\n")
        lines.append("| 配置 | Pixel AUROC | PRO | Δ PRO |")
        lines.append("|------|:---:|:---:|:---:|")

        raw = next((r for r in results if r['config'] in ('raw', 'raw (无后处理)')), None)
        for r in results[:15]:  # 只显示前15个
            pro_str = f"{r['pixel_PRO']*100:.2f}%"
            p_auroc_str = f"{r['pixel_AUROC']*100:.2f}%"
            delta = r['pro_delta']
            delta_str = f"+{delta*100:.2f}%" if delta > 0 else f"{delta*100:.2f}%"
            highlight = "**" if r['pixel_PRO'] == max(rr['pixel_PRO'] for rr in results) else ""
            lines.append(f"| {highlight}{r['config']}{highlight} | "
                         f"{p_auroc_str} | {pro_str} | {delta_str} |")
        lines.append("")

    # 总结：每个模型/数据集的最优配置
    lines.append("## 最优配置推荐\n")
    lines.append("| 模型 | 数据集 | 最优配置 | 原始 PRO | 优化后 PRO | 提升 |")
    lines.append("|------|--------|----------|:---:|:---:|:---:|")

    for combo_key, results in sorted(all_results.items()):
        if not results or len(results) < 2:
            continue
        model, cat = combo_key.split('@')
        raw = next((r for r in results if r['config'] in ('raw', 'raw (无后处理)')), None)
        if not raw:
            continue
        # 找最优（非 raw）
        best = max(
            [r for r in results if r['config'] not in ('raw', 'raw (无后处理)')],
            key=lambda r: r['pixel_PRO']
        )
        improvement = best['pixel_PRO'] - raw['pixel_PRO']
        lines.append(
            f"| {model.upper()} | {cat} | {best['config']} | "
            f"{raw['pixel_PRO']*100:.2f}% | {best['pixel_PRO']*100:.2f}% | "
            f"+{improvement*100:.2f}% |"
        )

    lines.append(f"\n---\n*报告由 run_post_process_eval.py 自动生成*")

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"\n报告已保存: {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='异常热力图后处理优化评测'
    )
    parser.add_argument('--model', '-m', type=str, default='all')
    parser.add_argument('--data_path', '-d', type=str, default='./data')
    parser.add_argument('--category', '-c', type=str, default='all')
    parser.add_argument('--output_dir', '-o', type=str, default='./results')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--grid_search', action='store_true',
                        help='网格搜索最优参数（较耗时）')
    parser.add_argument('--presets', action='store_true',
                        help='仅评测预设配置（light/medium/strong等）')
    parser.add_argument('--save', action='store_true',
                        help='保存评测结果 JSON')
    args = parser.parse_args()

    models_to_eval = SUPPORTED_MODELS if args.model == 'all' else [args.model]
    categories_to_eval = (
        get_all_categories(args.data_path) if args.category == 'all'
        else [args.category]
    )

    if not categories_to_eval:
        print("[ERROR] 未找到任何有效的数据类别")
        raise SystemExit(1)

    print("=" * 70)
    print("异常热力图后处理优化评测")
    print(f"  模型: {', '.join([m.upper() for m in models_to_eval])}")
    print(f"  数据集: {', '.join(categories_to_eval)}")
    print(f"  模式: {'网格搜索' if args.grid_search else '预设评测'}")
    print("=" * 70)

    all_results: Dict[str, List[Dict]] = {}
    failed: List[str] = []

    for cat_idx, category in enumerate(categories_to_eval, 1):
        for model_name in models_to_eval:
            combo = f"{model_name}@{category}"
            print(f"\n[{combo}]")

            try:
                if args.grid_search:
                    results = run_grid_search(
                        model_name, category, args.data_path, args.output_dir, args.device
                    )
                else:
                    results = run_presets_evaluation(
                        model_name, category, args.data_path, args.output_dir, args.device
                    )
                    # 找出最优配置
                    best = max(
                        [r for r in results if r['config'] != 'raw (无后处理)'],
                        key=lambda r: r['pixel_PRO']
                    )
                    print(f"  → 最优: {best['config']} (PRO 提升 +{(best['pixel_PRO'] - results[0]['pixel_PRO'])*100:.2f}%)")

                all_results[combo] = results
            except Exception as e:
                print(f"  [FAIL] {e}")
                import traceback
                traceback.print_exc()
                failed.append(combo)

    # 保存 JSON
    if args.save:
        save_dir = Path(args.output_dir) / 'comparison' / 'post_process'
        save_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        json_path = save_dir / f'post_process_results_{timestamp}.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\n[JSON] 结果已保存: {json_path}")

    # 生成报告
    report_path = Path(args.output_dir) / 'comparison' / 'post_process_report.md'
    generate_report(all_results, report_path)

    # 总结
    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    for combo, results in sorted(all_results.items()):
        if not results or len(results) < 2:
            continue
        model, cat = combo.split('@')
        raw = next((r for r in results if r['config'] in ('raw', 'raw (无后处理)')), None)
        best = max(
            [r for r in results if r['config'] not in ('raw', 'raw (无后处理)')],
            key=lambda r: r['pixel_PRO']
        )
        delta = (best['pixel_PRO'] - raw['pixel_PRO']) * 100 if raw else 0
        print(f"  {model.upper():<12} @ {cat:<10}  "
              f"PRO {raw['pixel_PRO']*100:.2f}% → {best['pixel_PRO']*100:.2f}%  "
              f"(+{delta:.2f}%)  [{best['config']}]")

    if failed:
        print(f"\n失败组合: {', '.join(failed)}")


if __name__ == '__main__':
    main()
