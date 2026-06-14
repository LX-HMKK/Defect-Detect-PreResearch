#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
参数消融实验脚本

对关键超参数进行受控网格搜索，论证选参合理性。

消融维度:
    PatchCore: coreset_sampling_ratio (0.01/0.05/0.1/0.2)
    PaDiM: backbone (resnet18/wide_resnet50_2)
    FRE: latent_dim (100/220/500)

用法:
    python tools/run_ablation.py -m all -c bottle -d ./data
"""

import io
import json
import os
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(PROJECT_ROOT))

from modules._runtime import resolve_project_path
from modules.algorithm.trainer import AnomalyDetectionTrainer
from modules.config import get


def modify_yaml_and_run(model_name: str, category: str, data_path: str,
                        overrides: Dict[str, Any]) -> Dict[str, Any]:
    """
    临时修改模型 YAML 配置 → 训练 → 评估 → 恢复原配置。
    返回评估指标字典。
    """
    config_path = PROJECT_ROOT / 'configs' / f'{model_name}.yaml'
    original = yaml.safe_load(config_path.read_text(encoding='utf-8')) or {}

    # 深度覆盖
    modified = json.loads(json.dumps(original))  # deep copy
    for key, val in overrides.items():
        if 'model' in modified and 'init_args' in modified['model']:
            modified['model']['init_args'][key] = val

    temp_path = config_path.with_suffix('.ablation.tmp.yaml')
    temp_path.write_text(yaml.dump(modified, allow_unicode=True), encoding='utf-8')

    try:
        temp_dir = resolve_project_path(get('paths.temp_dir', './.cache'))
        trainer = AnomalyDetectionTrainer(
            model_name=model_name,
            data_path=data_path,
            category=category,
            output_dir=str(temp_dir / 'ablation' / model_name),
            device='auto',
            seed=42,
            config_path=temp_path,
        )
        trainer.setup()
        trainer.train(max_epochs=1 if model_name in ('patchcore', 'padim') else None)
        # patchcore/padim 仅构建记忆库/高斯模型，无需多轮训练，1 epoch 为占位值
        metrics = trainer.evaluate()
        # 添加消融元信息
        for key, val in overrides.items():
            metrics[f'ablation_{key}'] = val
        return metrics
    finally:
        # 清理临时文件
        if temp_path.exists():
            temp_path.unlink()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='参数消融实验')
    parser.add_argument('--model', '-m', type=str, default='all')
    parser.add_argument('--data_path', '-d', type=str, default='./data')
    parser.add_argument('--category', '-c', type=str, default='bottle')
    args = parser.parse_args()

    data_path = str(Path(args.data_path).resolve())

    experiments = []

    # PatchCore 消融
    if args.model in ('patchcore', 'all'):
        experiments.extend([
            ('patchcore', 'coreset_sampling_ratio', v, {'coreset_sampling_ratio': v})
            for v in [0.01, 0.05, 0.1, 0.2]
        ])

    # PaDiM 消融
    if args.model in ('padim', 'all'):
        for backbone in ['resnet18', 'wide_resnet50_2']:
            # wide_resnet50_2 需要调整 layers 避免维度爆炸
            layers = ['layer1', 'layer2', 'layer3'] if backbone == 'resnet18' else ['layer2', 'layer3']
            experiments.append(
                ('padim', 'backbone', backbone,
                 {'backbone': backbone, 'layers': layers})
            )

    # FRE 消融
    if args.model in ('fre', 'all'):
        experiments.extend([
            ('fre', 'latent_dim', v, {'latent_dim': v})
            for v in [100, 220, 500]
        ])

    print("=" * 70)
    print("参数消融实验")
    print(f"  数据集: {args.category}")
    print(f"  实验数: {len(experiments)}")
    print("=" * 70)

    results = []
    for i, (model, param, val, overrides) in enumerate(experiments):
        print(f"\n[{i+1}/{len(experiments)}] {model.upper()} | {param}={val}")
        try:
            metrics = modify_yaml_and_run(model, args.category, data_path, overrides)
            results.append({
                'model': model,
                'param': param,
                'value': val,
                'image_AUROC': metrics.get('image_AUROC', 0),
                'image_AUPR': metrics.get('image_AUPR', 0),
                'pixel_AUROC': metrics.get('pixel_AUROC', 0),
                'pixel_PRO': metrics.get('pixel_PRO', 0),
            })
            print(f"  [OK] AUROC={metrics.get('image_AUROC', 0)*100:.1f}%")
        except Exception as e:
            print(f"  [ERROR] {e}")

    # 保存
    output_path = PROJECT_ROOT / 'results' / 'comparison' / 'ablation_results.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False),
                           encoding='utf-8')

    # 打印汇总表
    if results:
        print(f"\n{'=' * 70}")
        print("消融实验结果")
        print(f"{'=' * 70}")
        print(f"{'模型':<12} {'参数':<28} {'值':<15} {'AUROC':<10} {'PixelAUROC':<12}")
        print("-" * 77)
        for r in results:
            auroc = f"{r['image_AUROC']*100:.1f}%"
            p_auroc = f"{r['pixel_AUROC']*100:.1f}%"
            print(f"{r['model'].upper():<12} {r['param']:<28} {str(r['value']):<15} "
                  f"{auroc:<10} {p_auroc:<12}")

    print(f"\n结果已保存: {output_path}")


if __name__ == '__main__':
    main()
