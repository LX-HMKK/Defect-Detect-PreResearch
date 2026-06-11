#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
推理性能基准测试脚本

测量各算法在不同数据集上的推理速度、显存占用、参数量。
这是任务书要求的多维度对比的一部分。

用法:
    python tools/run_benchmark.py -m all -c all -d ./data
"""

import io
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(PROJECT_ROOT))


def get_all_categories(data_path: str) -> list[str]:
    data_dir = Path(data_path)
    if not data_dir.exists():
        return []
    categories: list[str] = []
    for item in sorted(data_dir.iterdir()):
        if item.is_file() or item.name.startswith('.'):
            continue
        if (item / 'train').exists():
            categories.append(item.name)
    return categories


def count_parameters(model) -> int:
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_inference_time(model, dataloader, device: str, warmup: int = 3, repeat: int = 10) -> dict:
    """测量推理时间"""
    model.eval()
    model.to(device)

    times = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if isinstance(batch, dict):
                image = batch.get('image', batch.get('input', None))
            elif isinstance(batch, (list, tuple)):
                image = batch[0]
            else:
                image = batch

            if image is None:
                continue

            image = image.to(device) if isinstance(image, torch.Tensor) else image

            # Warmup
            if batch_idx < warmup:
                _ = model(image)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                continue

            # Timed iterations
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(image)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)  # ms

            if len(times) >= repeat:
                break

    if not times:
        return {'avg_inference_ms': 0, 'std_inference_ms': 0, 'min_inference_ms': 0, 'max_inference_ms': 0}

    return {
        'avg_inference_ms': round(np.mean(times), 2),
        'std_inference_ms': round(np.std(times), 2),
        'min_inference_ms': round(np.min(times), 2),
        'max_inference_ms': round(np.max(times), 2),
    }


def measure_gpu_memory(model, dataloader, device: str) -> dict:
    """测量 GPU 显存占用"""
    if not torch.cuda.is_available():
        return {'peak_memory_mb': 0, 'note': 'CUDA not available'}

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    model.eval()
    model.to(device)

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, dict):
                image = batch.get('image', batch.get('input', None))
            elif isinstance(batch, (list, tuple)):
                image = batch[0]
            else:
                image = batch

            if image is None:
                continue

            image = image.to(device) if isinstance(image, torch.Tensor) else image
            _ = model(image)
            break

    peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    torch.cuda.empty_cache()

    return {'peak_memory_mb': round(peak_mb, 1)}


def run_benchmark(
    model_name: str,
    category: str,
    data_path: str,
    device: str = 'auto',
) -> dict:
    """对单个模型+数据集组合运行基准测试"""
    from modules.algorithm.trainer import AnomalyDetectionTrainer

    resolved_device = device
    if device == 'auto':
        resolved_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    trainer = AnomalyDetectionTrainer(
        model_name=model_name,
        data_path=data_path,
        category=category,
        output_dir='./temp',
        device=resolved_device,
        seed=42,
    )
    trainer.setup()
    model = trainer.model

    if model is None:
        # For PatchCore/PaDiM, model might be None until fit
        trainer.train()
        model = trainer.model

    params = count_parameters(model)

    test_loader = trainer.datamodule.test_dataloader()
    timing = measure_inference_time(model, test_loader, resolved_device)
    memory = measure_gpu_memory(model, test_loader, resolved_device)

    return {
        'model': model_name.upper(),
        'category': category,
        'device': resolved_device,
        'trainable_params': params,
        'trainable_params_m': round(params / 1e6, 1),
        'avg_inference_ms': timing['avg_inference_ms'],
        'std_inference_ms': timing['std_inference_ms'],
        'peak_gpu_memory_mb': memory.get('peak_memory_mb', 0),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='模型推理性能基准测试')
    parser.add_argument('--model', '-m', type=str, default='all',
                        choices=['fre', 'patchcore', 'draem', 'padim', 'all'])
    parser.add_argument('--data_path', '-d', type=str, default='./data')
    parser.add_argument('--category', '-c', type=str, default='bottle')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--output', '-o', type=str, default='./results/comparison/benchmark.json')
    args = parser.parse_args()

    from modules.algorithm.trainer import SUPPORTED_MODELS

    models_to_run = SUPPORTED_MODELS if args.model == 'all' else [args.model]
    categories_to_run = (
        get_all_categories(args.data_path) if args.category == 'all'
        else [args.category]
    )

    print()
    print("=" * 70)
    print("模型推理性能基准测试")
    print("=" * 70)
    print(f"  模型: {', '.join([m.upper() for m in models_to_run])}")
    print(f"  类别: {', '.join(categories_to_run)}")
    print(f"  设备: {args.device}")
    print("=" * 70)

    results = []

    for cat_idx, category in enumerate(categories_to_run, 1):
        for model_name in models_to_run:
            label = f"{model_name.upper()} @ {category}"
            print(f"\n[{len(results) + 1}/{len(models_to_run) * len(categories_to_run)}] {label}")
            print("-" * 50)

            try:
                result = run_benchmark(model_name, category, args.data_path, args.device)
                results.append(result)
                print(f"  参数: {result['trainable_params_m']}M | "
                      f"推理: {result['avg_inference_ms']}ms | "
                      f"显存: {result['peak_gpu_memory_mb']}MB")
            except Exception as e:
                print(f"  [ERROR] {e}")
                import traceback
                traceback.print_exc()

    # 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 打印汇总表
    if results:
        print(f"\n{'=' * 90}")
        print("性能汇总表")
        print(f"{'=' * 90}")
        print(f"{'Model':<12} {'Category':<12} {'Params(M)':>10} {'Infer(ms)':>10} {'GPU Mem(MB)':>12}")
        print("-" * 56)
        for r in results:
            print(f"{r['model']:<12} {r['category']:<12} "
                  f"{r['trainable_params_m']:>10.1f} {r['avg_inference_ms']:>10.2f} "
                  f"{r['peak_gpu_memory_mb']:>12.1f}")

    print(f"\n结果已保存: {output_path}")


if __name__ == '__main__':
    main()
