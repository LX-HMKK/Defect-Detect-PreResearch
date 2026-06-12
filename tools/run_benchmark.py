#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
推理性能基准测试脚本（修复版）

使用 anomalib Engine.predict() 正确测量各算法的推理速度和资源占用。

用法:
    python tools/run_benchmark.py -m all -c bottle -d ./data
"""

import io
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import tempfile
import torch

from anomalib.engine import Engine
from anomalib.data import PredictDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(PROJECT_ROOT))

from modules.algorithm.trainer import AnomalyDetectionTrainer
from modules.algorithm import SUPPORTED_MODELS
from modules.ui.demo import AnomalyDetector


def get_test_image_path(data_path: str, dataset: str) -> Optional[Path]:
    """从测试集中获取第一张图片路径"""
    test_dir = Path(data_path) / dataset / 'test'
    # 优先用 good 目录
    for subdir in ['good'] + sorted([d.name for d in test_dir.iterdir()
                                      if d.is_dir() and d.name != 'good']):
        sd = test_dir / subdir
        if not sd.exists():
            continue
        for f in sorted(sd.glob('*')):
            if f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp'):
                return f
    return None


def run_benchmark(model_name: str, category: str, data_path: str,
                  device: str = 'auto') -> Optional[dict]:
    """对单个模型+数据集组合运行基准测试"""

    resolved_device = 'cuda' if (device == 'auto' and torch.cuda.is_available()) else device

    print(f"\n  [{model_name.upper()}] 加载模型...")

    # 使用 trainer 创建模型和数据模块
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
        trainer.train()
        model = trainer.model

    if model is None:
        print(f"  [FAIL] 无法创建模型")
        return None

    # 参数量
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    # GPU 显存 — 模型加载后
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        model.to(resolved_device)

    # 获取测试图片
    img_path = get_test_image_path(data_path, category)
    if img_path is None:
        print(f"  [FAIL] 未找到测试图片")
        return None

    # 使用 detector 预测（通过 Engine.predict 正确方式）
    detector = AnomalyDetector()
    success, _ = detector.load_model(model_name, category)
    if not success:
        print(f"  [FAIL] 模型加载失败")
        return None

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  [FAIL] 无法读取图片: {img_path}")
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 预热
    print(f"  [BENCH] 预热 (3次)...")
    for _ in range(3):
        detector.predict(img)

    # 正式测量
    print(f"  [BENCH] 计时 (10次)...")
    times = []
    for _ in range(10):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        detector.predict(img)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    # 显存峰值
    mem_mb = 0
    if torch.cuda.is_available():
        mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    avg_ms = np.mean(times)
    std_ms = np.std(times)

    print(f"  [OK] 平均: {avg_ms:.1f}ms, 峰值显存: {mem_mb:.0f}MB, 参数量: {params/1e6:.1f}M")

    return {
        'model': model_name.upper(),
        'category': category,
        'device': resolved_device,
        'trainable_params': params,
        'total_params': total_params,
        'trainable_params_m': round(params / 1e6, 1),
        'total_params_m': round(total_params / 1e6, 1),
        'avg_inference_ms': round(avg_ms, 1),
        'std_inference_ms': round(std_ms, 1),
        'min_inference_ms': round(np.min(times), 1),
        'max_inference_ms': round(np.max(times), 1),
        'peak_gpu_memory_mb': round(mem_mb, 0),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='模型推理性能基准测试')
    parser.add_argument('--model', '-m', type=str, default='all',
                        choices=['fre', 'patchcore', 'draem', 'padim', 'all'])
    parser.add_argument('--data_path', '-d', type=str, default='./data')
    parser.add_argument('--category', '-c', type=str, default='bottle')
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    data_path = str(Path(args.data_path).resolve())

    if args.category == 'all':
        data_dir = Path(data_path)
        categories = [d.name for d in sorted(data_dir.iterdir())
                      if d.is_dir() and (d / 'train').exists()]
    else:
        categories = args.category.split(',')

    models = SUPPORTED_MODELS if args.model == 'all' else [args.model]

    print("=" * 70)
    print("模型推理性能基准测试")
    print(f"  模型: {[m.upper() for m in models]}")
    print(f"  类别: {categories}")
    print(f"  设备: {args.device}")
    print("=" * 70)

    results = []
    for model_name in models:
        print(f"\n[{model_name.upper()}]")
        for category in categories:
            try:
                result = run_benchmark(model_name, category, data_path, args.device)
                if result:
                    results.append(result)
            except Exception as e:
                print(f"  [ERROR] {model_name}/{category}: {e}")

    # 保存结果
    output_path = PROJECT_ROOT / 'results' / 'comparison' / 'benchmark.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False),
                           encoding='utf-8')

    # 打印汇总
    if results:
        print(f"\n{'=' * 70}")
        print(f"推理速度汇总")
        print(f"{'=' * 70}")
        print(f"{'模型':<12} {'参数(M)':<10} {'平均(ms)':<10} {'显存(MB)':<10}")
        print("-" * 42)
        for r in results:
            print(f"{r['model']:<12} {r['trainable_params_m']:<10.1f} "
                  f"{r['avg_inference_ms']:<10.1f} {r['peak_gpu_memory_mb']:<10.0f}")

    print(f"\n结果已保存: {output_path}")


if __name__ == '__main__':
    main()
