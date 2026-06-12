#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Entry script: evaluate saved model results.
Usage:
    python scripts/run_evaluation.py --model all --category bottle
"""

import argparse
import io
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(PROJECT_ROOT))
from modules._runtime import configure_runtime_temp

configure_runtime_temp()


def print_banner() -> None:
    print()
    print("=" * 70)
    print("Model Evaluation")
    print("Check 4 core metrics from saved results files")
    print("=" * 70)


def get_all_categories(data_path: str) -> list[str]:
    """自动发现数据目录中的所有类别（包含 train/ 子目录的文件夹）"""
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate saved metrics results."
    )
    parser.add_argument(
        "--results_dir",
        "-r",
        type=str,
        default="./results",
        help="Results root directory.",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="all",
        choices=["fre", "patchcore", "draem", "padim", "all"],
        help="Model name.",
    )
    parser.add_argument(
        "--category",
        "-c",
        type=str,
        default="bottle",
        help="Category name (or all).",
    )
    return parser.parse_args()


def main() -> None:
    print_banner()
    args = parse_args()

    models_to_eval = ["fre", "patchcore", "draem", "padim"] if args.model == "all" else [args.model]
    categories_to_eval = get_all_categories(args.results_dir) if args.category == "all" else [args.category]
    if args.category == "all":
        # For evaluation, check results/comparison for JSON files instead
        comparison_dir = Path(args.results_dir) / "comparison"
        if comparison_dir.exists():
            cats_from_results: set[str] = set()
            for f in comparison_dir.glob("*_results.json"):
                # Extract category from filename like "patchcore_bottle_results.json"
                name = f.stem.replace("_results", "")
                for m in models_to_eval:
                    if name.startswith(m + "_"):
                        cats_from_results.add(name[len(m) + 1:])
            if cats_from_results:
                categories_to_eval = sorted(cats_from_results)

    if not categories_to_eval:
        print("[ERROR] No categories found.")
        raise SystemExit(1)

    print()
    print("Config")
    print("-" * 70)
    print(f"  models: {', '.join([m.upper() for m in models_to_eval])}")
    print(f"  categories: {', '.join(categories_to_eval)}")
    print(f"  results_dir: {args.results_dir}")
    print("-" * 70)

    from modules.evaluation import load_and_evaluate

    all_passed = []
    all_failed = []

    for cat_idx, category in enumerate(categories_to_eval, 1):
        if len(categories_to_eval) > 1:
            print(f"\n{'=' * 70}")
            print(f"[{cat_idx}/{len(categories_to_eval)}] Category: {category}")
            print(f"{'=' * 70}")

        for i, model_name in enumerate(models_to_eval, 1):
            print(f"\n[{i}/{len(models_to_eval)}] Evaluate: {model_name.upper()} @ {category}")
            print("-" * 70)
            try:
                ok = load_and_evaluate(args.results_dir, model_name, category)
                if ok:
                    all_passed.append(f"{model_name}@{category}")
                    print(f"Done: {model_name.upper()}")
                else:
                    all_failed.append(f"{model_name}@{category}")
                    print(f"Failed: {model_name.upper()} (missing/invalid result file)")
            except Exception as exc:
                all_failed.append(f"{model_name}@{category}")
                print(f"Failed: {model_name.upper()} ({exc})")

    print()
    print("=" * 70)
    print("Evaluation Summary")
    print("=" * 70)
    print(f"  passed: {len(all_passed)} -> {', '.join(all_passed) if all_passed else '-'}")
    print(f"  failed: {len(all_failed)} -> {', '.join(all_failed) if all_failed else '-'}")
    print("=" * 70)

    if all_failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
