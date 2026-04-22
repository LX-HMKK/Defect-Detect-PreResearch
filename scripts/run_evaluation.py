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

def _configure_runtime_temp() -> None:
    temp_dir = PROJECT_ROOT / "temp"
    pycache_dir = temp_dir / "pycache"
    temp_dir.mkdir(exist_ok=True)
    pycache_dir.mkdir(exist_ok=True)
    sys.pycache_prefix = str(pycache_dir)
    os.environ["PYTHONPYCACHEPREFIX"] = str(pycache_dir)


_configure_runtime_temp()

sys.path.insert(0, str(PROJECT_ROOT))


def print_banner() -> None:
    print()
    print("=" * 70)
    print("Model Evaluation")
    print("Check 4 core metrics from saved results files")
    print("=" * 70)


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
        choices=["fre", "patchcore", "draem", "all"],
        help="Model name.",
    )
    parser.add_argument(
        "--category",
        "-c",
        type=str,
        default="bottle",
        help="Category name.",
    )
    return parser.parse_args()


def main() -> None:
    print_banner()
    args = parse_args()

    models_to_eval = ["fre", "patchcore", "draem"] if args.model == "all" else [args.model]
    print()
    print("Config")
    print("-" * 70)
    print(f"  models: {', '.join([m.upper() for m in models_to_eval])}")
    print(f"  category: {args.category}")
    print(f"  results_dir: {args.results_dir}")
    print("-" * 70)

    from modules.evaluation.metrics import load_and_evaluate

    passed = []
    failed = []

    for i, model_name in enumerate(models_to_eval, 1):
        print(f"\n[{i}/{len(models_to_eval)}] Evaluate: {model_name.upper()}")
        print("=" * 70)
        try:
            ok = load_and_evaluate(args.results_dir, model_name, args.category)
            if ok:
                passed.append(model_name)
                print(f"Done: {model_name.upper()}")
            else:
                failed.append(model_name)
                print(f"Failed: {model_name.upper()} (missing/invalid result file)")
        except Exception as exc:
            failed.append(model_name)
            print(f"Failed: {model_name.upper()} ({exc})")

    print()
    print("=" * 70)
    print("Evaluation Summary")
    print("=" * 70)
    print(f"  passed: {len(passed)} -> {', '.join([m.upper() for m in passed]) if passed else '-'}")
    print(f"  failed: {len(failed)} -> {', '.join([m.upper() for m in failed]) if failed else '-'}")
    print("=" * 70)

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
