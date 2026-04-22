#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Entry script: compute optimal threshold (Youden's J) with restored checkpoints.
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
    print("Threshold Computation")
    print("=" * 70)


def get_all_categories(data_path: str) -> list[str]:
    data_dir = Path(data_path)
    if not data_dir.exists():
        return []

    categories: list[str] = []
    for item in data_dir.iterdir():
        if item.is_file() or item.name.startswith("."):
            continue
        if (item / "train").exists():
            categories.append(item.name)
    return sorted(categories)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute optimal threshold from checkpoint predictions."
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="patchcore",
        choices=["fre", "patchcore", "draem", "all"],
        help="Model name.",
    )
    parser.add_argument(
        "--data_path",
        "-d",
        type=str,
        default="./data",
        help="Dataset root path.",
    )
    parser.add_argument(
        "--category",
        "-c",
        type=str,
        default="bottle",
        help="Category name (or all).",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default="./results",
        help="Output directory.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path. If omitted, auto-discover latest checkpoint for model+category.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Persist threshold result to results/comparison JSON.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Compute device (auto/cpu/cuda).",
    )
    return parser.parse_args()


def _resolve_checkpoint(
    model_name: str,
    category: str,
    output_dir: str,
    checkpoint_arg: str | None,
) -> Path:
    from modules.algorithm.trainer import find_latest_checkpoint

    if checkpoint_arg:
        ckpt_path = Path(checkpoint_arg)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        return ckpt_path

    latest_ckpt = find_latest_checkpoint(output_dir, model_name, category)
    if latest_ckpt is None:
        raise FileNotFoundError(
            "No checkpoint found for model/category. "
            f"model={model_name}, category={category}. "
            "Please train first or pass --checkpoint explicitly."
        )
    return latest_ckpt


def compute_threshold(
    model_name: str,
    data_path: str,
    category: str,
    output_dir: str,
    device: str,
    save: bool,
    checkpoint_arg: str | None = None,
) -> float | None:
    from anomalib.engine import Engine
    from modules.algorithm.trainer import AnomalyDetectionTrainer

    print(f"\n{'=' * 70}")
    print(f"Compute threshold: {model_name.upper()} + {category}")
    print(f"{'=' * 70}")

    try:
        trainer = AnomalyDetectionTrainer(
            model_name=model_name,
            data_path=data_path,
            category=category,
            output_dir=output_dir,
            device=device,
            seed=42,
        )

        ckpt_path = _resolve_checkpoint(model_name, category, output_dir, checkpoint_arg)
        trainer.setup()
        trainer.engine = Engine(
            accelerator=device,
            devices=1,
            default_root_dir=str(PROJECT_ROOT / "temp" / "lightning_logs" / model_name),
            logger=False,
            enable_progress_bar=False,
        )
        threshold = trainer._compute_optimal_threshold(checkpoint_path=str(ckpt_path))

        print("\nDone")
        print(f"  model: {model_name.upper()}")
        print(f"  category: {category}")
        print(f"  checkpoint: {ckpt_path}")
        print(f"  optimal_threshold: {threshold:.3f}")

        if save:
            if not isinstance(trainer.results, dict):
                trainer.results = {}
            trainer.results["optimal_threshold"] = threshold
            trainer._save_results()
            print(f"  saved: {output_dir}/comparison/{model_name}_{category}_results.json")

        return threshold

    except Exception as exc:
        print(f"\nFailed: {exc}")
        import traceback

        traceback.print_exc()
        return None


def main() -> None:
    print_banner()
    args = parse_args()

    models = ["fre", "patchcore", "draem"] if args.model == "all" else [args.model]
    if args.category == "all":
        categories = get_all_categories(args.data_path)
        if not categories:
            print("No valid category found under data path.")
            return
    else:
        categories = [args.category]

    if args.checkpoint and (len(models) > 1 or len(categories) > 1):
        raise ValueError(
            "--checkpoint can only be used with a single model and a single category."
        )

    print()
    print("Config")
    print("-" * 70)
    print(f"  models: {', '.join([m.upper() for m in models])}")
    print(f"  data_path: {args.data_path}")
    print(f"  categories: {', '.join(categories)}")
    print(f"  checkpoint: {args.checkpoint or 'auto-discover latest'}")
    print(f"  save: {args.save}")
    print(f"  device: {args.device}")
    print("-" * 70)

    results = []
    failed = []
    for category in categories:
        for model in models:
            threshold = compute_threshold(
                model_name=model,
                data_path=args.data_path,
                category=category,
                output_dir=args.output_dir,
                device=args.device,
                save=args.save,
                checkpoint_arg=args.checkpoint,
            )
            if threshold is not None:
                results.append(
                    {
                        "model": model,
                        "category": category,
                        "threshold": threshold,
                    }
                )
            else:
                failed.append({"model": model, "category": category})

    if results:
        print()
        print("=" * 70)
        print("Threshold Summary")
        print("=" * 70)
        print(f"{'model':<12} {'category':<12} {'threshold':<10}")
        print("-" * 70)
        for item in results:
            print(f"{item['model']:<12} {item['category']:<12} {item['threshold']:.3f}")
        print("=" * 70)

    if failed:
        print()
        failed_pairs = [f"{item['model']}:{item['category']}" for item in failed]
        print(f"Failed combinations: {', '.join(failed_pairs)}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
