#!/usr/bin/env python
"""Data directory validation script — checks magic bytes, readability, structure, and class distribution.

Standalone: no dependency on project modules/ or torch/anomalib.
Usage: python tools/validate_data.py [--data-root ./data]
"""

import os
import sys
import struct
import argparse
from pathlib import Path
from datetime import datetime

KNOWN_DATASETS = {
    'bottle', 'carpet', 'region1', 'region2', 'region3', 'region5',
}
REFERENCED_DATASETS = {'bottle', 'carpet', 'region1', 'region2', 'region3', 'region4', 'region5'}

ERRORS: list[str] = []
WARNINGS: list[str] = []
INFOS: list[str] = []


def error(msg: str) -> None:
    ERRORS.append(msg)
    print(f"  [ERROR] {msg}")


def warn(msg: str) -> None:
    WARNINGS.append(msg)
    print(f"  [WARN]  {msg}")


def info(msg: str) -> None:
    INFOS.append(msg)
    print(f"  [INFO]  {msg}")


def check_magic_bytes(data_root: Path) -> None:
    print("\n[1/7] Checking magic bytes vs extension...")
    png_files = list(data_root.rglob("*.png"))
    bmp_count = 0
    png_count = 0
    other_count = 0
    for fpath in png_files:
        try:
            with open(fpath, "rb") as fh:
                magic = fh.read(2)
            if magic == b"\x89P":
                png_count += 1
            elif magic == b"BM":
                bmp_count += 1
                error(f"BMP file with .png extension: {fpath}")
            else:
                other_count += 1
                error(f"Unknown format (magic={magic!r}): {fpath}")
        except OSError as e:
            error(f"Cannot read {fpath}: {e}")
    info(f"Stats: {png_count} true PNGs, {bmp_count} BMP-as-PNG, {other_count} unknown")


def check_empty_files(data_root: Path) -> None:
    print("\n[2/7] Checking empty files...")
    empty_count = 0
    for fpath in data_root.rglob("*"):
        if fpath.is_file():
            try:
                if fpath.stat().st_size == 0:
                    error(f"Empty file: {fpath}")
                    empty_count += 1
            except OSError as e:
                warn(f"Cannot stat {fpath}: {e}")
    info(f"Found {empty_count} empty files")


def check_readability(data_root: Path) -> None:
    print("\n[3/7] Checking image readability (PIL + cv2)...")
    try:
        from PIL import Image
    except ImportError:
        warn("Pillow not installed, skipping PIL readability check")
        return
    try:
        import cv2
    except ImportError:
        warn("OpenCV not installed, skipping cv2 readability check")
        return
    image_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    fail_count = 0
    for fpath in data_root.rglob("*"):
        if fpath.suffix.lower() in image_exts:
            try:
                img = Image.open(fpath)
                img.verify()
            except Exception as e:
                error(f"PIL cannot open: {fpath} ({e})")
                fail_count += 1
                continue
            try:
                img_cv = cv2.imread(str(fpath))
                if img_cv is None:
                    error(f"cv2.imread returned None: {fpath}")
                    fail_count += 1
            except Exception as e:
                error(f"cv2 cannot open: {fpath} ({e})")
                fail_count += 1
    if fail_count == 0:
        info("All images passed PIL and cv2 readability checks")


def check_directory_structure(data_root: Path) -> None:
    print("\n[4/7] Checking MVTec AD directory structure...")
    for d in sorted(data_root.iterdir()):
        if d.is_file() or d.name.startswith('.'):
            continue
        train_good = d / "train" / "good"
        if not train_good.exists():
            warn(f"{d.name}: missing train/good/ directory")
        test_dir = d / "test"
        if not test_dir.exists():
            warn(f"{d.name}: missing test/ directory")
            continue
        for sub in sorted(test_dir.iterdir()):
            if sub.is_dir() and sub.name != "good":
                gt_dir = d / "ground_truth" / sub.name
                if not gt_dir.exists():
                    warn(f"{d.name}: test/{sub.name}/ exists but ground_truth/{sub.name}/ missing")


def check_class_distribution(data_root: Path) -> None:
    print("\n[5/7] Checking test set class distribution...")
    for d in sorted(data_root.iterdir()):
        if d.is_file() or d.name.startswith('.'):
            continue
        test_dir = d / "test"
        if not test_dir.exists():
            continue
        for sub in sorted(test_dir.iterdir()):
            if sub.is_dir():
                count = sum(1 for f in sub.iterdir() if f.is_file())
                if sub.name != "good" and count < 5:
                    warn(f"{d.name}/test/{sub.name}: only {count} samples (< 5), statistics unreliable")


def check_missing_referenced(data_root: Path) -> None:
    print("\n[6/7] Checking README-referenced datasets missing from disk...")
    on_disk = {d.name for d in data_root.iterdir() if d.is_dir() and not d.name.startswith('.')}
    missing_refs = REFERENCED_DATASETS - on_disk
    for ds in sorted(missing_refs):
        warn(f"README references dataset not on disk: {ds}")
    unknown = on_disk - REFERENCED_DATASETS
    for ds in sorted(unknown):
        info(f"On disk but not in known list: {ds}")


def check_unknown_dirs(data_root: Path) -> None:
    print("\n[7/7] Checking unknown directories...")
    on_disk = {d.name for d in data_root.iterdir() if d.is_dir() and not d.name.startswith('.')}
    unknown = on_disk - REFERENCED_DATASETS
    if unknown:
        for ds in sorted(unknown):
            info(f"Unknown directory: {ds}")
    else:
        info("All directories are in the known list")


def write_report(data_root: str, output_dir: str) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = Path(output_dir) / f"data_validation_{timestamp}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Data Validation Report",
        f"",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Data directory:** {data_root}",
        f"**Exit code:** {'0 (no errors)' if not ERRORS else '1 (errors found)'}",
        f"",
        f"## Summary",
        f"",
        f"| Level | Count |",
        f"|-------|-------|",
        f"| ERROR | {len(ERRORS)} |",
        f"| WARN  | {len(WARNINGS)} |",
        f"| INFO  | {len(INFOS)} |",
        f"",
    ]
    if ERRORS:
        lines.append("## ERROR")
        lines.append("")
        for e in ERRORS:
            lines.append(f"- {e}")
        lines.append("")
    if WARNINGS:
        lines.append("## WARN")
        lines.append("")
        for w in WARNINGS:
            lines.append(f"- {w}")
        lines.append("")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"\n[SAVE] Report saved: {report_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate data/ directory integrity")
    parser.add_argument("--data-root", default="./data", help="Data directory path (default: ./data)")
    parser.add_argument("--output-dir", default="./results", help="Report output directory (default: ./results)")
    args = parser.parse_args()
    data_root = Path(args.data_root).resolve()
    if not data_root.exists():
        print(f"[FATAL] Data directory does not exist: {data_root}")
        return 1
    print(f"Data Validation: {data_root}")
    print(f"=" * 60)
    check_magic_bytes(data_root)
    check_empty_files(data_root)
    check_readability(data_root)
    check_directory_structure(data_root)
    check_class_distribution(data_root)
    check_missing_referenced(data_root)
    check_unknown_dirs(data_root)
    write_report(str(data_root), args.output_dir)
    print(f"\n{'=' * 60}")
    print(f"Validation complete: {len(ERRORS)} ERROR, {len(WARNINGS)} WARN, {len(INFOS)} INFO")
    return 1 if ERRORS else 0


if __name__ == "__main__":
    sys.exit(main())
