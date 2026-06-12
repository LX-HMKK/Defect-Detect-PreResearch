"""Tests for modules/evaluation/metrics.py — AUROC, AUPR, pixel AUROC, PRO with synthetic arrays."""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from modules.evaluation.metrics import MetricsEvaluator


evaluator = MetricsEvaluator()


def test_auroc_perfect_separation():
    scores = np.array([0.1, 0.1, 0.1, 0.9, 0.9, 0.9], dtype=np.float64)
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
    result = evaluator.compute_image_auroc(scores, labels)
    assert abs(result - 1.0) < 1e-6, f"Expected 1.0, got {result}"


def test_auroc_chance():
    scores = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float64)
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
    result = evaluator.compute_image_auroc(scores, labels)
    assert abs(result - 0.5) < 1e-6, f"Expected 0.5, got {result}"


def test_auroc_single_class_guard():
    scores = np.array([0.1, 0.1, 0.1], dtype=np.float64)
    labels = np.array([0, 0, 0], dtype=np.int32)
    result = evaluator.compute_image_auroc(scores, labels)
    assert result == 0.5


def test_aupr_perfect():
    scores = np.array([0.1, 0.1, 0.1, 0.9, 0.9, 0.9], dtype=np.float64)
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
    result = evaluator.compute_image_aupr(scores, labels)
    assert abs(result - 1.0) < 1e-6, f"Expected 1.0, got {result}"


def test_aupr_single_class_guard():
    scores = np.array([0.1, 0.1], dtype=np.float64)
    labels = np.array([0, 0], dtype=np.int32)
    result = evaluator.compute_image_aupr(scores, labels)
    assert result == 0.0


def test_pixel_auroc_perfect():
    anomaly_maps = np.array([
        [[0, 0], [0, 0]],
        [[1, 1], [1, 1]],
    ], dtype=np.float64)
    gt_masks = np.array([
        [[0, 0], [0, 0]],
        [[1, 1], [1, 1]],
    ], dtype=np.float64)
    result = evaluator.compute_pixel_auroc(anomaly_maps, gt_masks)
    assert abs(result - 1.0) < 1e-6, f"Expected 1.0, got {result}"


def test_pixel_auroc_single_class():
    anomaly_maps = np.zeros((2, 4, 4), dtype=np.float64)
    gt_masks = np.zeros((2, 4, 4), dtype=np.float64)
    result = evaluator.compute_pixel_auroc(anomaly_maps, gt_masks)
    assert result == 0.5


def test_pro_synthetic():
    anomaly_maps = np.zeros((2, 10, 10), dtype=np.float64)
    gt_masks = np.zeros((2, 10, 10), dtype=np.float64)
    gt_masks[0, 0:3, 0:3] = 1.0
    anomaly_maps[0, 0:2, 0:2] = 0.8  # partial defect detection (4 of 9 pixels)
    anomaly_maps[0, 5:7, 5:7] = 0.4  # false positive region
    result = evaluator.compute_pro(anomaly_maps, gt_masks)
    assert result > 0.5, f"PRO should be > 0.5, got {result}"
    assert result < 1.0, f"PRO should be < 1.0, got {result}"
