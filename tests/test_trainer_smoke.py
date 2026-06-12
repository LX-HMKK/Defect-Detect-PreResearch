"""Smoke tests for SUPPORTED_MODELS — reads the list without importing trainer.py.

No GPU required. No anomalib needed.
"""

import ast
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def test_supported_models_contains_all():
    """Parse SUPPORTED_MODELS from trainer.py source — avoids torch/anomalib imports."""
    trainer_path = PROJECT_ROOT / "modules" / "algorithm" / "trainer.py"
    source = trainer_path.read_text(encoding="utf-8")

    for line in source.splitlines():
        if line.strip().startswith("SUPPORTED_MODELS"):
            models = ast.literal_eval(line.split("=", 1)[1].strip())
            assert set(models) == {"fre", "patchcore", "draem", "padim"}, f"Got: {models}"
            return

    assert False, "SUPPORTED_MODELS not found in trainer.py"
