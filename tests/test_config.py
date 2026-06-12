"""Tests for modules/config/manager.py — ConfigManager singleton, YAML loading, get()."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from modules.config.manager import ConfigManager, get_config, reset_config, get


def _reset():
    reset_config()


def test_singleton_identity():
    _reset()
    a = get_config()
    b = get_config()
    assert a is b


def test_yaml_loads_data_section():
    _reset()
    batch = get('data.train_batch_size')
    assert isinstance(batch, int)
    assert batch > 0
    eval_batch = get('data.eval_batch_size')
    assert isinstance(eval_batch, int)
    assert eval_batch > 0
    workers = get('data.num_workers')
    assert isinstance(workers, int)
    assert workers >= 0


def test_get_nested_key():
    _reset()
    import yaml
    config_path = PROJECT_ROOT / 'configs' / 'config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f)
    expected = raw['data']['train_batch_size']
    assert get('data.train_batch_size') == expected


def test_get_model_config_returns_dict_patchcore():
    _reset()
    from modules.config import get_model_config
    cfg = get_model_config('patchcore')
    assert isinstance(cfg, dict)
    for key in ['backbone', 'layers', 'coreset_sampling_ratio', 'num_neighbors', 'pre_trained']:
        assert key in cfg, f"patchcore config missing key: {key}"


def test_get_model_config_returns_dict_all():
    _reset()
    from modules.config import get_model_config
    for model_name in ['fre', 'draem', 'padim']:
        cfg = get_model_config(model_name)
        assert isinstance(cfg, dict), f"{model_name} config is not dict"
        assert len(cfg) > 0, f"{model_name} config is empty"


def test_threshold_default_fallback():
    _reset()
    result = get('threshold.default', 0.5)
    assert result == 0.5
    result_none = get('completely.nonexistent.key.xyz', None)
    assert result_none is None


def test_missing_key_returns_none():
    _reset()
    result = get('completely.nonexistent.key.xyz')
    assert result is None
