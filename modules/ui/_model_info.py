"""
UI 共享的轻量模型信息 — 避免导入 heavy 依赖（anomalib/torch/gradio）。

server.py 的 /api/models、/api/train 等端点只需模型名称与方向，无需实例化
anomalib 模型类，因此将该映射独立出来，使 API 测试可在无 GPU/无 anomalib
环境下导入。
"""
from pathlib import Path
from typing import Any

from modules._runtime import resolve_project_path
from modules.config import get as cfg_get


MODEL_CONFIGS = {
    'fre': {
        'name': 'FRE',
        'direction': '基于特征重构',
    },
    'patchcore': {
        'name': 'PatchCore',
        'direction': '基于特征建模',
    },
    'draem': {
        'name': 'DRAEM',
        'direction': '基于判别重构',
    },
    'padim': {
        'name': 'PaDiM',
        'direction': '基于概率建模',
    },
}


MODEL_RESULT_SUBDIRS = {
    "fre": "Fre",
    "patchcore": "Patchcore",
    "draem": "Draem",
    "padim": "Padim",
}


def _scan_dataset_source(model_path: Path, source: str, model_key: str, results_dir: Path) -> list[dict[str, Any]]:
    """扫描指定模型路径下 {source} 中的数据集类别，返回对象列表。"""
    datasets = []
    source_path = model_path / source
    if not source_path.exists():
        return datasets

    for cat_dir in source_path.iterdir():
        if not cat_dir.is_dir() or cat_dir.name == '__pycache__':
            continue
        # 要求目录下存在 vX 版本子目录，避免误把临时目录当数据集
        if any(
            child.is_dir() and child.name.startswith('v')
            for child in cat_dir.iterdir()
        ):
            value = f"{source}/{cat_dir.name}"
            label = _resolve_display_name(model_key, cat_dir.name, source, results_dir)
            datasets.append({
                'value': value,
                'label': label,
                'source': source,
            })
    return datasets


def _resolve_display_name(model_key: str, category: str, source: str, results_dir: Path) -> str:
    """解析数据集显示名称。用户训练结果优先读取结果 JSON 中的 display_name。"""
    if source == 'default':
        return category
    result_json = results_dir / 'comparison' / f'{model_key}_{category}_results.json'
    if result_json.exists():
        try:
            import json
            data = json.loads(result_json.read_text(encoding='utf-8'))
            display_name = data.get('display_name')
            if display_name:
                return display_name
        except Exception:
            pass
    return category


def get_available_datasets() -> list[dict[str, Any]]:
    """
    自动检测可用的数据集。

    结果格式：
        - 默认（精调）结果：{"value": "default/{category}", "label": "{category}", "source": "default"}
        - 用户自训练结果：{"value": "user/{category}", "label": "显示名称", "source": "user"}

    扫描路径基于 configs/config.yaml 的 paths.results_root，避免依赖启动工作目录。
    """
    results_dir = resolve_project_path(cfg_get('paths.results_root', './results'))
    datasets = []

    for model_key, subdir in MODEL_RESULT_SUBDIRS.items():
        model_path = results_dir / model_key / subdir
        if not model_path.exists():
            continue

        # 1) 默认/精调结果：results/{model}/Patchcore/default/{category}
        datasets.extend(_scan_dataset_source(model_path, 'default', model_key, results_dir))

        # 2) 用户自训练结果：results/{model}/Patchcore/user/{category}
        datasets.extend(_scan_dataset_source(model_path, 'user', model_key, results_dir))

    # 按 value 去重：同一类别在多个模型目录下会被多次扫描到
    unique: dict = {}
    for ds in datasets:
        value = ds['value']
        if value not in unique:
            unique[value] = ds
        elif ds.get('display_name'):
            unique[value] = ds
    datasets = list(unique.values())

    return sorted(datasets, key=lambda d: (d['source'] != 'default', d['label']))


def get_self_trained_models(model_key: str) -> list[dict[str, Any]]:
    """扫描指定算法下所有用户自训练模型。

    路径结构: results/{model_key}/{ModelName}/user/{category}/vX
    返回对象包含 path、category、version、display_name。
    """
    results_dir = resolve_project_path(cfg_get('paths.results_root', './results'))
    subdir = MODEL_RESULT_SUBDIRS.get(model_key)
    if not subdir:
        return []

    user_root = results_dir / model_key / subdir / "user"
    if not user_root.exists():
        return []

    models = []
    for cat_dir in user_root.iterdir():
        if not cat_dir.is_dir() or cat_dir.name == '__pycache__':
            continue
        for version_dir in sorted(cat_dir.iterdir()):
            if not version_dir.is_dir() or not version_dir.name.startswith('v'):
                continue
            try:
                version = int(version_dir.name[1:])
            except ValueError:
                continue
            # 要求目录下存在有效的 lightning checkpoint
            ckpts = list(version_dir.glob('*.ckpt'))
            if not ckpts:
                continue
            display_name = _resolve_display_name(model_key, cat_dir.name, 'user', results_dir)
            models.append({
                'path': str(version_dir),
                'category': cat_dir.name,
                'version': version,
                'display_name': display_name,
            })

    return sorted(models, key=lambda m: (m['category'], m['version']))
