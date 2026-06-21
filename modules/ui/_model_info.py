"""
UI 共享的轻量模型信息 — 避免导入 heavy 依赖（anomalib/torch/gradio）。

server.py 的 /api/models、/api/train 等端点只需模型名称与方向，无需实例化
anomalib 模型类，因此将该映射独立出来，使 API 测试可在无 GPU/无 anomalib
环境下导入。
"""
from pathlib import Path

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


def _scan_dataset_source(model_path: Path, source: str) -> set:
    """扫描指定模型路径下 {source} 中的数据集类别。"""
    datasets = set()
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
            datasets.add(f"{source}/{cat_dir.name}")
    return datasets


def get_available_datasets():
    """
    自动检测可用的数据集。

    结果格式：
        - 默认（精调）结果："default/{category}"
        - 用户自训练结果："user/{category}"

    扫描路径基于 configs/config.yaml 的 paths.results_root，避免依赖启动工作目录。
    """
    results_dir = resolve_project_path(cfg_get('paths.results_root', './results'))
    datasets = set()
    model_dirs = {
        "fre": "Fre",
        "patchcore": "Patchcore",
        "draem": "Draem",
        "padim": "Padim",
    }

    for model_key, subdir in model_dirs.items():
        model_path = results_dir / model_key / subdir
        if not model_path.exists():
            continue

        # 1) 默认/精调结果：results/{model}/Patchcore/default/{category}
        datasets.update(_scan_dataset_source(model_path, 'default'))

        # 2) 用户自训练结果：results/{model}/Patchcore/user/{category}
        datasets.update(_scan_dataset_source(model_path, 'user'))

    return sorted(datasets)
