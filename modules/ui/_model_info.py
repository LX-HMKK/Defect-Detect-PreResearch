"""
UI 共享的轻量模型信息 — 避免导入 heavy 依赖（anomalib/torch/gradio）。

server.py 的 /api/models、/api/train 等端点只需模型名称与方向，无需实例化
anomalib 模型类，因此将该映射独立出来，使 API 测试可在无 GPU/无 anomalib
环境下导入。
"""
from pathlib import Path


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


def get_available_datasets():
    """自动检测可用的数据集（支持 MVTec AD 与 Folder 两种输出结构）。"""
    results_dir = Path("./results")
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

        # 1) MVTec AD 结构: results/{model}/Patchcore/MVTec/{category}
        mvtec_path = model_path / "MVTec"
        if mvtec_path.exists():
            for cat_dir in mvtec_path.iterdir():
                if cat_dir.is_dir() and cat_dir.name not in ["__pycache__"]:
                    datasets.add(cat_dir.name)

        # 2) Folder 结构: results/{model}/Patchcore/{category}/v0/weights
        for cat_dir in model_path.iterdir():
            if not cat_dir.is_dir() or cat_dir.name in ["__pycache__", "MVTec"]:
                continue
            if any(
                child.is_dir() and child.name.startswith("v")
                for child in cat_dir.iterdir()
            ):
                datasets.add(cat_dir.name)

    return sorted(datasets)
