"""项目运行时工具 — pycache 重定向等功能，供 scripts/ 和 modules/ 共享使用。"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(path_value: str | Path) -> Path:
    """将项目内相对路径解析为绝对路径。"""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def get_runtime_cache_dir() -> Path:
    """返回项目统一运行时缓存目录。"""
    cache_dir = resolve_project_path(".cache")
    cache_dir.mkdir(exist_ok=True)
    return cache_dir


def configure_runtime_temp() -> None:
    """将 Python 字节码缓存重定向到项目内的 .cache/pycache/ 目录。"""
    pycache_dir = get_runtime_cache_dir() / "pycache"
    pycache_dir.mkdir(exist_ok=True)
    sys.pycache_prefix = str(pycache_dir)
    os.environ["PYTHONPYCACHEPREFIX"] = str(pycache_dir)


def get_all_categories(data_path: str | Path) -> list[str]:
    """自动发现数据目录中的所有类别（包含 train/ 子目录的文件夹），按名称排序。"""
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
