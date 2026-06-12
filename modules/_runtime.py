"""项目运行时工具 — pycache 重定向等功能，供 scripts/ 和 modules/ 共享使用。"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def configure_runtime_temp() -> None:
    """将 Python 字节码缓存重定向到项目内的 temp/pycache/ 目录。"""
    temp_dir = PROJECT_ROOT / "temp"
    pycache_dir = temp_dir / "pycache"
    temp_dir.mkdir(exist_ok=True)
    pycache_dir.mkdir(exist_ok=True)
    sys.pycache_prefix = str(pycache_dir)
    os.environ["PYTHONPYCACHEPREFIX"] = str(pycache_dir)
