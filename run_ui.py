#!/usr/bin/env python
"""
入口脚本 4: 启动 UI 演示
用法: python run_ui.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from modules.ui.demo import main

if __name__ == '__main__':
    main()
