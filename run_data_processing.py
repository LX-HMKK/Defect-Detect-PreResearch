#!/usr/bin/env python
"""
入口脚本 1: 数据集处理
用法: python run_data_processing.py --input_dir ./data/raw --output_dir ./data/processed
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from modules.data_processing.dataset_formatter import main

if __name__ == '__main__':
    main()
