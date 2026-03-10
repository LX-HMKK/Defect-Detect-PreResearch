#!/usr/bin/env python
"""
入口脚本 2: 模型训练
用法: python run_training.py --model all --category my_product --data_path ./data/processed
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from modules.algorithm.trainer import main

if __name__ == '__main__':
    main()
