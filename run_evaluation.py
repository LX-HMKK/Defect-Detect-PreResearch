#!/usr/bin/env python
"""
入口脚本 3: 指标评测
用法: python run_evaluation.py --model all --category my_product
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from modules.evaluation.metrics import main

if __name__ == '__main__':
    main()
