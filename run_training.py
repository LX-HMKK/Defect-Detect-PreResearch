#!/usr/bin/env python
"""
入口脚本 2: 模型训练
用法: 
    python run_training.py              # 直接运行（从 patchcore.yaml 读取配置）
    python run_training.py --model all  # 训练所有模型
    python run_training.py -m patchcore # 指定模型
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from modules.algorithm.trainer import main

if __name__ == '__main__':
    main()
