"""
Training Studio 共享的轻量组件 — 避免导入 heavy 依赖。

该模块仅包含线程锁、样本格式化等可在无 anomalib/torch 环境下使用的工具，
供 server.py 的 API 端点在导入阶段直接使用；实际的训练逻辑（需要 anomalib）
保留在 training_backend.py 中，按需延迟导入。
"""
import queue
import random
import shutil
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


MAX_TRAIN_SAMPLES = 150


def format_uploaded_samples(
    upload_dir: Path,
    image_files: List[Path],
    max_samples: int = MAX_TRAIN_SAMPLES,
    seed: int = 42,
) -> Path:
    """将上传的图片整理成 MVTec AD 临时结构。仅含正常样本：train/good/ + test/good/（从 train 中 hold-out 10%）。"""
    random.seed(seed)

    unique_files = sorted(set(str(p.resolve()) for p in image_files))
    unique_files = [Path(p) for p in unique_files]
    if len(unique_files) > max_samples:
        unique_files = random.sample(unique_files, max_samples)

    upload_dir.mkdir(parents=True, exist_ok=True)
    train_dir = upload_dir / 'train' / 'good'
    test_dir = upload_dir / 'test' / 'good'
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    random.shuffle(unique_files)
    # 大样本时至少保留 2 个测试样本，避免 val split 除零；小样本按原始比例
    n_total = len(unique_files)
    if n_total >= 4:
        n_test = min(max(2, int(n_total * 0.1)), n_total - 2)
    else:
        n_test = min(max(1, int(n_total * 0.1)), n_total - 1)
    test_files = unique_files[:n_test]
    train_files = unique_files[n_test:]

    for idx, src in enumerate(train_files, 1):
        dst = train_dir / f"{idx:04d}{src.suffix}"
        shutil.copy2(str(src), str(dst))
    for idx, src in enumerate(test_files, 1):
        dst = test_dir / f"{idx:04d}{src.suffix}"
        shutil.copy2(str(src), str(dst))

    return upload_dir


class TrainingTaskManager:
    """全局单训练任务锁与停止事件管理器。使用 threading.Lock 保证线程安全。"""

    def __init__(self):
        self._lock = threading.Lock()
        self._locked = False
        self._current: Optional[Dict] = None
        self._started_at: Optional[str] = None
        self.stop_event = threading.Event()

    def try_start(self, model: str, category: str, total_epochs: int) -> bool:
        with self._lock:
            if self._locked:
                return False
            self._locked = True
            self.stop_event.clear()
            self._current = {
                'model': model,
                'category': category,
                'current_epoch': 0,
                'total_epochs': total_epochs,
            }
            self._started_at = datetime.now().isoformat()
            return True

    def update_epoch(self, epoch: int):
        with self._lock:
            if self._current:
                self._current['current_epoch'] = epoch

    def stop(self):
        """仅设置停止信号，不释放锁。"""
        self.stop_event.set()

    def finish(self):
        """任务完成后释放锁并重置状态。"""
        with self._lock:
            self._locked = False
            self._current = None
            self._started_at = None
            self.stop_event.clear()

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._locked

    def to_dict(self) -> Dict:
        with self._lock:
            if not self._locked:
                return {'running': False}
            return {
                'running': True,
                'started_at': self._started_at,
                **self._current,
            }


training_manager = TrainingTaskManager()
