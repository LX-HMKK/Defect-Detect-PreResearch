"""
训练后端模块 — 供 FastAPI SSE 训练端点使用
"""
import io
import json
import queue
import shutil
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import yaml
from pytorch_lightning.callbacks import Callback

from modules._runtime import resolve_project_path
from modules.algorithm.trainer import AnomalyDetectionTrainer
from modules.config import get as cfg_get


MAX_TRAIN_SAMPLES = 150


class TrainingMetricsCallback(Callback):
    """PyTorch Lightning 回调，将训练指标写入队列供 SSE 读取。"""

    def __init__(self, metrics_queue: queue.Queue, stop_event: threading.Event):
        self.metrics_queue = metrics_queue
        self.stop_event = stop_event
        self.start_time: Optional[float] = None

    def on_train_start(self, trainer, pl_module):
        self.start_time = time.time()
        self._log(f"训练开始，共 {trainer.max_epochs} 个 epoch")

    def _put(self, payload: Dict):
        """将事件放入队列；关键事件（error/done/completed）必须送达，其余允许丢弃。"""
        event = payload.get('event')
        critical = event in ('error', 'done', 'completed')
        try:
            self.metrics_queue.put(payload, block=critical, timeout=5.0 if critical else 0)
        except queue.Full:
            pass
        except Exception:
            # 队列不可用时不应中断训练
            pass

    def _log(self, message: str, level: str = 'info'):
        """推送日志事件到 SSE 队列，附带时间戳。"""
        self._put({
            'event': 'log',
            'message': message,
            'level': level,
            'timestamp': time.time(),
        })

    def _check_stop(self, trainer):
        if self.stop_event.is_set():
            trainer.should_stop = True
            self._put({'event': 'status', 'status': 'stopping', 'message': '收到停止信号，当前 epoch 结束后终止...'})

    def on_train_epoch_end(self, trainer, pl_module):
        self._check_stop(trainer)
        epoch = trainer.current_epoch
        metrics = trainer.callback_metrics
        train_loss = None
        if 'train_loss' in metrics and metrics['train_loss'] is not None:
            try:
                train_loss = float(metrics['train_loss'].cpu().item())
            except Exception:
                train_loss = None
        lr = None
        if trainer.optimizers:
            try:
                lr = float(trainer.optimizers[0].param_groups[0]['lr'])
            except Exception:
                lr = None
        self._put({
            'event': 'metric',
            'epoch': epoch,
            'total_epochs': trainer.max_epochs,
            'train_loss': train_loss,
            'learning_rate': lr,
        })
        self._log(f"Epoch {epoch + 1}/{trainer.max_epochs} 完成" + (f"，loss={train_loss:.4f}" if train_loss is not None else ""))

    def on_validation_epoch_end(self, trainer, pl_module):
        self._check_stop(trainer)
        epoch = trainer.current_epoch
        metrics = trainer.callback_metrics
        val_auroc = None
        if 'val_image_AUROC' in metrics and metrics['val_image_AUROC'] is not None:
            try:
                val_auroc = float(metrics['val_image_AUROC'].cpu().item())
            except Exception:
                val_auroc = None
        elapsed = time.time() - self.start_time if self.start_time else 0
        epoch_per_sec = (epoch + 1) / elapsed if elapsed > 0 and epoch >= 0 else 0
        remaining_epochs = max(0, trainer.max_epochs - epoch - 1)
        eta_seconds = int(remaining_epochs / epoch_per_sec) if epoch_per_sec > 0 else 0
        self._put({
            'event': 'metric',
            'epoch': epoch,
            'total_epochs': trainer.max_epochs,
            'val_image_AUROC': val_auroc,
            'eta_seconds': eta_seconds,
        })
        self._log("验证完成" + (f"，val_image_AUROC={val_auroc:.4f}" if val_auroc is not None else ""))

    def on_train_end(self, trainer, pl_module):
        self._put({'event': 'status', 'status': 'training_end'})
        self._log("训练结束")


def format_uploaded_samples(
    upload_dir: Path,
    image_files: List[Path],
    max_samples: int = MAX_TRAIN_SAMPLES,
    seed: int = 42,
) -> Path:
    """将上传的图片整理成 MVTec AD 临时结构。仅含正常样本：train/good/ + test/good/（从 train 中 hold-out 10%）。"""
    import random
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
    n_test = min(max(1, int(len(unique_files) * 0.1)), len(unique_files) - 1)
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


def run_training_job(
    model_name: str,
    dataset_path: Path,
    category: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    metrics_queue: queue.Queue,
) -> Dict:
    output_dir = resolve_project_path(cfg_get('paths.results_dir', './results'))
    base_config_path = Path(__file__).resolve().parents[2] / 'configs' / f'{model_name}.yaml'

    config = None
    temp_config_path: Optional[Path] = None
    try:
        if base_config_path.exists():
            with open(base_config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            if config and 'data' in config and 'init_args' in config['data']:
                config['data']['init_args']['train_batch_size'] = batch_size
                config['data']['init_args']['eval_batch_size'] = batch_size

        print(f"[TRAIN] 请求 learning_rate={learning_rate}，实际由各模型 YAML 决定")

        # 若未读取到配置，构造最小配置避免 None 崩溃
        if config is None:
            config = {
                'data': {
                    'class_path': 'anomalib.data.Folder',
                    'init_args': {
                        'root': str(dataset_path),
                        'normal_dir': 'train/good',
                        'abnormal_dir': 'test/good',
                        'normal_test_dir': 'test/good',
                        'task': 'segmentation',
                        'train_batch_size': batch_size,
                        'eval_batch_size': batch_size,
                        'num_workers': 0,
                        'image_size': [256, 256],
                    }
                }
            }

        temp_config_path = dataset_path / f'{model_name}_train_config.yaml'
        with open(temp_config_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(config, f)

        metrics_callback = TrainingMetricsCallback(metrics_queue, training_manager.stop_event)

        trainer = AnomalyDetectionTrainer(
            model_name=model_name,
            data_path=str(dataset_path),
            category=category,
            output_dir=str(output_dir),
            config_path=str(temp_config_path),
            seed=seed,
            extra_callbacks=[metrics_callback],
        )

        metrics_queue.put({
            'event': 'status',
            'status': 'setup',
            'message': f'正在加载 {model_name} 数据与模型...',
        })

        trainer.train(max_epochs=epochs)

        metrics_queue.put({
            'event': 'status',
            'status': 'evaluating',
            'message': '训练完成，正在评估并计算阈值...',
        })

        results = trainer.evaluate()

        metrics_queue.put({
            'event': 'completed',
            'model': model_name,
            'category': category,
            'results': results,
        })

        return {
            'status': 'completed',
            'model': model_name,
            'category': category,
            'results': results,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        metrics_queue.put({
            'event': 'error',
            'message': str(e),
            'code': 'TRAINING_ERROR',
        })
        return {
            'status': 'error',
            'model': model_name,
            'category': category,
            'message': str(e),
        }
    finally:
        if temp_config_path is not None:
            temp_config_path.unlink(missing_ok=True)
