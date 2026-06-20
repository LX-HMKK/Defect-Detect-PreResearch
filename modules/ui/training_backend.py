"""
训练后端模块 — 供 FastAPI SSE 训练端点使用。

注意：本模块在导入阶段不引入 anomalib/torch 等 heavy 依赖。
AnomalyDetectionTrainer 与 PyTorch Lightning Callback 均在 run_training_job
被调用时延迟导入，以便 API 端点与测试可在轻量环境中加载。
"""
import io
import json
import queue
import shutil
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from modules._runtime import resolve_project_path
from modules.config import get as cfg_get
from modules.ui._training_common import (
    MAX_TRAIN_SAMPLES,
    TrainingTaskManager,
    format_uploaded_samples,
    training_manager,
)


# format_uploaded_samples / TrainingTaskManager 已迁移至 _training_common.py，
# 保留同名导出以兼容现有调用方。
__all__ = [
    "MAX_TRAIN_SAMPLES",
    "format_uploaded_samples",
    "TrainingTaskManager",
    "training_manager",
    "run_training_job",
]


def _move_excluded_samples(dataset_path: Path, excluded_samples: List[str]) -> List[Tuple[Path, Path]]:
    """
    将被排除的样本从 train/good 临时移出，避免参与训练。

    Returns:
        移动记录列表，每项为 (原始路径, 临时路径)，供训练结束后恢复。
    """
    if not excluded_samples:
        return []

    train_dir = dataset_path / 'train' / 'good'
    excluded_dir = dataset_path / '.excluded'
    excluded_dir.mkdir(parents=True, exist_ok=True)

    moved: List[Tuple[Path, Path]] = []
    for name in excluded_samples:
        src = train_dir / name
        if not src.exists():
            continue
        dst = excluded_dir / name
        # 处理文件名冲突
        counter = 1
        stem = dst.stem
        suffix = dst.suffix
        while dst.exists():
            dst = excluded_dir / f"{stem}_{counter}{suffix}"
            counter += 1
        shutil.move(str(src), str(dst))
        moved.append((src, dst))

    return moved


def _restore_excluded_samples(moved: List[Tuple[Path, Path]]) -> None:
    """训练结束后将被排除样本恢复至 train/good。"""
    for src, dst in moved:
        if dst.exists():
            src.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(dst), str(src))

    if moved:
        excluded_dir = moved[0][1].parent
        try:
            excluded_dir.rmdir()
        except OSError:
            pass


def run_training_job(
    model_name: str,
    dataset_path: Path,
    category: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    metrics_queue: queue.Queue,
    excluded_samples: Optional[List[str]] = None,
) -> Dict:
    """在线程中执行训练并通过 SSE 队列推送状态/指标/日志。"""
    # 延迟导入 heavy 依赖，保证模块导入阶段不触发 anomalib/torch。
    import threading
    from pytorch_lightning.callbacks import Callback
    from modules.algorithm.trainer import AnomalyDetectionTrainer

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

    output_dir = resolve_project_path(cfg_get('paths.results_dir', './results'))
    base_config_path = Path(__file__).resolve().parents[2] / 'configs' / f'{model_name}.yaml'

    config = None
    temp_config_path: Optional[Path] = None
    moved_samples: List[Tuple[Path, Path]] = []
    try:
        # 训练前将被排除样本从 train/good 移出
        moved_samples = _move_excluded_samples(dataset_path, excluded_samples or [])

        # 上传数据集目录本身就是 MVTec AD 格式的类别目录（train/good + test/good）。
        # 因此 data_path 应为其父目录，category 为其目录名。
        data_root = str(dataset_path.parent)
        data_category = dataset_path.name

        if base_config_path.exists():
            with open(base_config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            if config and 'data' in config and 'init_args' in config['data']:
                config['data']['init_args']['root'] = data_root
                config['data']['init_args']['category'] = data_category
                config['data']['init_args']['train_batch_size'] = batch_size
                config['data']['init_args']['eval_batch_size'] = batch_size

        print(f"[TRAIN] 使用 learning_rate={learning_rate}（DRAEM/FRE 生效；PatchCore/PaDiM 忽略）")

        # 若未读取到配置，构造最小配置避免 None 崩溃
        # 上传数据集无 ground_truth，实际由 get_datamodule_from_config 识别为 Folder
        if config is None:
            config = {
                'data': {
                    'class_path': 'anomalib.data.Folder',
                    'init_args': {
                        'root': data_root,
                        'category': data_category,
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

        # 上传数据集通常不含 ground_truth，关闭像素级指标避免评估时 gt_mask 缺失
        enable_pixel_metrics = (dataset_path / 'ground_truth').exists()

        trainer = AnomalyDetectionTrainer(
            model_name=model_name,
            data_path=data_root,
            category=data_category,
            output_dir=str(output_dir),
            config_path=str(temp_config_path),
            seed=seed,
            extra_callbacks=[metrics_callback],
            enable_pixel_metrics=enable_pixel_metrics,
            learning_rate=learning_rate,
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
        _restore_excluded_samples(moved_samples)
        if temp_config_path is not None:
            temp_config_path.unlink(missing_ok=True)
