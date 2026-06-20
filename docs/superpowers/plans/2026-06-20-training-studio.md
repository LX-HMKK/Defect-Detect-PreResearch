# Training Studio 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有 FastAPI + Alpine.js SPA 中新增 Training Studio 页面，实现本地上传训练样本、前端配置并启动训练、SSE 实时推送训练指标、训练完成后模型可被推理页使用。

**Architecture:** 后端复用 `AnomalyDetectionTrainer` 并注入 Lightning 回调以采集 loss/learning_rate；通过线程安全队列将指标传递给 SSE 生成器；前端新增 `training.js` 管理训练状态机和 SSE 消费；页面结构从 3 页扩展为 4 页 snap 滚动。

**Tech Stack:** Python 3.10 · FastAPI · SSE-Starlette · PyTorch Lightning Callbacks · Alpine.js · Vanilla JS · CSS Variables

---

## 文件结构

| 文件 | 类型 | 职责 |
|------|------|------|
| `modules/algorithm/trainer.py` | 修改 | `AnomalyDetectionTrainer` 支持传入额外 callbacks |
| `modules/ui/training_backend.py` | 新增 | 训练任务封装、回调、单任务锁、临时目录格式化 |
| `modules/ui/server.py` | 修改 | 新增 `/api/upload-samples`、`/api/train`、`/api/train/stop`、`/api/train-status` |
| `modules/ui/static/index.html` | 修改 | 新增 `#s1-training` section，snap-dots 改为 4 个 |
| `modules/ui/static/js/app.js` | 修改 | `sectionCount` 和 `sectionNames` 更新为 4 页，注册全局训练状态 |
| `modules/ui/static/js/training.js` | 新增 | `TrainingRunner`、样本上传、训练状态机、实时曲线 |
| `modules/ui/static/css/app.css` | 修改 | 新增训练页样式（玻璃卡、药丸标签、监控区） |
| `tests/test_training_api.py` | 新增 | 上传、训练参数校验、SSE 解析单元测试 |
| `configs/config.yaml` | 修改 | 确认 `paths` 包含 `temp_dir`（上传目录使用 `.cache/uploads`） |

---

## 任务分解

### Task 1: 让 AnomalyDetectionTrainer 支持外部回调

**Files:**
- Modify: `modules/algorithm/trainer.py:406-415`（`__init__` 签名）
- Modify: `modules/algorithm/trainer.py:580-592`（Engine 创建处）
- Test: `tests/test_trainer_smoke.py`

- [ ] **Step 1: 修改 `__init__` 签名，新增 `extra_callbacks`**

```python
def __init__(
    self,
    model_name: str,
    data_path: str,
    category: str,
    output_dir: str = './results',
    config_path: Optional[str] = None,
    device: str = 'auto',
    seed: int = 42,
    extra_callbacks: Optional[List] = None,
):
```

保存参数：
```python
self.extra_callbacks = extra_callbacks or []
```

- [ ] **Step 2: 修改 Engine 创建，合并外部回调与早停回调**

找到 `self.engine = Engine(... callbacks=[early_stopping_callback] if early_stopping_callback else None)` 处，替换为：

```python
callbacks = list(self.extra_callbacks)
if early_stopping_callback:
    callbacks.append(early_stopping_callback)

self.engine = Engine(
    max_epochs=max_epochs,
    accelerator=self.device,
    devices=1,
    default_root_dir=str(self.output_dir / self.model_name),
    logger=False,
    enable_progress_bar=False,
    callbacks=callbacks if callbacks else None,
)
```

- [ ] **Step 3: 运行 trainer 烟雾测试**

Run: `python -m pytest tests/test_trainer_smoke.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add modules/algorithm/trainer.py
git commit -m "feat(trainer): 支持传入外部 Lightning callbacks"
```

---

### Task 2: 创建训练后端模块

**Files:**
- Create: `modules/ui/training_backend.py`
- Modify: `modules/ui/server.py`（后续任务导入）

- [ ] **Step 1: 创建文件并写入 TrainingMetricsCallback**

```python
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
from pytorch_lightning.callbacks import Callback

from modules._runtime import resolve_project_path
from modules.algorithm.trainer import AnomalyDetectionTrainer
from modules.config import get as cfg_get


MAX_TRAIN_SAMPLES = 150
TRAINING_LOCK_TIMEOUT = 5.0


class TrainingMetricsCallback(Callback):
    """Lightning 回调：将训练指标写入队列供 SSE 读取；支持外部停止信号。"""

    def __init__(self, metrics_queue: queue.Queue, stop_event: threading.Event):
        self.metrics_queue = metrics_queue
        self.stop_event = stop_event
        self.start_time: Optional[float] = None

    def on_train_start(self, trainer, pl_module):
        self.start_time = time.time()

    def _put(self, payload: Dict):
        try:
            self.metrics_queue.put(payload, block=False)
        except queue.Full:
            pass

    def _check_stop(self, trainer):
        if self.stop_event.is_set():
            trainer.should_stop = True
            self._put({'event': 'status', 'status': 'stopping', 'message': '收到停止信号，当前 epoch 结束后终止...'})

    def on_train_epoch_end(self, trainer, pl_module):
        self._check_stop(trainer)
        epoch = trainer.current_epoch
        metrics = trainer.callback_metrics
        train_loss = None
        if 'train_loss' in metrics:
            train_loss = float(metrics['train_loss'].cpu().item())

        lr = None
        if trainer.optimizers:
            lr = float(trainer.optimizers[0].param_groups[0]['lr'])

        self._put({
            'event': 'metric',
            'epoch': epoch,
            'total_epochs': trainer.max_epochs,
            'train_loss': train_loss,
            'learning_rate': lr,
        })

    def on_validation_epoch_end(self, trainer, pl_module):
        self._check_stop(trainer)
        epoch = trainer.current_epoch
        metrics = trainer.callback_metrics
        val_auroc = None
        if 'val_image_AUROC' in metrics:
            val_auroc = float(metrics['val_image_AUROC'].cpu().item())

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

    def on_train_end(self, trainer, pl_module):
        self._put({'event': 'status', 'status': 'training_end'})
```

- [ ] **Step 2: 写入临时目录格式化和训练运行函数**

```python
def format_uploaded_samples(
    upload_dir: Path,
    image_files: List[Path],
    max_samples: int = MAX_TRAIN_SAMPLES,
    seed: int = 42,
) -> Path:
    """
    将上传的图片整理成 MVTec AD 临时结构。
    仅含正常样本：train/good/ + test/good/（从 train 中 hold-out 10%，用于 anomalib validation 不报错）。
    """
    import random
    random.seed(seed)

    # 去重并限制数量
    unique_files = sorted(set(str(p.resolve()) for p in image_files))
    unique_files = [Path(p) for p in unique_files]
    if len(unique_files) > max_samples:
        unique_files = random.sample(unique_files, max_samples)

    upload_dir.mkdir(parents=True, exist_ok=True)
    train_dir = upload_dir / 'train' / 'good'
    test_dir = upload_dir / 'test' / 'good'
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # hold-out 10% 到 test/good
    random.shuffle(unique_files)
    n_test = max(1, int(len(unique_files) * 0.1)) if len(unique_files) >= 10 else 0
    test_files = unique_files[:n_test]
    train_files = unique_files[n_test:]

    # 复制文件（保持原始扩展名）
    for idx, src in enumerate(train_files, 1):
        dst = train_dir / f"{idx:04d}{src.suffix}"
        shutil.copy2(str(src), str(dst))
    for idx, src in enumerate(test_files, 1):
        dst = test_dir / f"{idx:04d}{src.suffix}"
        shutil.copy2(str(src), str(dst))

    return upload_dir


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
    """在线程中执行训练，指标写入队列。"""
    import yaml

    output_dir = resolve_project_path(cfg_get('paths.results_dir', './results'))
    base_config_path = Path(__file__).resolve().parents[2] / 'configs' / f'{model_name}.yaml'

    # 加载基础配置并覆盖 batch_size
    config = None
    if base_config_path.exists():
        with open(base_config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        if config and 'data' in config and 'init_args' in config['data']:
            config['data']['init_args']['train_batch_size'] = batch_size
            config['data']['init_args']['eval_batch_size'] = batch_size

    # learning_rate：当前版本使用模型 YAML 默认优化器学习率。
    # UI 保留该字段以便后续通过临时 YAML 的 model.init_args.optimizer 注入。
    # 这里仅记录到日志，不报错。
    print(f"[TRAIN] 请求 learning_rate={learning_rate}，实际由各模型 YAML 决定")

    # 写入临时配置
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

    return {
        'status': 'completed',
        'model': model_name,
        'category': category,
        'results': results,
    }
```

- [ ] **Step 3: 添加全局训练任务锁**

```python
class TrainingTaskManager:
    """管理全局唯一的训练任务状态。"""

    def __init__(self):
        self._lock = False
        self._current: Optional[Dict] = None
        self._started_at: Optional[str] = None
        self.stop_event = threading.Event()

    def try_start(self, model: str, category: str, total_epochs: int) -> bool:
        if self._lock:
            return False
        self._lock = True
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
        if self._current:
            self._current['current_epoch'] = epoch

    def stop(self):
        self._lock = False
        self._current = None
        self._started_at = None
        self.stop_event.clear()

    @property
    def is_running(self) -> bool:
        return self._lock

    def to_dict(self) -> Dict:
        if not self._lock:
            return {'running': False}
        return {
            'running': True,
            'started_at': self._started_at,
            **self._current,
        }


training_manager = TrainingTaskManager()
```

- [ ] **Step 4: Commit**

```bash
git add modules/ui/training_backend.py
git commit -m "feat(ui): 新增 Training Studio 后端模块（回调、格式化、任务锁）"
```

---

### Task 3: 实现 `/api/upload-samples` 端点

**Files:**
- Modify: `modules/ui/server.py`
- Modify: `configs/config.yaml`（确认路径配置）

- [ ] **Step 1: 在 `server.py` 添加 FastAPI 导入**

在现有导入中加入：

```python
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from typing import List
```

- [ ] **Step 2: 添加 upload-samples 端点**

在 `/api/models` 之后添加：

```python
@app.post("/api/upload-samples")
async def upload_samples(
    files: List[UploadFile] = File(...),
):
    """
    上传训练样本图片，格式化为 MVTec AD 临时结构。
    仅保留图片文件，超过 150 张则截断。
    """
    if not files:
        raise HTTPException(status_code=400, detail="未上传任何文件")

    # 过滤非图片文件
    image_files = [f for f in files if f.content_type and f.content_type.startswith("image/")]
    if len(image_files) != len(files):
        raise HTTPException(status_code=400, detail="只能上传图片文件")

    if len(image_files) > 150:
        image_files = image_files[:150]

    session_id = uuid.uuid4().hex
    upload_root = resolve_project_path(cfg_get('paths.temp_dir', './.cache')) / 'uploads' / f'training_{session_id}'
    upload_root.mkdir(parents=True, exist_ok=True)

    saved_paths: List[Path] = []
    for file in image_files:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        ext = Path(file.filename).suffix or '.png'
        dst = upload_root / f'{len(saved_paths):04d}{ext}'
        cv2.imwrite(str(dst), img)
        saved_paths.append(dst)

    if not saved_paths:
        raise HTTPException(status_code=400, detail="没有有效的图片文件")

    from modules.ui.training_backend import format_uploaded_samples
    format_uploaded_samples(upload_root, saved_paths)

    return {
        'session_id': session_id,
        'dataset_path': str(upload_root),
        'category': f'training_{session_id}',
        'total': len(saved_paths),
        'max_allowed': 150,
        'samples': [p.name for p in sorted((upload_root / 'train' / 'good').glob('*'))],
    }
```

- [ ] **Step 3: 确认 `configs/config.yaml` 含 `paths.temp_dir`**

若不存在，在 `config.yaml` 顶部添加：

```yaml
paths:
  temp_dir: "./.cache"
  results_dir: "./results"
```

- [ ] **Step 4: 启动服务器并测试上传**

Run: `python scripts/run_ui.py --no-browser`
Then in another shell:
```bash
curl -X POST -F "files=@/path/to/sample.png" http://127.0.0.1:8000/api/upload-samples
```
Expected: JSON with session_id, total, samples

- [ ] **Step 5: Commit**

```bash
git add modules/ui/server.py configs/config.yaml
git commit -m "feat(ui): 新增 /api/upload-samples 训练样本上传端点"
```

---

### Task 4: 实现 `/api/train`、`/api/train-status`、`/api/train/stop` 端点

**Files:**
- Modify: `modules/ui/server.py`

- [ ] **Step 1: 添加训练端点相关导入**

在 `server.py` 顶部添加：

```python
import queue
from pydantic import BaseModel

from modules.ui.training_backend import (
    run_training_job,
    training_manager,
    MAX_TRAIN_SAMPLES,
)
```

- [ ] **Step 2: 定义请求模型**

```python
class TrainRequest(BaseModel):
    model: str
    dataset_path: str
    category: str
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.0001
    seed: int = 42
```

- [ ] **Step 3: 实现 `/api/train-status`**

```python
@app.get("/api/train-status")
async def train_status():
    return training_manager.to_dict()
```

- [ ] **Step 4: 实现 `/api/train/stop`**

```python
@app.post("/api/train/stop")
async def stop_train():
    """请求停止当前训练任务（在当前 epoch 结束后优雅停止）。"""
    if not training_manager.is_running:
        return {'status': 'idle'}
    training_manager.stop_event.set()
    return {'status': 'stop_requested'}
```

- [ ] **Step 5: 实现 `/api/train` SSE**

```python
@app.post("/api/train")
async def train(req: TrainRequest):
    if not training_manager.try_start(req.model, req.category, req.epochs):
        raise HTTPException(status_code=409, detail="已有训练任务进行中")

    metrics_queue: queue.Queue = queue.Queue(maxsize=200)

    async def event_generator():
        # 启动训练线程
        import threading
        result_container = {}

        def thread_target():
            try:
                result = run_training_job(
                    model_name=req.model,
                    dataset_path=Path(req.dataset_path),
                    category=req.category,
                    epochs=req.epochs,
                    batch_size=req.batch_size,
                    learning_rate=req.learning_rate,
                    seed=req.seed,
                    metrics_queue=metrics_queue,
                )
                result_container['result'] = result
            except Exception as e:
                result_container['error'] = str(e)
                metrics_queue.put({
                    'event': 'error',
                    'message': str(e),
                    'code': 'TRAINING_ERROR',
                })
            finally:
                training_manager.stop()
                metrics_queue.put({'event': 'done'})

        thread = threading.Thread(target=thread_target, daemon=True)
        thread.start()

        yield {
            'event': 'status',
            'data': json.dumps({'status': 'started', 'message': '训练已启动'}, ensure_ascii=False),
        }

        while True:
            try:
                payload = metrics_queue.get(timeout=1.0)
            except queue.Empty:
                if not thread.is_alive() and metrics_queue.empty():
                    break
                continue

            event_type = payload.pop('event', 'metric')

            if event_type == 'metric':
                if 'epoch' in payload:
                    training_manager.update_epoch(payload['epoch'])
                yield {'event': 'metric', 'data': json.dumps(payload, ensure_ascii=False)}
            elif event_type == 'status':
                yield {'event': 'status', 'data': json.dumps(payload, ensure_ascii=False)}
            elif event_type == 'error':
                yield {'event': 'error', 'data': json.dumps(payload, ensure_ascii=False)}
                break
            elif event_type == 'done':
                if 'result' in result_container:
                    yield {
                        'event': 'completed',
                        'data': json.dumps(result_container['result'], ensure_ascii=False),
                    }
                break

    return EventSourceResponse(event_generator())
```

- [ ] **Step 6: 挂载上传样本目录为静态资源**

在 `server.py` 的静态文件挂载附近添加：

```python
upload_root = resolve_project_path(cfg_get('paths.temp_dir', './.cache')) / 'uploads'
upload_root.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(upload_root)), name="uploads")
```

这样前端可通过 `/uploads/training_xxx/train/good/0001.png` 直接访问样本缩略图。

- [ ] **Step 7: 启动服务器并测试训练端点**

Run: `python scripts/run_ui.py --no-browser`
Test with curl after uploading samples:
```bash
curl -N -X POST -H "Content-Type: application/json" \
  -d '{"model":"patchcore","dataset_path":".cache/uploads/training_xxx","category":"training_xxx","epochs":1}' \
  http://127.0.0.1:8000/api/train
```
Expected: SSE stream with status and metric events

- [ ] **Step 8: Commit**

```bash
git add modules/ui/server.py modules/ui/training_backend.py
git commit -m "feat(ui): 新增训练 SSE 端点、状态查询与停止接口"
```

---

### Task 5: 扩展 snap 结构为 4 页并新增训练页 HTML

**Files:**
- Modify: `modules/ui/static/index.html`

- [ ] **Step 1: 在 `sectionNames`（JS 中）预期之外，先修改 HTML 的 snap-dots**

`index.html` 中没有显式的 3 个 dot；dot 数量由 JS 动态读取 section 数量决定。因此只需在 `#s0` 之后、`#s2` 之前插入 `#s1-training`。

- [ ] **Step 2: 在 `index.html` 的 snap-container 中插入训练页 section**

在 `</section>`（#s0 结束）和 `<!-- ============================================== -->`（#s2 开始）之间插入：

```html
        <!-- ============================================== -->
        <!-- Section 1: Training Studio -->
        <!-- ============================================== -->
        <section id="s1-training" class="snap-page" x-ref="section1">
            <div class="snap-page-inner training-studio" x-data="training" x-init="init()">
                <div class="training-hero">
                    <div class="label">TRAINING STUDIO</div>
                    <h2 class="training-title">训练你的检测模型</h2>
                    <p class="training-subtitle">选择算法、上传正常样本，让模型只学习“什么是正常”。</p>
                </div>

                <div class="training-workspace">
                    <!-- 左侧配置卡 -->
                    <div class="training-config-card">
                        <h3 class="training-card-title">配置</h3>

                        <div class="training-field">
                            <div class="training-label">算法</div>
                            <div class="training-algo-pills">
                                <template x-for="m in models" :key="m.key">
                                    <div class="training-algo-pill"
                                         :class="{ 'is-active': selectedModel === m.key }"
                                         :style="`--algo-color: ${m.color}`"
                                         @click="selectedModel = m.key"
                                         x-text="m.name"></div>
                                </template>
                            </div>
                        </div>

                        <div class="training-field">
                            <div class="training-label">训练样本</div>
                            <div class="training-uploader"
                                 :class="{ 'is-dragover': isDragOver }"
                                 @dragover.prevent="isDragOver = true"
                                 @dragleave.prevent="isDragOver = false"
                                 @drop.prevent="onDropSamples($event)">
                                <div class="training-uploader-icon">+</div>
                                <div class="training-uploader-text">拖拽图片或点击上传</div>
                                <div class="training-uploader-count" x-text="`已选 ${sampleCount} / 150 张`"></div>
                                <input type="file" multiple accept="image/*" class="training-uploader-input"
                                       @change="onSelectSamples($event)" x-ref="sampleInput">
                            </div>
                        </div>

                        <div class="training-params-grid">
                            <div class="training-field">
                                <div class="training-label">Epoch</div>
                                <input type="number" class="training-input" x-model="epochs" min="1">
                            </div>
                            <div class="training-field">
                                <div class="training-label">Batch</div>
                                <input type="number" class="training-input" x-model="batchSize" min="1">
                            </div>
                            <div class="training-field">
                                <div class="training-label">LR</div>
                                <input type="number" class="training-input" x-model="learningRate" step="0.0001">
                            </div>
                            <div class="training-field">
                                <div class="training-label">Seed</div>
                                <input type="number" class="training-input" x-model="seed">
                            </div>
                        </div>

                        <button class="training-start-btn"
                                :disabled="trainingState === 'training' || sampleCount === 0"
                                @click="startTraining()">
                            <span x-show="trainingState !== 'training'">开始训练</span>
                            <span x-show="trainingState === 'training'">训练中...</span>
                        </button>
                        <button class="training-stop-btn" x-show="trainingState === 'training'" @click="stopTraining()">停止</button>
                    </div>

                    <!-- 右侧工作区 -->
                    <div class="training-right">
                        <div class="training-gallery-card">
                            <div class="training-card-header">
                                <h3 class="training-card-title">样本画廊</h3>
                                <span class="training-count" x-text="`已选择 ${sampleCount} 张`"></span>
                            </div>
                            <div class="training-gallery">
                                <template x-for="(sample, idx) in samples" :key="idx">
                                    <div class="training-thumb" :class="{ 'is-excluded': sample.excluded }"
                                         @click="toggleExclude(idx)">
                                        <img :src="sample.url" :alt="sample.name">
                                        <div class="training-thumb-overlay" x-show="sample.excluded">已排除</div>
                                    </div>
                                </template>
                            </div>
                        </div>

                        <div class="training-monitor-card">
                            <div class="training-card-header">
                                <h3 class="training-card-title">实时监控</h3>
                                <span class="training-status-badge"
                                      :class="`is-${trainingState}`"
                                      x-text="statusText"></span>
                            </div>
                            <div class="training-monitor-body">
                                <div class="training-chart-area">
                                    <canvas class="training-chart" x-ref="trainingChart"></canvas>
                                    <div class="training-chart-placeholder" x-show="!hasMetrics">训练开始后显示曲线</div>
                                </div>
                                <div class="training-metrics">
                                    <div class="training-metric">
                                        <div class="training-metric-label">Epoch</div>
                                        <div class="training-metric-value"><span x-text="currentEpoch"></span><span class="training-metric-delim">/</span><span x-text="totalEpochs"></span></div>
                                    </div>
                                    <div class="training-metric">
                                        <div class="training-metric-label">Loss</div>
                                        <div class="training-metric-value" x-text="latestLoss ?? '—'"></div>
                                    </div>
                                    <div class="training-metric">
                                        <div class="training-metric-label">LR</div>
                                        <div class="training-metric-value" x-text="latestLR ?? '—'"></div>
                                    </div>
                                    <div class="training-metric">
                                        <div class="training-metric-label">val AUROC</div>
                                        <div class="training-metric-value" x-text="latestAUROC ?? '—'"></div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </section>
```

- [ ] **Step 3: 引入 training.js**

在 `index.html` 底部 script 标签区域（在 `app.js` 之后）添加：

```html
<script src="/static/js/training.js"></script>
```

- [ ] **Step 4: Commit**

```bash
git add modules/ui/static/index.html
git commit -m "feat(ui): 新增 Training Studio HTML 结构与 4 页 snap 布局"
```

---

### Task 6: 更新 `app.js` 以支持 4 页并暴露全局状态

**Files:**
- Modify: `modules/ui/static/js/app.js`

- [ ] **Step 1: 更新 sectionNames**

将：
```javascript
sectionNames: ['算法介绍', '单模型推理', '四模型对比'],
```
替换为：
```javascript
sectionNames: ['算法介绍', '训练工作室', '单模型推理', '四模型对比'],
```

- [ ] **Step 2: 在 `init()` 中监听训练完成事件以刷新模型列表**

在 `init()` 函数中，在 `self.startHealthCheck()` 之后添加：

```javascript
window.addEventListener('training-completed', function () {
    self.fetchModels();
});
```

- [ ] **Step 3: 在导航栏数据集选择器旁添加“训练”快捷跳转**

在 `.navbar-right` 的 `.custom-select` 之前添加：

```html
<button class="navbar-jump-btn" @click="scrollToSection(1)">训练</button>
```

（样式在后续 CSS 任务中补充）

- [ ] **Step 4: Commit**

```bash
git add modules/ui/static/js/app.js
git commit -m "feat(ui): app.js 支持 4 页导航与训练页快捷入口"
```

---

### Task 7: 实现 `training.js`

**Files:**
- Create: `modules/ui/static/js/training.js`

- [ ] **Step 1: 创建 TrainingRunner**

```javascript
/**
 * TrainingRunner — SSE 训练客户端（基于 fetch ReadableStream，支持 POST 与中断）
 */
var TrainingRunner = {
    _abortController: null,

    run: function (payload, handlers) {
        var self = this;
        self.cancel();
        self._abortController = new AbortController();
        self._postStream('/api/train', payload, handlers, self._abortController.signal);
    },

    _postStream: function (url, payload, handlers, signal) {
        var self = this;
        fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
            signal: signal,
        }).then(function (response) {
            var reader = response.body.getReader();
            var decoder = new TextDecoder();
            var buffer = '';

            function read() {
                return reader.read().then(function (result) {
                    if (result.done) {
                        if (handlers.onDone) handlers.onDone();
                        return;
                    }
                    buffer += decoder.decode(result.value, { stream: true });
                    var lines = buffer.split('\n');
                    buffer = lines.pop();

                    var eventName = 'message';
                    var dataLines = [];
                    lines.forEach(function (line) {
                        if (line.startsWith('event:')) {
                            eventName = line.slice(6).trim();
                        } else if (line.startsWith('data:')) {
                            dataLines.push(line.slice(5).trim());
                        } else if (line === '') {
                            if (dataLines.length > 0) {
                                var data = JSON.parse(dataLines.join('\n'));
                                self._dispatch(eventName, data, handlers);
                            }
                            eventName = 'message';
                            dataLines = [];
                        }
                    });
                    return read();
                }).catch(function (err) {
                    if (err.name === 'AbortError') {
                        if (handlers.onDone) handlers.onDone();
                    } else if (handlers.onError) {
                        handlers.onError(String(err));
                    }
                });
            }
            return read();
        }).catch(function (err) {
            if (err.name !== 'AbortError' && handlers.onError) {
                handlers.onError(String(err));
            }
        });
    },

    _dispatch: function (event, data, handlers) {
        if (event === 'metric' && handlers.onMetric) handlers.onMetric(data);
        if (event === 'status' && handlers.onStatus) handlers.onStatus(data);
        if (event === 'completed' && handlers.onCompleted) handlers.onCompleted(data);
        if (event === 'error' && handlers.onError) handlers.onError(data.message || '训练失败');
    },

    cancel: function () {
        if (this._abortController) {
            this._abortController.abort();
            this._abortController = null;
        }
    },
};

    cancel: function () {
        if (this.es) {
            this.es.close();
            this.es = null;
        }
    },
};
```

- [ ] **Step 2: 创建 Alpine training data**

```javascript
document.addEventListener('alpine:init', function () {
    Alpine.data('training', function () {
        return {
            // 模型与算法
            models: [
                { key: 'patchcore', name: 'PatchCore', color: '#2997ff' },
                { key: 'padim', name: 'PaDiM', color: '#30d158' },
                { key: 'fre', name: 'FRE', color: '#ff9f0a' },
                { key: 'draem', name: 'DRAEM', color: '#bf5af2' },
            ],
            selectedModel: 'patchcore',

            // 样本
            samples: [],
            isDragOver: false,
            sessionId: null,
            datasetPath: null,
            category: null,

            // 参数
            epochs: 100,
            batchSize: 32,
            learningRate: 0.0001,
            seed: 42,

            // 训练状态
            trainingState: 'idle', // idle | uploading | training | completed | error
            currentEpoch: 0,
            totalEpochs: 0,
            latestLoss: null,
            latestLR: null,
            latestAUROC: null,
            etaSeconds: null,
            errorMessage: '',
            metricsHistory: [],

            init: function () {
                this.resetMonitor();
            },

            get sampleCount() {
                return this.samples.filter(function (s) { return !s.excluded; }).length;
            },

            get hasMetrics() {
                return this.metricsHistory.length > 0;
            },

            get statusText() {
                var map = {
                    idle: '就绪',
                    uploading: '上传中',
                    training: '训练中',
                    completed: '完成',
                    error: '错误',
                };
                return map[this.trainingState] || this.trainingState;
            },

            onSelectSamples: function (event) {
                this._uploadFiles(event.target.files);
            },

            onDropSamples: function (event) {
                this.isDragOver = false;
                this._uploadFiles(event.dataTransfer.files);
            },

            _uploadFiles: function (fileList) {
                var self = this;
                var files = Array.from(fileList).filter(function (f) { return f.type.startsWith('image/'); });
                if (files.length === 0) return;

                self.trainingState = 'uploading';
                var form = new FormData();
                files.forEach(function (f) { form.append('files', f); });

                fetch('/api/upload-samples', {
                    method: 'POST',
                    body: form,
                }).then(function (res) {
                    return res.json();
                }).then(function (data) {
                    self.sessionId = data.session_id;
                    self.datasetPath = data.dataset_path;
                    self.category = data.category;
                    self.samples = data.samples.map(function (name) {
                        return {
                            name: name,
                            url: '/uploads/' + self.category + '/train/good/' + name,
                            excluded: false,
                        };
                    });
                    self.trainingState = 'idle';
                }).catch(function (err) {
                    self.trainingState = 'error';
                    self.errorMessage = String(err);
                });
            },

            toggleExclude: function (idx) {
                this.samples[idx].excluded = !this.samples[idx].excluded;
            },

            resetMonitor: function () {
                this.currentEpoch = 0;
                this.totalEpochs = 0;
                this.latestLoss = null;
                this.latestLR = null;
                this.latestAUROC = null;
                this.etaSeconds = null;
                this.metricsHistory = [];
            },

            startTraining: function () {
                var self = this;
                if (!self.datasetPath) return;

                self.resetMonitor();
                self.trainingState = 'training';

                TrainingRunner.run({
                    model: self.selectedModel,
                    dataset_path: self.datasetPath,
                    category: self.category,
                    epochs: parseInt(self.epochs, 10),
                    batch_size: parseInt(self.batchSize, 10),
                    learning_rate: parseFloat(self.learningRate),
                    seed: parseInt(self.seed, 10),
                }, {
                    onMetric: function (data) {
                        self.currentEpoch = data.epoch;
                        self.totalEpochs = data.total_epochs;
                        if (data.train_loss !== undefined) self.latestLoss = data.train_loss.toFixed(4);
                        if (data.learning_rate !== undefined) self.latestLR = data.learning_rate.toExponential(2);
                        if (data.val_image_AUROC !== undefined) self.latestAUROC = (data.val_image_AUROC * 100).toFixed(1) + '%';
                        if (data.eta_seconds !== undefined) self.etaSeconds = data.eta_seconds;
                        self.metricsHistory.push(data);
                        self.drawChart();
                    },
                    onStatus: function (data) {
                        // 可选：显示状态消息
                    },
                    onCompleted: function (data) {
                        self.trainingState = 'completed';
                        // 通知全局应用刷新模型/数据集列表
                        window.dispatchEvent(new CustomEvent('training-completed', {
                            detail: { model: self.selectedModel, category: self.category }
                        }));
                    },
                    onError: function (msg) {
                        self.trainingState = 'error';
                        self.errorMessage = msg;
                    },
                });
            },

            stopTraining: function () {
                fetch('/api/train/stop', { method: 'POST' });
                TrainingRunner.cancel();
                this.trainingState = 'idle';
            },

            drawChart: function () {
                // 占位：后续 Task 中补充 canvas 绘制
            },
        };
    });
});
```

- [ ] **Step 3: Commit**

```bash
git add modules/ui/static/js/training.js
git commit -m "feat(ui): 新增 TrainingRunner 与 Alpine training 状态机"
```

---

### Task 8: 添加 Training Studio CSS

**Files:**
- Modify: `modules/ui/static/css/app.css`

- [ ] **Step 1: 在 `app.css` 末尾追加训练页样式**

```css
/* ═══════════════════════════════════════════════════════════════════════════════
   Training Studio
   ═══════════════════════════════════════════════════════════════════════════════ */

.training-studio {
    padding-top: 80px;
    padding-bottom: 40px;
    min-height: 100dvh;
    display: flex;
    flex-direction: column;
}

.training-hero {
    text-align: center;
    margin-bottom: 32px;
}

.training-title {
    font-size: clamp(28px, 4vw, 44px);
    font-weight: 600;
    margin: 8px 0 6px;
    letter-spacing: -0.02em;
}

.training-subtitle {
    font-size: 15px;
    color: var(--text-secondary);
}

.training-workspace {
    display: flex;
    gap: 24px;
    flex: 1;
    max-width: 1200px;
    width: 100%;
    margin: 0 auto;
    padding: 0 24px;
    align-items: flex-start;
}

.training-config-card,
.training-gallery-card,
.training-monitor-card {
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 24px;
    padding: 24px;
    backdrop-filter: blur(20px) saturate(160%);
}

.training-config-card {
    flex: 0 0 320px;
}

.training-right {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 20px;
}

.training-card-title {
    font-size: 17px;
    font-weight: 600;
    margin: 0 0 16px;
}

.training-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
}

.training-card-header .training-card-title {
    margin: 0;
}

.training-field {
    margin-bottom: 16px;
}

.training-label {
    font-size: 12px;
    color: var(--text-tertiary);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 8px;
}

.training-algo-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}

.training-algo-pill {
    padding: 8px 14px;
    border-radius: 999px;
    font-size: 13px;
    color: var(--text-secondary);
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    cursor: pointer;
    transition: all 0.2s ease;
}

.training-algo-pill.is-active {
    color: var(--algo-color, var(--accent));
    background: color-mix(in srgb, var(--algo-color, var(--accent)) 15%, transparent);
    border-color: color-mix(in srgb, var(--algo-color, var(--accent)) 50%, transparent);
}

.training-uploader {
    padding: 18px;
    border: 1px dashed rgba(255, 255, 255, 0.25);
    border-radius: 16px;
    text-align: center;
    background: rgba(0, 0, 0, 0.18);
    cursor: pointer;
    position: relative;
    transition: all 0.2s ease;
}

.training-uploader.is-dragover {
    border-color: var(--accent);
    background: rgba(41, 151, 255, 0.08);
}

.training-uploader-input {
    position: absolute;
    inset: 0;
    opacity: 0;
    cursor: pointer;
}

.training-uploader-icon {
    font-size: 28px;
    line-height: 1;
    margin-bottom: 4px;
}

.training-uploader-text {
    font-size: 13px;
    color: var(--text-secondary);
}

.training-uploader-count {
    font-size: 11px;
    color: var(--text-tertiary);
    margin-top: 4px;
}

.training-params-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
}

.training-input {
    width: 100%;
    padding: 10px 12px;
    background: rgba(0, 0, 0, 0.2);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    color: var(--text);
    font-size: 14px;
    outline: none;
    transition: border-color 0.2s ease;
}

.training-input:focus {
    border-color: rgba(255, 255, 255, 0.25);
}

.training-start-btn {
    width: 100%;
    padding: 14px;
    background: linear-gradient(135deg, #2997ff, #5ac8fa);
    border: none;
    border-radius: 999px;
    color: white;
    font-size: 15px;
    font-weight: 600;
    cursor: pointer;
    transition: opacity 0.2s ease;
}

.training-start-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
}

.training-stop-btn {
    width: 100%;
    margin-top: 10px;
    padding: 10px;
    background: rgba(255, 69, 58, 0.15);
    border: 1px solid rgba(255, 69, 58, 0.3);
    border-radius: 999px;
    color: #ff453a;
    font-size: 13px;
    cursor: pointer;
}

.training-gallery {
    display: flex;
    gap: 12px;
    overflow-x: auto;
    padding-bottom: 6px;
}

.training-thumb {
    width: 96px;
    height: 96px;
    border-radius: 12px;
    overflow: hidden;
    flex-shrink: 0;
    position: relative;
    cursor: pointer;
    background: rgba(0, 0, 0, 0.2);
}

.training-thumb img {
    width: 100%;
    height: 100%;
    object-fit: cover;
}

.training-thumb.is-excluded img {
    opacity: 0.35;
}

.training-thumb-overlay {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 12px;
    color: white;
    background: rgba(0, 0, 0, 0.5);
}

.training-monitor-body {
    display: flex;
    gap: 20px;
    height: 200px;
}

.training-chart-area {
    flex: 1;
    background: rgba(0, 0, 0, 0.2);
    border-radius: 16px;
    position: relative;
    overflow: hidden;
}

.training-chart {
    width: 100%;
    height: 100%;
}

.training-chart-placeholder {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--text-tertiary);
    font-size: 13px;
}

.training-metrics {
    flex: 0 0 180px;
    display: flex;
    flex-direction: column;
    gap: 12px;
}

.training-metric {
    flex: 1;
    background: rgba(0, 0, 0, 0.2);
    border-radius: 16px;
    padding: 14px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}

.training-metric-label {
    font-size: 12px;
    color: var(--text-tertiary);
    margin-bottom: 4px;
}

.training-metric-value {
    font-size: 26px;
    font-weight: 600;
}

.training-metric-delim {
    font-size: 14px;
    color: var(--text-tertiary);
    margin: 0 4px;
}

.training-status-badge {
    padding: 5px 10px;
    border-radius: 999px;
    font-size: 12px;
    background: rgba(48, 209, 88, 0.15);
    color: #30d158;
}

.training-status-badge.is-training {
    background: rgba(41, 151, 255, 0.15);
    color: #5ac8fa;
}

.training-status-badge.is-error {
    background: rgba(255, 69, 58, 0.15);
    color: #ff453a;
}

/* 导航栏训练跳转按钮 */
.navbar-jump-btn {
    padding: 6px 14px;
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 999px;
    color: var(--text-secondary);
    font-size: 13px;
    cursor: pointer;
    transition: all 0.2s ease;
}

.navbar-jump-btn:hover {
    background: rgba(255, 255, 255, 0.14);
    color: var(--text);
}
```

- [ ] **Step 2: Commit**

```bash
git add modules/ui/static/css/app.css
git commit -m "feat(ui): 新增 Training Studio 样式（玻璃卡、药丸标签、监控区）"
```

---

### Task 9: 绘制实时训练曲线

**Files:**
- Modify: `modules/ui/static/js/training.js`

- [ ] **Step 1: 替换 `drawChart` 占位函数**

```javascript
drawChart: function () {
    var canvas = this.$refs.trainingChart;
    if (!canvas) return;
    var ctx = canvas.getContext('2d');
    var dpr = window.devicePixelRatio || 1;
    var rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    var w = rect.width;
    var h = rect.height;
    var padding = 30;

    ctx.clearRect(0, 0, w, h);

    if (this.metricsHistory.length < 2) return;

    var losses = this.metricsHistory
        .filter(function (m) { return m.train_loss !== undefined && m.train_loss !== null; })
        .map(function (m) { return m.train_loss; });
    if (losses.length < 2) return;

    var maxLoss = Math.max.apply(null, losses);
    var minLoss = Math.min.apply(null, losses);
    var range = Math.max(0.001, maxLoss - minLoss);

    // 网格线
    ctx.strokeStyle = 'rgba(255,255,255,0.06)';
    ctx.lineWidth = 1;
    for (var i = 0; i <= 4; i++) {
        var y = padding + (h - 2 * padding) * i / 4;
        ctx.beginPath();
        ctx.moveTo(padding, y);
        ctx.lineTo(w - padding, y);
        ctx.stroke();
    }

    // Loss 曲线
    ctx.strokeStyle = '#5ac8fa';
    ctx.lineWidth = 2;
    ctx.beginPath();
    losses.forEach(function (loss, idx) {
        var x = padding + (w - 2 * padding) * idx / (losses.length - 1);
        var y = padding + (h - 2 * padding) * (1 - (loss - minLoss) / range);
        if (idx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // 轴标签
    ctx.fillStyle = 'rgba(255,255,255,0.4)';
    ctx.font = '10px sans-serif';
    ctx.fillText('Loss', padding, padding - 8);
    ctx.fillText('Epoch', w - padding - 28, h - padding + 16);
}
```

- [ ] **Step 2: 在 window resize 时重绘**

在 `init` 函数中添加：

```javascript
var self = this;
window.addEventListener('resize', function () {
    self.$nextTick(function () { self.drawChart(); });
});
```

- [ ] **Step 3: Commit**

```bash
git add modules/ui/static/js/training.js
git commit -m "feat(ui): 实现训练 loss 实时曲线绘制"
```

---

### Task 10: 添加 API 单元测试

**Files:**
- Create: `tests/test_training_api.py`

- [ ] **Step 1: 编写上传测试**

```python
import io
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image


def _make_image_bytes():
    img = Image.new('RGB', (64, 64), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()


def test_upload_samples_accepts_images(tmp_path):
    from modules.ui.server import app

    client = TestClient(app)
    data = _make_image_bytes()
    response = client.post(
        '/api/upload-samples',
        files={'files': ('sample.png', io.BytesIO(data), 'image/png')},
    )
    assert response.status_code == 200
    body = response.json()
    assert body['total'] == 1
    assert body['max_allowed'] == 150
    assert 'session_id' in body
    assert Path(body['dataset_path']).exists()


def test_upload_samples_rejects_non_image():
    from modules.ui.server import app

    client = TestClient(app)
    response = client.post(
        '/api/upload-samples',
        files={'files': ('readme.txt', io.BytesIO(b'hello'), 'text/plain')},
    )
    assert response.status_code == 400


def test_train_status_initially_idle():
    from modules.ui.server import app

    client = TestClient(app)
    response = client.get('/api/train-status')
    assert response.status_code == 200
    assert response.json()['running'] is False
```

- [ ] **Step 2: 运行测试**

Run: `python -m pytest tests/test_training_api.py -v`
Expected: 3 PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_training_api.py
git commit -m "test(ui): 新增 Training Studio API 单元测试"
```

---

### Task 11: 集成验证与修复

**Files:**
- 可能修改：`modules/ui/server.py`、`modules/ui/static/js/training.js`、`modules/ui/static/css/app.css`

- [ ] **Step 1: 启动 UI 并完整走通一次训练**

Run: `python scripts/run_ui.py`
Expected: 浏览器自动打开 http://127.0.0.1:8000

操作流程：
1. 滚动到第 2 页（训练工作室）
2. 选择算法 PatchCore
3. 上传 5-10 张正常图片
4. 设置 epoch=1
5. 点击“开始训练”
6. 观察 SSE 推送的 metric 事件和曲线
7. 训练完成后刷新 `/api/models`，确认新数据集出现

- [ ] **Step 2: 验证 4 页 snap 导航**

检查：
- 进度环显示 `1 / 4` 到 `4 / 4`
- 键盘 ↑↓ 可切换 4 页
- 导航栏 section 名称随页面变化

- [ ] **Step 3: 验证亮/暗主题覆盖训练页**

切换主题，确认训练卡片、输入框、按钮颜色正确变化。

- [ ] **Step 4: Commit 修复（如有）**

```bash
git add -A
git commit -m "fix(ui): Training Studio 集成验证修复"
```

---

## 自审清单

### Spec 覆盖
- [x] 本地上传训练样本 → Task 3 `/api/upload-samples`
- [x] 样本画廊/反选 → Task 7 `training.js` `samples` + `toggleExclude`
- [x] 算法选择 → Task 7 `training-algo-pills`
- [x] 超参数配置 → Task 7 输入框 + Task 4 `TrainRequest`
- [x] 前端启动训练 → Task 7 `startTraining` + Task 4 `/api/train`
- [x] SSE 实时 loss/LR/AUROC → Task 2 `TrainingMetricsCallback` + Task 4 SSE
- [x] 训练停止 → Task 4 `/api/train/stop`
- [x] 训练状态查询 → Task 4 `/api/train-status`
- [x] 模型对推理页可见 → Task 2 `run_training_job` 输出到 `results/`
- [x] 4 页 snap 结构 → Task 5 + Task 6

### Placeholder 扫描
- [x] 无 TBD/TODO
- [x] 无 "add appropriate error handling" 类模糊描述
- [x] 每步含代码或命令

### 类型一致性
- [x] `TrainRequest` 字段与 `training.js` 参数名一致
- [x] SSE event 名称在前后端一致（metric/status/completed/error/done）
- [x] `training_manager` API 在 Task 2 和 Task 4 中一致

---

## 执行交接

**Plan complete and saved to `docs/superpowers/plans/2026-06-20-training-studio.md`. Two execution options:**

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
