"""
================================================================================
FastAPI 服务器 — Phase 2 Apple UI 重设计
================================================================================

提供 REST API + 静态文件服务，替代 Gradio 成为默认 UI 后端。

用法:
    python scripts/run_ui.py              # 默认 FastAPI → http://127.0.0.1:8000
    python scripts/run_ui.py --gradio     # Gradio fallback → http://127.0.0.1:7860
================================================================================
"""

import asyncio
import base64
import io
import json
import queue
import re
import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ── Windows UTF-8 编码设置（必须在任何导入之前）──
# pytest 运行时跳过，避免破坏其 stdout/stderr capture 机制
if "pytest" not in sys.modules:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# ── 项目根路径 ──
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# ── pycache 重定向 ──
from modules._runtime import configure_runtime_temp
configure_runtime_temp()

# ── cv2 必须在 anomalib 之前导入 ──
import cv2

from fastapi import FastAPI, Request, UploadFile, File, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware
from sse_starlette.sse import EventSourceResponse

# ── 轻量 UI 组件（无 anomalib/torch/gradio 依赖）──
from modules.ui._model_info import MODEL_CONFIGS, get_available_datasets, get_self_trained_models
from modules.ui._training_common import (
    format_uploaded_samples,
    training_manager,
    MAX_TRAIN_SAMPLES,
)
from modules.ui import theme
from modules.config import get as cfg_get, get_threshold, get_data_config
from modules._runtime import resolve_project_path

# detector 与 run_training_job 为 heavy 组件，在各自端点中延迟导入。

# ============================================================================
# FastAPI 应用实例
# ============================================================================

app = FastAPI(title="工业异常检测系统")

# ── CORS（开发环境允许所有来源）──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Cache-Control 中间件：每次使用前强制验证（no-cache）──
class CacheControlMiddleware(BaseHTTPMiddleware):
    """为 /static/ 路径下的资源添加 Cache-Control 头。

    使用 no-cache（而非 no-store）：浏览器会缓存但每次使用前
    必须先通过 ETag/If-None-Match 向服务器验证。文件未变则 304，
    带宽开销近乎为零。既保证实时更新，又保留缓存性能。
    """
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache"
        return response


app.add_middleware(CacheControlMiddleware)

# ── 静态文件挂载 ──
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# ── 上传样本目录挂载 ──
upload_root = resolve_project_path(cfg_get('paths.temp_dir', './.cache')) / 'uploads'
upload_root.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(upload_root)), name="uploads")


# ============================================================================
# 生命周期事件
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """启动时输出可用模型和数据集信息。"""
    datasets = get_available_datasets()
    print(f"[server] 可用数据集: {datasets}")
    print(f"[server] 可用模型: {list(MODEL_CONFIGS.keys())}")
    print(f"[server] 访问地址: http://127.0.0.1:8000")
    print(f"[server] API 文档: http://127.0.0.1:8000/docs")


# ============================================================================
# 路由：页面
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def index():
    """返回 Alpine.js SPA 入口页面。"""
    index_path = static_dir / "index.html"
    if not index_path.exists():
        return HTMLResponse(
            content="<h1>index.html 未找到</h1><p>请确保 modules/ui/static/index.html 存在。</p>",
            status_code=500,
        )
    return HTMLResponse(content=index_path.read_text(encoding='utf-8'))


# ============================================================================
# 路由：API
# ============================================================================

@app.get("/api/health")
async def health_check():
    """健康检查端点。"""
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/models")
async def list_models():
    """返回可用模型列表和数据集列表。"""
    models: List[Dict[str, str]] = []
    for key, cfg in MODEL_CONFIGS.items():
        models.append({
            "key": key,
            "name": cfg['name'],
            "direction": cfg['direction'],
        })
    return {
        "models": models,
        "datasets": get_available_datasets(),
    }


@app.get("/api/self-trained-models")
async def api_self_trained_models(model: str = Query(...)) -> dict:
    """
    返回指定模型的用户自训练模型列表。

    Args:
        model: 模型标识（patchcore / padim / fre / draem）。

    Returns:
        dict: {"models": [...]}，每个模型包含 path、category、version、display_name。

    Raises:
        HTTPException: 400 — 传入未知模型时抛出。
    """
    if model not in MODEL_CONFIGS:
        raise HTTPException(status_code=400, detail=f"未知模型: {model}")
    return {"models": get_self_trained_models(model)}


@app.get("/api/test-images")
async def api_test_images(dataset: str = Query(...)) -> dict:
    """
    返回指定数据集测试目录下的所有图片相对路径列表。

    Args:
        dataset: 数据集名称（如 bottle、region1 等）。

    Returns:
        dict: {"images": ["test/good/001.png", ...]}。

    Raises:
        HTTPException: 400 — 数据集名称包含非法字符或数据集不存在时抛出。
    """
    if not _is_safe_category(dataset):
        raise HTTPException(status_code=400, detail=f"非法数据集名称: {dataset}")

    data_root = resolve_project_path(cfg_get('paths.data_root', './data'))
    dataset_dir = data_root / dataset
    if not dataset_dir.is_dir():
        raise HTTPException(status_code=400, detail=f"数据集不存在: {dataset}")

    test_dir = dataset_dir / "test"
    if not test_dir.exists():
        return {"images": []}

    allowed_suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
    images = []
    for img_path in test_dir.rglob("*"):
        if img_path.is_file() and img_path.suffix.lower() in allowed_suffixes:
            try:
                rel = img_path.relative_to(dataset_dir).as_posix()
                images.append(rel)
            except ValueError:
                continue

    return {"images": sorted(images)}


@app.get("/api/theme/light-css")
async def get_light_css():
    """返回亮色模式 CSS 变量（用于前端动态加载）。"""
    css = theme.get_light_css()
    return Response(content=css, media_type="text/css")


# ============================================================================
# Pydantic 请求模型
# ============================================================================

class TrainRequest(BaseModel):
    """训练请求体。"""
    model: str
    dataset_path: str
    category: str
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.0001
    seed: int = 42
    excluded_samples: List[str] = []  # 被排除的样本文件名列表
    advanced_params: Dict[str, Any] = {}  # 模型特定高级参数


_CATEGORY_RE = re.compile(r"^[A-Za-z0-9_\-]+$")

def _is_safe_category(name: str) -> bool:
    """校验数据集/类别名称是否仅包含合法字符（字母、数字、下划线、连字符）。"""
    return bool(_CATEGORY_RE.match(name))


def _resolve_upload_dataset_path(dataset_path: str) -> Path:
    """
    解析训练请求中的数据集路径，并校验其必须位于上传目录下。

    Args:
        dataset_path: 请求传入的路径（相对或绝对）。

    Returns:
        Path: 绝对路径。

    Raises:
        HTTPException: 路径越界或不存在时抛出 400。
    """
    path = Path(dataset_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()

    upload_root = (resolve_project_path(cfg_get('paths.temp_dir', './.cache')) / 'uploads').resolve()
    try:
        path.relative_to(upload_root)
    except ValueError:
        raise HTTPException(status_code=400, detail="数据集路径不在允许的上传目录内")

    if not path.exists():
        raise HTTPException(status_code=400, detail="数据集路径不存在")
    return path


# ============================================================================
# 路由：训练状态查询
# ============================================================================

@app.get("/api/train-status")
async def train_status():
    """查询当前训练状态。"""
    return training_manager.to_dict()


# ============================================================================
# 路由：训练停止
# ============================================================================

@app.post("/api/train/stop")
async def train_stop():
    """请求停止当前训练任务。"""
    if not training_manager.is_running:
        return {"status": "idle"}
    training_manager.stop_event.set()
    return {"status": "stop_requested"}


# ============================================================================
# 路由：SSE 流式训练
# ============================================================================

@app.post("/api/train")
async def train(request: TrainRequest):
    """
    SSE 流式训练端点。
    接收训练参数，通过 SSE 推送状态、指标、日志和结果。
    """
    # 1. 参数校验
    if request.model not in MODEL_CONFIGS:
        raise HTTPException(status_code=400, detail=f"不支持的模型: {request.model}")
    if not _is_safe_category(request.category):
        raise HTTPException(status_code=400, detail="category 只能包含字母、数字、下划线和连字符")
    if request.epochs < 1 or request.epochs > 1000:
        raise HTTPException(status_code=400, detail="epochs 必须在 1-1000 之间")
    if request.batch_size < 1 or request.batch_size > 128:
        raise HTTPException(status_code=400, detail="batch_size 必须在 1-128 之间")
    if request.learning_rate <= 0 or request.learning_rate >= 1.0:
        raise HTTPException(status_code=400, detail="learning_rate 必须在 (0, 1) 之间")

    # 2. 解析并校验数据集路径
    dataset_path = _resolve_upload_dataset_path(request.dataset_path)

    # 2.1 读取上传时生成的计数器，并按 {模型名}-custom-{批次} 格式生成最终显示名称
    upload_dir = dataset_path.parents[1]
    display_name = request.category
    counter_file = upload_dir / ".counter"
    if counter_file.exists():
        try:
            counter = int(counter_file.read_text(encoding="utf-8").strip() or "1")
            display_name = f"{request.model}-custom-{counter:03d}"
        except ValueError:
            pass
    display_name_file = upload_dir / ".display_name"
    display_name_file.write_text(display_name, encoding="utf-8")

    # 3. 尝试获取全局训练锁
    started = training_manager.try_start(
        request.model, request.category, request.epochs
    )
    if not started:
        raise HTTPException(status_code=409, detail="已有训练任务正在运行")

    try:
        # 4. 创建指标队列
        metrics_queue = queue.Queue(maxsize=200)
        result_container: Dict[str, Any] = {}

        # 5. 在独立线程中执行训练（延迟导入 heavy 训练逻辑）
        def _training_thread():
            try:
                from modules.ui.training_backend import run_training_job
                result = run_training_job(
                    model_name=request.model,
                    dataset_path=dataset_path,
                    category=request.category,
                    epochs=request.epochs,
                    batch_size=request.batch_size,
                    learning_rate=request.learning_rate,
                    seed=request.seed,
                    excluded_samples=request.excluded_samples,
                    advanced_params=request.advanced_params,
                    display_name=display_name,
                    metrics_queue=metrics_queue,
                )
                result_container["result"] = result
            except Exception as e:
                import traceback
                traceback.print_exc()
                metrics_queue.put({
                    "event": "error",
                    "message": str(e),
                    "code": "TRAINING_EXCEPTION",
                })
            finally:
                metrics_queue.put({"event": "done"})

        thread = threading.Thread(target=_training_thread, daemon=True)
        thread.start()

        # 6. SSE 生成器
        async def event_generator():
            try:
                while True:
                    # 带超时的队列读取，避免训练线程异常时永久阻塞
                    try:
                        payload = await asyncio.to_thread(
                            metrics_queue.get, timeout=1.0
                        )
                    except queue.Empty:
                        if not thread.is_alive() and metrics_queue.empty():
                            break
                        continue

                    event_type = payload.get("event")

                    if event_type == "done":
                        # 训练结束，推送 completed 事件（如果有结果）
                        if "result" in result_container:
                            result = result_container["result"]
                            yield {
                                "event": "completed",
                                "data": json.dumps({
                                    "status": result.get("status"),
                                    "model": result.get("model"),
                                    "category": result.get("category"),
                                    "results": result.get("results"),
                                }, ensure_ascii=False),
                            }
                        break

                    elif event_type == "error":
                        yield {
                            "event": "error",
                            "data": json.dumps({
                                "message": payload.get("message"),
                                "code": payload.get("code", "UNKNOWN"),
                            }, ensure_ascii=False),
                        }

                    elif event_type == "status":
                        yield {
                            "event": "status",
                            "data": json.dumps({
                                "status": payload.get("status"),
                                "message": payload.get("message"),
                            }, ensure_ascii=False),
                        }

                    elif event_type == "metric":
                        yield {
                            "event": "metric",
                            "data": json.dumps({
                                "epoch": payload.get("epoch"),
                                "total_epochs": payload.get("total_epochs"),
                                "train_loss": payload.get("train_loss"),
                                "learning_rate": payload.get("learning_rate"),
                                "val_image_AUROC": payload.get("val_image_AUROC"),
                                "eta_seconds": payload.get("eta_seconds"),
                            }, ensure_ascii=False),
                        }

                    elif event_type == "log":
                        yield {
                            "event": "log",
                            "data": json.dumps({
                                "message": payload.get("message"),
                                "level": payload.get("level", "info"),
                                "timestamp": payload.get("timestamp"),
                            }, ensure_ascii=False),
                        }

                    elif event_type == "completed":
                        # run_training_job 内部也会推送 completed，兼容处理
                        yield {
                            "event": "completed",
                            "data": json.dumps({
                                "status": payload.get("status"),
                                "model": payload.get("model"),
                                "category": payload.get("category"),
                                "results": payload.get("results"),
                            }, ensure_ascii=False),
                        }

            finally:
                # 客户端断开或训练结束均释放训练锁
                training_manager.finish()

        return EventSourceResponse(event_generator())
    except Exception:
        # 创建 EventSourceResponse 或启动线程失败时立即释放锁
        training_manager.finish()
        raise


# ============================================================================
# /api/upload-samples — 上传训练样本并格式化为 MVTec AD 临时结构
# ============================================================================

@app.post("/api/upload-samples")
async def upload_samples(files: List[UploadFile] = File(...)) -> JSONResponse:
    """
    接收多张图片上传，保存为临时 MVTec AD 目录结构。

    Args:
        files: 图片文件列表（multipart/form-data）。

    Returns:
        JSONResponse: 包含 session_id、dataset_path、category、total、max_allowed、samples。

    Raises:
        HTTPException: 400 — 未上传文件、包含非图片文件、或没有有效图片。
    """
    # 1. 校验必须上传文件
    if not files:
        raise HTTPException(status_code=400, detail="未上传文件")

    # 2. 过滤非图片文件
    for f in files:
        content_type = f.content_type or ""
        if not content_type.startswith("image/"):
            raise HTTPException(
                status_code=400,
                detail=f"非图片文件: {f.filename} (type={content_type})",
            )

    # 3. 超过最大样本数时截断
    max_allowed = MAX_TRAIN_SAMPLES
    if len(files) > max_allowed:
        files = files[:max_allowed]

    # 4. 生成 session_id 与保存路径
    temp_dir = resolve_project_path(cfg_get("paths.temp_dir", "./.cache"))
    counter_file = temp_dir / "user_training_counter"
    counter = 1
    if counter_file.exists():
        try:
            counter = int(counter_file.read_text(encoding="utf-8").strip() or "1")
        except ValueError:
            counter = 1
    session_id = f"training_{uuid.uuid4().hex}"
    display_name = f"custom-{counter:03d}"
    counter_file.write_text(str(counter + 1), encoding="utf-8")

    upload_dir = temp_dir / "uploads" / session_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    # 保存计数器与临时显示名称，训练开始后会根据模型名重写为最终格式
    (upload_dir / ".counter").write_text(str(counter), encoding="utf-8")
    (upload_dir / ".display_name").write_text(f"custom-{counter:03d}", encoding="utf-8")

    # 5. 使用 cv2 读取并保存上传图片，读取失败的跳过
    saved_paths: List[Path] = []
    for file in files:
        try:
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            del nparr
            if img is None:
                continue
            # 保持原始扩展名，若无则默认 .png
            suffix = Path(file.filename or "img.png").suffix or ".png"
            dest_path = upload_dir / f"{uuid.uuid4().hex}{suffix}"
            cv2.imwrite(str(dest_path), img)
            saved_paths.append(dest_path)
        except (cv2.error, ValueError, OSError):
            continue
        finally:
            await file.close()

    # 6. 如果没有有效图片，返回 400
    if not saved_paths:
        raise HTTPException(status_code=400, detail="没有有效图片（读取或解码失败）")

    # 7. 调用 format_uploaded_samples 整理为 MVTec AD 结构（放到 user/{session_id} 下）
    dataset_path = format_uploaded_samples(
        upload_dir=upload_dir,
        image_files=saved_paths,
        max_samples=max_allowed,
        category=session_id,
    )

    # 8. 收集 train/good/ 下的文件名列表
    train_good_dir = dataset_path / "train" / "good"
    samples = sorted([p.name for p in train_good_dir.iterdir() if p.is_file()])

    # 9. 返回相对路径（若 temp_dir 在 PROJECT_ROOT 外则回退到绝对路径）
    try:
        dataset_path_str = str(dataset_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        dataset_path_str = str(dataset_path).replace("\\", "/")

    return JSONResponse(
        content={
            "session_id": session_id,
            "display_name": display_name,
            "dataset_path": dataset_path_str,
            "category": session_id,
            "total": len(samples),
            "max_allowed": max_allowed,
            "samples": samples,
        },
        status_code=200,
    )


# ============================================================================
# 共享推理辅助函数
# ============================================================================

def _parse_dataset(dataset: str) -> Tuple[str, str]:
    """解析数据集标识为 (source, category)。"""
    if dataset and '/' in dataset:
        source, category = dataset.split('/', 1)
        return source, category
    return 'default', dataset or 'bottle'


def _run_prediction(img: np.ndarray, model_key: str, dataset: str) -> dict:
    """
    执行单模型推理的共享逻辑，供 /api/predict 和 /api/compare 复用。

    Args:
        img: RGB 格式的输入图片 (H, W, C)。
        model_key: 模型标识 (patchcore/padim/fre/draem)。
        dataset: 数据集名称。

    Returns:
        dict: 包含 score/label/heatmap_b64/bboxes 等的结果数据。

    Raises:
        ValueError: 模型加载失败时抛出。
        RuntimeError: 推理过程失败时抛出。
    """
    # 延迟导入 heavy 组件，避免 server 模块导入阶段依赖 anomalib/gradio。
    from anomalib.data import PredictDataset
    from modules.ui.demo import detector

    # 加载模型
    success, msg = detector.load_model(model_key, dataset)
    if not success:
        raise ValueError(msg)

    # 保存临时文件（PredictDataset 需要文件路径）
    temp_dir = resolve_project_path(cfg_get('paths.temp_dir', './.cache')) / "predict"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / f'predict_{uuid.uuid4().hex}.png'
    cv2.imwrite(str(temp_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    try:
        # 创建预测数据集
        data_config = get_data_config(detector.current_model)
        image_size = tuple(data_config.get('image_size', [256, 256]))
        dataset_obj = PredictDataset(
            path=temp_path,
            image_size=image_size,
        )

        # 执行推理
        predictions = detector.engine.predict(
            model=detector.model,
            dataset=dataset_obj,
            ckpt_path=str(detector.current_checkpoint),
        )

        if not isinstance(predictions, list):
            predictions = list(predictions)

        if len(predictions) == 0:
            raise RuntimeError("推理未返回结果")

        # 提取结果
        pred = predictions[0]
        anomaly_map = pred.anomaly_map
        pred_score = float(pred.pred_score.cpu().max().item())
        pred_label = int(pred.pred_label)

        # Bbox 检测 + 热力图生成
        orig_h, orig_w = img.shape[:2]
        bboxes = (
            detector._apply_nms_to_map(anomaly_map, 0.3, orig_h, orig_w)
            if anomaly_map is not None else []
        )
        heatmap_overlay, anomaly_gray = detector._generate_heatmap(
            img, anomaly_map, bboxes=bboxes,
        )

        # Base64 编码
        _, orig_buf = cv2.imencode(
            '.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        _, heat_buf = cv2.imencode(
            '.png', cv2.cvtColor(heatmap_overlay, cv2.COLOR_RGB2BGR))
        _, gray_buf = cv2.imencode('.png', anomaly_gray)

        orig_b64 = 'data:image/png;base64,' + \
            base64.b64encode(orig_buf).decode()
        heat_b64 = 'data:image/png;base64,' + \
            base64.b64encode(heat_buf).decode()
        gray_b64 = 'data:image/png;base64,' + \
            base64.b64encode(gray_buf).decode()

        # 阈值与置信度
        _, category = _parse_dataset(dataset)
        threshold = get_threshold(model_key, category)
        is_anomaly = pred_score > threshold
        confidence = float(np.clip(
            pred_score if is_anomaly else 1 - pred_score,
            0.0, 1.0,
        ))

        return {
            "model": model_key,
            "score": pred_score,
            "label": pred_label,
            "is_anomaly": is_anomaly,
            "threshold": threshold,
            "confidence": confidence,
            "image_b64": orig_b64,
            "heatmap_b64": heat_b64,
            "anomaly_map_b64": gray_b64,
            "bboxes": bboxes,
            "model_name": MODEL_CONFIGS[model_key].name,
        }

    finally:
        # 清理临时文件
        temp_path.unlink(missing_ok=True)


# ============================================================================
# /api/predict — 单模型 SSE 流式推理
# ============================================================================

@app.post("/api/predict")
async def predict(request: Request):
    """
    SSE 流式推理端点。
    接收上传图片，通过 SSE 推送加载进度、推理进度和最终结果。
    """
    form = await request.form()
    image_file = form.get("image")
    model_key = form.get("model", "patchcore")
    dataset = form.get("dataset", "bottle")

    if image_file is None:
        async def error_gen():
            yield {"event": "error", "data": json.dumps(
                {"message": "未上传图片"}, ensure_ascii=False)}
        return EventSourceResponse(error_gen())

    # 读取上传图片
    contents = await image_file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        async def error_gen():
            yield {"event": "error", "data": json.dumps(
                {"message": "无法解码图片，请确认文件格式正确（PNG/JPG/BMP）"}, ensure_ascii=False)}
        return EventSourceResponse(error_gen())

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    async def event_generator():
        # ── 阶段 1: 加载模型 ──
        model_name = MODEL_CONFIGS.get(model_key, MODEL_CONFIGS['patchcore']).name
        yield {
            "event": "progress",
            "data": json.dumps({
                "stage": "loading_model",
                "message": f"正在加载 {model_name}...",
                "pct": 10,
            }, ensure_ascii=False),
        }
        await asyncio.sleep(0.1)

        # ── 阶段 2: 预处理 ──
        yield {
            "event": "progress",
            "data": json.dumps({
                "stage": "preprocess",
                "message": "正在预处理图片...",
                "pct": 30,
            }, ensure_ascii=False),
        }
        await asyncio.sleep(0.1)

        try:
            # ── 阶段 3: 推理 ──
            yield {
                "event": "progress",
                "data": json.dumps({
                    "stage": "inference",
                    "message": "正在推理...",
                    "pct": 60,
                }, ensure_ascii=False),
            }
            await asyncio.sleep(0.1)

            # 调用共享推理逻辑（线程池执行，避免阻塞事件循环导致 SSE 断流）
            result_data = await asyncio.to_thread(_run_prediction, img, model_key, dataset)

            # ── 阶段 4: 后处理 ──
            yield {
                "event": "progress",
                "data": json.dumps({
                    "stage": "postprocess",
                    "message": "正在生成热力图...",
                    "pct": 80,
                }, ensure_ascii=False),
            }
            await asyncio.sleep(0.1)

            yield {
                "event": "progress",
                "data": json.dumps({
                    "stage": "done",
                    "message": "推理完成",
                    "pct": 100,
                }, ensure_ascii=False),
            }
            await asyncio.sleep(0.05)

            yield {
                "event": "result",
                "data": json.dumps(result_data, ensure_ascii=False),
            }

        except ValueError as e:
            yield {
                "event": "error",
                "data": json.dumps({"message": str(e)}, ensure_ascii=False),
            }
            return

        except Exception as e:
            import traceback
            traceback.print_exc()
            yield {
                "event": "error",
                "data": json.dumps({"message": f"推理失败: {str(e)}"}, ensure_ascii=False),
            }
            return

        yield {"event": "done", "data": "{}"}

    return EventSourceResponse(event_generator())


# ============================================================================
# /api/compare — 四模型对比 SSE 流式推理
# ============================================================================

@app.post("/api/compare")
async def compare(request: Request):
    """
    四模型对比 SSE 端点。
    接收上传图片，依次对 4 种算法执行推理，
    通过 SSE 推送每个模型的结果和最终排名摘要。
    """
    form = await request.form()
    image_file = form.get("image")
    dataset = form.get("dataset", "bottle")

    if image_file is None:
        async def error_gen():
            yield {"event": "error", "data": json.dumps(
                {"message": "未上传图片"}, ensure_ascii=False)}
        return EventSourceResponse(error_gen())

    # 读取上传图片（与 /api/predict 相同）
    contents = await image_file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        async def error_gen():
            yield {"event": "error", "data": json.dumps(
                {"message": "无法解码图片，请确认文件格式正确（PNG/JPG/BMP）"}, ensure_ascii=False)}
        return EventSourceResponse(error_gen())

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    async def event_generator():
        models = ['patchcore', 'padim', 'fre', 'draem']
        results: List[dict] = []

        for model_key in models:
            # 通知前端当前模型开始推理
            model_name = MODEL_CONFIGS[model_key].name
            yield {
                "event": "model_start",
                "data": json.dumps({
                    "model": model_key,
                    "name": model_name,
                }, ensure_ascii=False),
            }
            await asyncio.sleep(0.1)

            try:
                # 调用共享推理逻辑（线程池执行，避免阻塞事件循环）
                result_data = await asyncio.to_thread(_run_prediction, img, model_key, dataset)
                results.append(result_data)
                yield {
                    "event": "model_result",
                    "data": json.dumps(result_data, ensure_ascii=False),
                }

            except ValueError as e:
                yield {
                    "event": "model_error",
                    "data": json.dumps({
                        "model": model_key,
                        "name": model_name,
                        "message": str(e),
                    }, ensure_ascii=False),
                }

            except Exception as e:
                import traceback
                traceback.print_exc()
                yield {
                    "event": "model_error",
                    "data": json.dumps({
                        "model": model_key,
                        "name": model_name,
                        "message": f"推理失败: {str(e)}",
                    }, ensure_ascii=False),
                }

        # 生成排名摘要（得分最低 = 最正常 = 最优）
        if results:
            results_sorted = sorted(results, key=lambda r: r.get('score', 1.0))
            best = results_sorted[0]
            summary = {
                "best_model": best["model"],
                "best_name": best["model_name"],
                "best_score": best["score"],
                "ranking": [
                    {
                        "model": r["model"],
                        "name": r["model_name"],
                        "score": r["score"],
                    }
                    for r in results_sorted
                ],
            }
            yield {
                "event": "summary",
                "data": json.dumps(summary, ensure_ascii=False),
            }

        yield {"event": "done", "data": "{}"}

    return EventSourceResponse(event_generator())
