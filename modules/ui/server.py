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
import sys
import uuid
from pathlib import Path

import numpy as np

# ── Windows UTF-8 编码设置（必须在任何导入之前）──
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

from datetime import datetime
from typing import Dict, List

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from sse_starlette.sse import EventSourceResponse

# ── anomalib 导入（cv2 已在前面导入，DLL 加载顺序正确）──
from anomalib.data import PredictDataset

# ── 复用现有 Gradio 模块中的核心组件 ──
from modules.ui.demo import detector, MODEL_CONFIGS, get_available_datasets
from modules.ui import theme
from modules.config import get as cfg_get, get_threshold, get_data_config
from modules._runtime import resolve_project_path

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


# ── Cache-Control 中间件：静态资源缓存 1 小时 ──
class CacheControlMiddleware(BaseHTTPMiddleware):
    """为 /static/ 路径下的资源添加 Cache-Control 头。"""
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "public, max-age=3600"
        return response


app.add_middleware(CacheControlMiddleware)

# ── 静态文件挂载 ──
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


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
            "name": cfg.name,
            "direction": cfg.direction,
        })
    return {
        "models": models,
        "datasets": get_available_datasets(),
    }


@app.get("/api/theme/light-css")
async def get_light_css():
    """返回亮色模式 CSS 变量（用于前端动态加载）。"""
    css = theme.get_light_css()
    return Response(content=css, media_type="text/css")


# ============================================================================
# 共享推理辅助函数
# ============================================================================

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
        threshold = get_threshold(model_key, dataset)
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
        return EventSourceResponse(error_gen)

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

            # 调用共享推理逻辑
            result_data = _run_prediction(img, model_key, dataset)

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
        return EventSourceResponse(error_gen)

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
                # 调用共享推理逻辑
                result_data = _run_prediction(img, model_key, dataset)

                # 附加模型显示名（_run_prediction 已包含 model 和 model_name）
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
