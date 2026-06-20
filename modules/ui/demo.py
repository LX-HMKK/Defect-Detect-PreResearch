"""
================================================================================
模块 4: UI 界面演示模块 (UI Demo Module) - Anomalib 2.x
================================================================================

功能: 使用 Gradio 构建交互式 Web 界面，用于展示算法推理结果

界面组件:
    1. 下拉菜单: 切换 3 种算法 (AutoEncoder / PatchCore / DRAEM)
    2. 上传按钮: 选择/上传测试图片
    3. 开始推理按钮: 执行异常检测
    4. 图片展示区: 并排显示原图和缺陷热力图

使用方式:
    python modules/ui/demo.py
    然后访问 http://127.0.0.1:7860

或者从根目录启动:
    python scripts/run_ui.py
================================================================================
"""

import base64
import io
import json
import uuid
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np

import cv2
import gradio as gr
import torch

# anomalib 2.x 导入
from anomalib.engine import Engine
from anomalib.data import PredictDataset
from anomalib.models import (
    Patchcore,
    Draem,
    Fre,
    Padim,
)

# 配置管理
from modules._runtime import resolve_project_path
from modules.config import get, get_threshold, get_model_config, get_data_config

# 主题管理器
from modules.ui import theme

# 忽略警告
warnings.filterwarnings('ignore')


# ================================================================================
# 配置常量（从配置文件读取）
# ================================================================================
# 移除硬编码的阈值配置，改为从 configs/config.yaml 和训练结果动态读取

@dataclass
class ModelConfig:
    """模型配置数据类"""
    name: str
    direction: str          # 算法方向
    description: str        # 详细描述
    weight_path: str        # 权重文件路径
    model_class: type       # 模型类
    model_kwargs: dict = None  # 模型初始化参数


# 3 种算法的配置
MODEL_CONFIGS = {
    'fre': ModelConfig(
        name='FRE',
        direction='基于特征重构',
        description='''
**算法原理**: 预训练CNN(ResNet50)提取特征，线性自编码器重构特征，
重构误差作为异常分数（误差大=异常）。

**特点**:
- 重构法改进版，效果优秀
- 支持像素级定位
- 训练快速，效果优秀
''',
        weight_path='./results/fre/Fre/MVTec/bottle/v0/weights/lightning/model.ckpt',
        model_class=Fre,
        model_kwargs={
            'backbone': 'resnet50',
            'layer': 'layer3',
            'pre_trained': True,
        },
    ),
    'patchcore': ModelConfig(
        name='PatchCore',
        direction='基于特征建模',
        description='''
**算法原理**: 使用预训练 CNN 提取局部特征，构建记忆库存储正常样本特征。
测试时通过计算测试样本特征与记忆库的最近邻距离来判断异常。

**特点**:
- 无需训练，直接构建特征记忆库
- 工业界目前效果最好的方法
- 推理速度最快，适合实时检测
''',
        weight_path='./results/patchcore/Patchcore/MVTec/bottle/v0/weights/lightning/model.ckpt',
        model_class=Patchcore,
        model_kwargs={},
    ),
    'draem': ModelConfig(
        name='DRAEM',
        direction='基于自监督学习',
        description='''
**算法原理**: 在训练阶段通过数据增强生成合成异常样本，
训练一个判别网络来学习区分正常区域和异常区域。

**特点**:
- 无需真实异常样本即可训练
- 对小缺陷检测效果较好
- 推理速度较慢，但定位精度高
''',
        weight_path='./results/draem/Draem/MVTec/bottle/v1/weights/lightning/model.ckpt',
        model_class=Draem,
        model_kwargs={},
    ),
    'padim': ModelConfig(
        name='PaDiM',
        direction='基于特征建模（概率分布）',
        description='''
**算法原理**: 对每个 patch 位置建立多元高斯分布模型，
通过马氏距离度量测试样本与正常分布的偏离程度。

**特点**:
- 与 PatchCore 同属特征建模类，但使用概率建模
- 无需训练，仅需一次前向传播构建统计量
- 推理速度快，内存占用适中
''',
        weight_path='./results/padim/Padim/MVTec/bottle/v0/weights/lightning/model.ckpt',
        model_class=Padim,
        model_kwargs={
            'backbone': 'resnet18',
            'layers': ['layer1', 'layer2', 'layer3'],
            'pre_trained': True,
        },
    )
}



def get_available_datasets():
    """自动检测可用的数据集（支持 MVTec AD 与 Folder 两种输出结构）。"""
    results_dir = Path("./results")
    datasets = set()
    model_dirs = {
        "fre": "Fre",
        "patchcore": "Patchcore",
        "draem": "Draem",
        "padim": "Padim",
    }
    for model_key, subdir in model_dirs.items():
        model_path = results_dir / model_key / subdir
        if not model_path.exists():
            continue

        # 1) MVTec AD 结构: results/{model}/Patchcore/MVTec/{category}
        mvtec_path = model_path / "MVTec"
        if mvtec_path.exists():
            for cat_dir in mvtec_path.iterdir():
                if cat_dir.is_dir() and cat_dir.name not in ["__pycache__"]:
                    datasets.add(cat_dir.name)

        # 2) Folder 结构: results/{model}/Patchcore/{category}/v0/weights
        for cat_dir in model_path.iterdir():
            if not cat_dir.is_dir() or cat_dir.name in ["__pycache__", "MVTec"]:
                continue
            if any(
                child.is_dir() and child.name.startswith("v")
                for child in cat_dir.iterdir()
            ):
                datasets.add(cat_dir.name)

    return sorted(list(datasets))


# ================================================================================
# 异常检测器类
# ================================================================================

# NMS bbox detection: use a SEPARATE lower threshold from classification threshold
# Classification threshold (from get_threshold) decides if image is anomalous
# NMS threshold decides where to draw bounding boxes on the heatmap
# Empirical: anomaly_map values are typically 0.3-0.6 max, far below classification threshold (0.8-0.9)
NMS_BBOX_THRESHOLD = 0.3  # Lower threshold to detect actual anomaly regions on map

class AnomalyDetector:
    """
    异常检测器
    
    管理模型加载和推理，支持3种算法切换
    """
    
    def __init__(self):
        self.current_model: Optional[str] = None
        self.current_dataset: Optional[str] = None
        self.current_checkpoint: Optional[Path] = None
        self.model = None
        self.engine = None

    def _resolve_weight_path(self, model_key: str, dataset: str) -> Optional[Path]:
        """Resolve checkpoint path with model/category priority."""
        from modules.algorithm import find_latest_checkpoint

        latest_dataset = find_latest_checkpoint("./results", model_key, dataset)
        if latest_dataset and latest_dataset.exists():
            return latest_dataset

        fallback = Path(MODEL_CONFIGS[model_key].weight_path)
        if fallback.exists() and dataset in str(fallback):
            return fallback

        return None

    def load_model(self, model_key: str, dataset: str = None) -> Tuple[bool, str]:
        """
        加载指定模型
        
        Args:
            model_key: 模型标识 (fre/patchcore/draem)
            dataset: 数据集名称 (region1/bottle)
        
        Returns:
            Tuple[bool, str]: (是否成功, 状态信息)
        """
        if dataset is None:
            dataset = "region1"
        
        # 如果模型和数据都已加载，直接返回
        if model_key == self.current_model and self.current_dataset == dataset and self.model is not None:
            return True, f"[OK] 模型已加载: {MODEL_CONFIGS[model_key].name} ({dataset})"
        
        ui_config = MODEL_CONFIGS.get(model_key)
        config = ui_config
        if ui_config is None:
            return False, f"[FAIL] 未知模型: {model_key}"
        
        # 查找权重文件 - 优先查找对应数据集的权重
        weight_path = self._resolve_weight_path(model_key, dataset)
        
        # 如果默认路径不存在或数据集不匹配，搜索对应数据集的权重
        if weight_path is None:
            search_base = Path('./results')
            
            # 首先尝试查找对应数据集的权重
            for subdir in ['Fre', 'Patchcore', 'Draem', 'Padim']:
                model_subdir = search_base / model_key / subdir / 'MVTec' / dataset
                if model_subdir.exists():
                    ckpt_files = list(model_subdir.glob('**/lightning/model.ckpt'))
                    if ckpt_files:
                        weight_path = max(ckpt_files, key=lambda p: p.stat().st_mtime)
                        break
            

        if weight_path is None or not weight_path.exists():
            return False, (
                f"[FAIL] 未找到模型权重: {config.weight_path}\n\n"
                f"请先训练模型:\n"
                f"```bash\n"
                f"python modules/algorithm/trainer.py --model {model_key} --category <your_category> --data_path <data_path>\n"
                f"```"
            )
        
        try:
            # 创建模型实例（使用配置的自定义参数）
            from modules.algorithm import get_model_from_config
            model_config = get_model_config(model_key) or None
            self.model = get_model_from_config(model_key, model_config)
            
            temp_dir = resolve_project_path(get('paths.temp_dir', './.cache'))
            self.engine = Engine(
                default_root_dir=str(temp_dir / "lightning_logs"),
                logger=False,
                enable_progress_bar=False,
            )
            
            self.model.eval()
            self.current_model = model_key
            self.current_dataset = dataset
            
            self.current_checkpoint = weight_path
            return True, f"[OK] 成功加载 {config.name} ({dataset})"
        
        except Exception as e:
            return False, f"[FAIL] 模型加载失败: {str(e)}"
    
    def predict(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        执行异常检测
        
        Args:
            image: 输入图片 (H, W, C) RGB格式
        
        Returns:
            Tuple: (原图, 热力图, 结果文本)
        """
        if self.model is None or self.engine is None:
            return image, image, "[FAIL] 模型未加载"
        if self.current_checkpoint is None:
            return image, image, "[FAIL] 未找到当前模型 checkpoint"
        
        temp_path: Optional[Path] = None

        try:
            # 确保图片格式正确
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            
            # 保存临时文件用于 PredictDataset（传入文件路径而不是目录）
            temp_dir = resolve_project_path(get('paths.temp_dir', './.cache')) / "predict"
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_path = temp_dir / f'predict_{uuid.uuid4().hex}.png'
            cv2.imwrite(str(temp_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            
            # 创建预测数据集 - 传入文件路径
            data_config = get_data_config(self.current_model)
            image_size = tuple(data_config.get('image_size', [256, 256]))
            dataset = PredictDataset(
                path=temp_path,  # 传入文件路径而不是目录
                image_size=image_size,
            )
            
            # 执行推理
            predictions = self.engine.predict(
                model=self.model,
                dataset=dataset,
                ckpt_path=str(self.current_checkpoint),
            )
            
            # 获取预测结果
            if predictions is not None:
                # 转换为列表（如果是生成器）
                if not isinstance(predictions, list):
                    predictions = list(predictions)
                
                if len(predictions) > 0:
                    prediction = predictions[0]
                    
                    # 提取结果
                    anomaly_map = prediction.anomaly_map
                    # pred.pred_score 可能是多元素 tensor，取最大值作为图像级得分
                    pred_score = float(prediction.pred_score.cpu().max().item())
                    pred_label = int(prediction.pred_label)
                    
                    # 根据阈值和 NMS 过滤，得到 bbox 列表
                    # 将 anomaly_map 缩放到原始图片尺寸，在原图坐标空间内计算 bbox
                    # 注意：使用独立的 NMS_BBOX_THRESHOLD (0.3) 而非 classification threshold
                    # 原因：anomaly_map.max() 通常为 0.3-0.6，远低于 classification threshold (0.8-0.9)
                    orig_h, orig_w = image.shape[:2]
                    bboxes = self._apply_nms_to_map(anomaly_map, NMS_BBOX_THRESHOLD, orig_h, orig_w) if anomaly_map is not None else []
                    # 生成热力图和原始灰度图（供前端 hover 交互读取像素值）
                    heatmap, anomaly_gray = self._generate_heatmap(image, anomaly_map, bboxes=bboxes)
                    
                    # 将灰度图编码为 base64 PNG
                    from PIL import Image as PILImage
                    gray_img = PILImage.fromarray(anomaly_gray, mode='L')
                    buf = io.BytesIO()
                    gray_img.save(buf, format='PNG')
                    anomaly_map_b64 = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()
                    
                    # bbox 坐标编码为 JSON
                    bboxes_json = json.dumps(bboxes, ensure_ascii=False) if bboxes else '[]'
                    
                    # 生成结果文本（含隐藏数据）
                    result_text = self._format_result(pred_score, pred_label,
                                                      anomaly_map_b64=anomaly_map_b64,
                                                      bboxes_json=bboxes_json)
                    
                    return image, heatmap, result_text
            
            return image, image, "[FAIL] 推理失败: 未获取到预测结果"
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            return image, image, f"[FAIL] 推理失败: {str(e)}"
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
    
    def _generate_heatmap(
        self,
        original: np.ndarray,
        anomaly_map: torch.Tensor,
        bboxes: Optional[List[Tuple[int, int, int, int, float]]] = None
    ) -> np.ndarray:
        """生成异常热力图"""
        # 将 tensor 转换为 numpy
        if isinstance(anomaly_map, torch.Tensor):
            anomaly_map = anomaly_map.cpu().numpy()
        
        # 如果是多通道，取第一个通道
        if len(anomaly_map.shape) > 2:
            anomaly_map = anomaly_map[0]
        
        # 归一化
        anomaly_norm = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
        anomaly_uint8 = (anomaly_norm * 255).astype(np.uint8)
        
        # 调整尺寸
        h, w = original.shape[:2]
        anomaly_resized = cv2.resize(anomaly_uint8, (w, h))
        
        # 应用颜色映射 (JET: 蓝->绿->黄->红)
        heatmap_colored = cv2.applyColorMap(anomaly_resized, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # 叠加
        overlay = cv2.addWeighted(original, 0.5, heatmap_colored, 0.5, 0)
        # 如有 bbox，绘制在热力图上（红色边框）
        if bboxes:
            overlay = self._draw_bboxes(overlay, bboxes, color=(255, 0, 0))
        
        return overlay, anomaly_resized
    
    def _format_result(self, score: float, label: int,
                        anomaly_map_b64: str = "",
                        bboxes_json: str = "") -> str:
        """格式化结果 — Apple 极简面板，逐层入场动画"""
        if self.current_model is None or self.current_model not in MODEL_CONFIGS:
            return '<div class="result-card" style="text-align:center;padding:48px 28px;"><div style="font-size:15px;color:var(--text-tertiary);">模型未加载</div></div>'
        model_config = MODEL_CONFIGS[self.current_model]

        dataset = self.current_dataset or "bottle"
        threshold = get_threshold(self.current_model, dataset)

        is_anomaly = score > threshold
        confidence = score if is_anomaly else 1 - score
        score_width = float(np.clip(score, 0.0, 1.0) * 100)
        threshold_width = float(np.clip(threshold, 0.0, 1.0) * 100)
        confidence_display = float(np.clip(confidence, 0.0, 1.0))
        confidence_width = confidence_display * 100

        status_color = "var(--bad)" if is_anomaly else "var(--ok)"
        status_bg = "var(--bad-bg)" if is_anomaly else "var(--ok-bg)"
        status_label = "异常" if is_anomaly else "正常"
        operator = ">" if is_anomaly else "<"

        return f"""
<div class="result-card fade-in">
    <!-- 标题 + 状态 -->
    <div class="reveal-child-1" style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 32px;">
        <div>
            <div style="font-size:12px;font-weight:500;color:var(--text-tertiary);letter-spacing:0.02em;margin-bottom:4px;">检测模型</div>
            <div style="font-size:20px;font-weight:600;color:var(--text);letter-spacing:-0.01em;">{model_config.name}</div>
        </div>
        <div class="status-badge {'anomaly' if is_anomaly else 'normal'}">{status_label}</div>
    </div>

    <!-- 双列数字 -->
    <div class="reveal-child-2" style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 24px;">
        <div class="core-metric">
            <div class="label">异常得分</div>
            <div class="value {'anomaly' if is_anomaly else 'normal'}">{score:.4f}</div>
            <div class="progress-container">
                <div class="progress-bar-mini">
                    <div class="progress-fill {'anomaly' if is_anomaly else 'normal'}" style="width: {score_width}%;"></div>
                </div>
                <div class="threshold-line">
                    <div class="threshold-marker" style="left: {threshold_width}%;"></div>
                    <div class="threshold-label" style="left: {threshold_width}%;">τ {threshold:.3f}</div>
                </div>
            </div>
        </div>
        <div class="core-metric">
            <div class="label">置信度</div>
            <div class="value" style="color: var(--accent);">{confidence_display:.1%}</div>
            <div class="progress-container">
                <div class="progress-bar-mini">
                    <div class="progress-fill" style="width: {confidence_width}%; background: var(--accent);"></div>
                </div>
            </div>
        </div>
    </div>

    <!-- 判决 -->
    <div class="reveal-child-3" style="background: {status_bg}; border-radius: var(--r-md); padding: 16px 20px; display: flex; align-items: flex-start; gap: 12px;">
        <div style="font-size:18px;color:{status_color};line-height:1;">{'●' if is_anomaly else '●'}</div>
        <div>
            <div style="font-size:11px;font-weight:500;color:var(--text-tertiary);margin-bottom:4px;letter-spacing:0.02em;">判决</div>
            <div style="font-size:14px;color:var(--text-secondary);line-height:1.6;">
                得分 <b style="color:{status_color};">{score:.4f}</b> {operator} 阈值 <b>τ = {threshold:.3f}</b> → <b style="color:{status_color};">{status_label}</b>
            </div>
        </div>
    </div>

    <!-- 隐藏数据：异常灰度图 + bbox 坐标（供 inference-interact.js 读取） -->
    """ + (f"""<img id="anomaly-map-data" src="{anomaly_map_b64}" style="display:none" onerror="this.style.display='none'">
    <div id="bbox-data" data-bboxes='{bboxes_json}' style="display:none"></div>
    <div id="heatmap-tooltip" class="heatmap-tooltip"></div>""" if anomaly_map_b64 else "") + """
</div>
"""


    def _apply_nms_to_map(self, anomaly_map: torch.Tensor, threshold: float, target_h: int, target_w: int) -> List[Tuple[int, int, int, int, float]]:
        """Apply thresholding and NMS on anomaly_map to obtain bounding boxes.
        Computes bboxes at model resolution (256x256), then scales coordinates
        to target image size for correct alignment.
        Returns list of (x, y, w, h, score) in target image coordinates.
        """
        # Convert anomaly_map to numpy at model resolution
        if isinstance(anomaly_map, torch.Tensor):
            map_np = anomaly_map.cpu().numpy()
        else:
            map_np = anomaly_map
        if map_np.ndim == 3 and map_np.shape[0] == 1:
            map_np = map_np[0]
        
        map_h, map_w = map_np.shape[:2]
        
        # Binary mask at model resolution
        binary = (map_np > float(threshold)).astype(np.uint8)
        # Find contours
        cnts = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = cnts[0] if len(cnts) == 2 else cnts[1]
        
        if not contours:
            return []
        
        boxes = []
        # min_area at model resolution (256x256). Lower from 100 to 30 to catch small defects
        # Also filter out full-image bboxes (which are false positives from threshold artifacts)
        min_area = 30
        max_area_ratio = 0.8  # Filter out bboxes covering >80% of the image
        max_area = int(map_h * map_w * max_area_ratio)
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            if area < min_area or area > max_area:
                continue
            # Score: max anomaly score in the region at model resolution
            region = map_np[y:y+h, x:x+w]
            score = float(np.max(region)) if region.size > 0 else 0.0
            # Scale bbox coordinates to target image size
            sx = x / map_w * target_w
            sy = y / map_h * target_h
            sw = w / map_w * target_w
            sh = h / map_h * target_h
            boxes.append((int(sx), int(sy), int(sw), int(sh), score))
        
        if not boxes:
            return []
        
        # NMS: sort by score descending
        boxes.sort(key=lambda b: b[4], reverse=True)
        kept = []
        while boxes:
            best = boxes.pop(0)
            kept.append(best)
            rem = []
            for bb in boxes:
                iou = self._iou(best, bb)
                if iou < 0.3:
                    rem.append(bb)
            boxes = rem
        return kept

    def _draw_bboxes(self, image: np.ndarray, bboxes: List[Tuple[int,int,int,int,float]], color=(255,0,0)) -> np.ndarray:
        """Draw bounding boxes on RGB image. Returns image with red boxes."""
        if image is None:
            return image
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        for (x, y, w, h, s) in bboxes:
            cv2.rectangle(img_bgr, (int(x), int(y)), (int(x+w), int(y+h)), (0, 0, 255), 2)
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    def _iou(self, a: Tuple[int,int,int,int,float], b: Tuple[int,int,int,int,float]) -> float:
        ax, ay, aw, ah, _ = a
        bx, by, bw, bh, _ = b
        ix1 = max(ax, bx)
        iy1 = max(ay, by)
        ix2 = min(ax+aw, bx+bw)
        iy2 = min(ay+ah, by+bh)
        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter = iw * ih
        area_a = aw * ah
        area_b = bw * bh
        union = area_a + area_b - inter
        if union <= 0:
            return 0.0
        return inter / union

detector = AnomalyDetector()


# ================================================================================
# Gradio 界面构建
# ================================================================================

def create_interface(default_dataset: str = None) -> gr.Blocks:
    """创建 Gradio 界面"""
    
    # 获取默认数据集
    if default_dataset is None:
        available = get_available_datasets()
        default_dataset = available[0] if available else "region1"
    
    # 读取外部 CSS 文件
    css_path = Path(__file__).parent / "styles.css"
    try:
        css = css_path.read_text(encoding='utf-8')
    except FileNotFoundError:
        # 如果 CSS 文件不存在，使用默认样式
        css = """
        .gradio-container { max-width: 1400px; margin: 0 auto; }
        .title { text-align: center; }
        .center { text-align: center; }
        """
    
    # ── FOUC 反闪烁：head 内阻塞式设置 data-theme + CSS 兜底 ──
    _anti_fouc = (
        '<script>!function(){'
        'var t=localStorage.getItem("theme");'
        'if(!t){t=window.matchMedia("(prefers-color-scheme:dark)").matches?"dark":"light"};'
        'document.documentElement.setAttribute("data-theme",t)'
        '}()</script>'
        '<style>html:not([data-theme]){visibility:hidden}</style>'
    )
    with gr.Blocks(css=css, title="工业异常检测系统", head=_anti_fouc) as demo:

        # ═══════════════════════════════════════════════════════════
        # 主题系统注入 — 通过 gr.HTML 绕过 Gradio 6 CSS 作用域
        # Gradio 6 会对 css= 参数做选择器作用域处理，导致 :root 在
        # @media 内失效。gr.HTML 中的 <style>/<script>/<link> 标签
        # 不会被作用域处理，可以正确设置 CSS 变量和图标。
        # ═══════════════════════════════════════════════════════════

        # V2: 全页加载遮罩 — 首次访问时自动淡出
        gr.HTML("""<div class="page-loader"><div class="page-loader-core"></div></div>""")

        # Favicon（SVG 菱形图标，theme.js 在主题切换时动态更新 href）
        gr.HTML(theme.get_favicon_html())

        # 亮色模式 CSS 变量
        # - html[data-theme="light"]：手动切换时启用（高优先级）
        # - @media (prefers-color-scheme: light)：JS 禁用时的降级兜底
        gr.HTML(f"<style>{theme.get_light_css()}</style>")

        # ==================== 标题区域 ====================
        # 使用 gr.HTML 而非 gr.Markdown，原因：
        # 1. 标题区域包含 <button>/<svg> 交互元素（主题切换按钮）
        # 2. gr.Markdown 默认 sanitize=True 会剥离这些元素
        # 3. 内容为纯 HTML，无 Markdown 格式需求
        gr.HTML(f"""
        <div class="reveal reveal-1" style="padding: 48px 0 8px 0;">
            <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                <div style="display:flex;align-items:center;gap:14px;">
                    <!-- V3: Logo — 深色菱形 + 蓝色核心，与 Favicon 同源 -->
                    <svg width="36" height="36" viewBox="0 0 36 36" fill="none" xmlns="http://www.w3.org/2000/svg" style="flex-shrink:0;">
                        <defs>
                            <linearGradient id="logoGrad" x1="18" y1="2" x2="18" y2="34" gradientUnits="userSpaceOnUse">
                                <stop offset="0%" stop-color="#2997ff"/>
                                <stop offset="100%" stop-color="#0070d6"/>
                            </linearGradient>
                        </defs>
                        <rect x="2" y="18" width="22.63" height="22.63" rx="3" transform="rotate(-45 2 18)" fill="url(#logoGrad)" opacity="0.15"/>
                        <rect x="6" y="18" width="16.97" height="16.97" rx="2" transform="rotate(-45 6 18)" fill="url(#logoGrad)" opacity="0.35"/>
                        <rect x="9.5" y="18" width="12.02" height="12.02" rx="2" transform="rotate(-45 9.5 18)" fill="url(#logoGrad)"/>
                        <circle cx="18" cy="18" r="2.5" fill="#ffffff" opacity="0.9"/>
                    </svg>
                    <div>
                        <div class="title">缺陷检测</div>
                        <div class="subtitle">无监督异常检测系统 · Anomalib 2.3</div>
                    </div>
                </div>
                {theme.get_theme_switch_html()}
            </div>
        </div>
        """)

        # 主题切换 JavaScript（放在标题区之后，确保按钮 DOM 已就绪）
        # 内部有重试机制，即使按钮延迟渲染也能正确初始化
        gr.HTML(theme.get_theme_js())

        # 数据集选择（用 Row 包裹消除 .block.padded 默认 padding 错位）
        with gr.Row():
            dataset_dropdown = gr.Dropdown(
                choices=get_available_datasets(),
                value=default_dataset,
                label="数据集"
            )
        
        # ==================== 算法选择 Tabs ====================
        with gr.Tabs(elem_classes=["reveal", "reveal-2"]) as tabs:
            with gr.Tab("PatchCore", elem_classes=["algorithm-tab"]) as tab_patchcore:
                gr.HTML("""
                <div class="algo-card">
                    <h4 class="recommended">PatchCore <span style="font-weight:400;font-size:13px;color:var(--text-tertiary);">— 特征建模</span></h4>
                    <p>CNN 提取局部特征 → 记忆库存储正常样本 → 最近邻距离判别。零训练、推理最快、工业首选。</p>
                </div>
                """)
            with gr.Tab("FRE", elem_classes=["algorithm-tab"]) as tab_fre:
                gr.HTML("""
                <div class="algo-card">
                    <h4>FRE <span style="font-weight:400;font-size:13px;color:var(--text-tertiary);">— 特征重构</span></h4>
                    <p>ResNet50 提取特征 → 自编码器重构 → 重构误差定位异常。高解释性，适合质量追溯。</p>
                </div>
                """)
            with gr.Tab("DRAEM", elem_classes=["algorithm-tab"]) as tab_draem:
                gr.HTML("""
                <div class="algo-card">
                    <h4>DRAEM <span style="font-weight:400;font-size:13px;color:var(--text-tertiary);">— 自监督判别</span></h4>
                    <p>数据增强合成缺陷 → 训练判别网络。无需真实异常样本，微小缺陷检测灵敏。</p>
                </div>
                """)
            with gr.Tab("PaDiM", elem_classes=["algorithm-tab"]) as tab_padim:
                gr.HTML("""
                <div class="algo-card">
                    <h4>PaDiM <span style="font-weight:400;font-size:13px;color:var(--text-tertiary);">— 概率建模</span></h4>
                    <p>Patch 级高斯分布 → 马氏距离度量偏离。与 PatchCore 互补，内存小、速度快。</p>
                </div>
                """)
        
        # 隐藏的选择器用于跟踪当前算法
        current_algo = gr.State(value="patchcore")
        
        # ==================== 主体区域 ====================
        with gr.Row(elem_classes=["reveal", "reveal-3"]):
            
            # -------- 左侧：控制面板 --------
            with gr.Column(scale=1, min_width=300):
                # 算法选择下拉菜单（用 Row 包裹以消除 Gradio .block.padded 默认 padding 导致的错位）
                with gr.Row():
                    algo_dropdown = gr.Dropdown(
                        choices=[('FRE', 'fre'), ('PatchCore', 'patchcore'), ('DRAEM', 'draem'), ('PaDiM', 'padim')],
                        value='patchcore',
                        label="算法选择"
                    )

                # 操作区域：图片左侧，按钮和状态右侧垂直排列
                with gr.Row():
                    # 左侧：图片上传
                    with gr.Column(scale=1):
                        image_input = gr.Image(
                            type="numpy",
                            label="上传测试图片",
                            image_mode="RGB",
                            height=200
                        )
                    
                    # 右侧：按钮和状态垂直堆砌（与图片等高）
                    with gr.Column(scale=1):
                        # 开始推理按钮 - 占据大部分高度
                        run_button = gr.Button(
                            "开始推理", 
                            variant="primary", 
                            size="lg", 
                            elem_classes=["inference-btn"],
                            scale=3
                        )
                        
                        # 模型加载状态 - 带动态效果
                        load_status = gr.HTML(
                            value='<div class="status-panel"><div class="loading-spinner" style="opacity:0.3;"></div><div style="font-family:var(--font-body);font-size:14px;color:var(--text-tertiary);margin-top:10px;">等待上传图片</div></div>',
                            scale=2
                        )
            
            # -------- 右侧：结果展示 --------
            with gr.Column(scale=2, min_width=500):
                gr.Markdown("## 检测结果", elem_classes=["panel-title"])
                
                # 图片展示区 + 热力图比例尺
                with gr.Row():
                    # 原图
                    with gr.Column(scale=2):
                        original_output = gr.Image(
                            type="numpy",
                            label="原图",
                            elem_classes=["image-display"],
                            height=260
                        )
                    
                    # 缺陷热力图 + 颜色比例尺
                    with gr.Column(scale=2):
                        heatmap_output = gr.Image(
                            type="numpy",
                            label="异常热力图",
                            elem_classes=["image-display"],
                            height=260
                        )
                    
                    # 垂直颜色比例尺
                    with gr.Column(scale=0):
                        gr.HTML("""
                        <div class="heatmap-legend reveal-child-4">
                            <div style="font-family:var(--font-body);font-size:11px;font-weight:500;color:var(--text-tertiary);margin-bottom:8px;letter-spacing:0.02em;">得分</div>
                            <div style="display: flex; flex-direction: row; height: 260px; align-items: stretch;">
                                <div class="heatmap-legend-bar"></div>
                                <div class="heatmap-legend-labels">
                                    <span>1.0</span>
                                    <span>0.8</span>
                                    <span>0.6</span>
                                    <span>0.4</span>
                                    <span>0.2</span>
                                    <span>0.0</span>
                                </div>
                            </div>
                        </div>
                        """)
                
                # 结果数据展示区
                result_output = gr.HTML(
                    """<div class="result-card" style="text-align: center; padding: 48px 28px;">
                        <div style="font-family:var(--font-body);font-size:15px;color:var(--text-tertiary);">等待推理…</div>
                    </div>"""
                )

        # 推理交互增强 JS（热力图 hover tooltip + bbox 高亮）
        gr.HTML(theme.get_inference_js())
        
        # ==================== 模型对比区域 ====================
        gr.Markdown("## 四模型对比", elem_classes=["panel-title", "reveal", "reveal-4"])
        with gr.Accordion("展开对比模式 — 一键运行四种算法，并排展示检测效果", open=False):
            compare_button = gr.Button("四种算法同时推理", variant="primary", size="lg", elem_classes=["compare-btn"])

            with gr.Row():
                with gr.Column():
                    gr.Markdown("**PatchCore**  <small style='color:var(--ok);font-weight:400;font-size:11px;'>特征建模·首选</small>", elem_classes=["image-label"])
                    compare_orig_pc = gr.Image(type="numpy", label="原图", height=200)
                    compare_heat_pc = gr.Image(type="numpy", label="热力图", height=200)
                    compare_result_pc = gr.HTML("""<div class="compare-result-card"><div style="font-size:13px;color:var(--text-tertiary);text-align:center;padding:12px;">等待推理…</div></div>""")
                with gr.Column():
                    gr.Markdown("**PaDiM**  <small style='color:var(--accent);font-weight:400;font-size:11px;'>概率建模·对照</small>", elem_classes=["image-label"])
                    compare_orig_padim = gr.Image(type="numpy", label="原图", height=200)
                    compare_heat_padim = gr.Image(type="numpy", label="热力图", height=200)
                    compare_result_padim = gr.HTML("""<div class="compare-result-card"><div style="font-size:13px;color:var(--text-tertiary);text-align:center;padding:12px;">等待推理…</div></div>""")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**FRE**  <small style='color:var(--warn);font-weight:400;font-size:11px;'>重构法·备选</small>", elem_classes=["image-label"])
                    compare_orig_fre = gr.Image(type="numpy", label="原图", height=200)
                    compare_heat_fre = gr.Image(type="numpy", label="热力图", height=200)
                    compare_result_fre = gr.HTML("""<div class="compare-result-card"><div style="font-size:13px;color:var(--text-tertiary);text-align:center;padding:12px;">等待推理…</div></div>""")
                with gr.Column():
                    gr.Markdown("**DRAEM**  <small style='color:var(--bad);font-weight:400;font-size:11px;'>自监督·备选</small>", elem_classes=["image-label"])
                    compare_orig_draem = gr.Image(type="numpy", label="原图", height=200)
                    compare_heat_draem = gr.Image(type="numpy", label="热力图", height=200)
                    compare_result_draem = gr.HTML("""<div class="compare-result-card"><div style="font-size:13px;color:var(--text-tertiary);text-align:center;padding:12px;">等待推理…</div></div>""")

        # ==================== 底部说明 ====================
        gr.Markdown("""
        <div class="footer-section reveal reveal-5">
            <div class="footer-title">使用说明</div>
            <div class="footer-content">
                <div class="footer-item">
                    <h5>单模型推理</h5>
                    <ul>
                        <li>选择算法（顶部标签页或下拉菜单）</li>
                        <li>上传待检测图片</li>
                        <li>点击「开始推理」查看结果</li>
                    </ul>
                </div>
                <div class="footer-item">
                    <h5>四模型对比</h5>
                    <p>展开「对比模式」，一键运行四种算法并排展示检测效果，直观对比各算法优劣。</p>
                </div>
                <div class="footer-item">
                    <h5>热力图解读</h5>
                    <p>颜色偏红 = 异常置信度高。右侧比例尺标注异常得分范围。</p>
                </div>
            </div>
        </div>
        """)
        
        # ==================== 事件绑定 ====================
        
        # ── 骨架屏模板（模型加载 / 推理进行中）──
        SKELETON_HTML = '''<div class="skeleton-card reveal-child-1">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:32px;">
                <div><div class="skeleton-row w-60 h-12"></div><div class="skeleton-row w-40 h-36" style="margin-top:8px;"></div></div>
                <div class="skeleton-row h-36" style="width:64px;border-radius:100px;"></div>
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:24px;">
                <div class="skeleton-row h-36" style="height:120px;"></div>
                <div class="skeleton-row h-36" style="height:120px;"></div>
            </div>
            <div class="skeleton-row w-80 h-36" style="height:56px;border-radius:var(--r-md);"></div>
        </div>'''

        def format_status(message, is_loading=False):
            """格式化状态消息 — Apple 极简状态指示 + 进度条"""
            if is_loading:
                return (
                    f'<div class="status-panel">'
                    f'<div class="loading-spinner"></div>'
                    f'<div style="font-family:var(--font-body);font-size:14px;color:var(--accent);margin-top:10px;">{message}</div>'
                    f'<div class="progress-bar"></div>'
                    f'</div>'
                )
            elif "[OK]" in message or "成功" in message or "完成" in message:
                return f'<div class="status-panel" style="background:var(--ok-bg);"><div style="font-size:14px;color:var(--ok);">{message.replace("[OK]", "").strip()}</div></div>'
            elif "[FAIL]" in message or "失败" in message or "错误" in message:
                return f'<div class="status-panel" style="background:var(--bad-bg);"><div style="font-size:14px;color:var(--bad);">{message.replace("[FAIL]", "").strip()}</div></div>'
            elif "[WARN]" in message or "警告" in message or "请先" in message:
                return f'<div class="status-panel" style="background:var(--warn-bg);"><div style="font-size:14px;color:var(--warn);">{message.replace("[WARN]", "").strip()}</div></div>'
            else:
                return f'<div class="status-panel"><div class="loading-spinner" style="opacity:0.3;"></div><div style="font-family:var(--font-body);font-size:14px;color:var(--text-tertiary);margin-top:10px;">{message}</div></div>'
        
        def on_model_change(model_key, dataset):
            """算法切换事件"""
            config = MODEL_CONFIGS[model_key]

            yield format_status(f"正在加载 {config.name}...", is_loading=True), SKELETON_HTML
            success, message = detector.load_model(model_key, dataset)
            # 完成后恢复为占位卡片
            placeholder = '<div class="result-card" style="text-align: center; padding: 48px 28px;"><div style="font-size:15px;color:var(--text-tertiary);">等待推理…</div></div>'
            yield format_status(message), placeholder
        
        def on_run_click(model_key, dataset, image):
            """推理按钮点击事件"""
            if image is None:
                yield None, None, "<div class='result-card' style='text-align:center;padding:48px 28px;'><div style='font-size:15px;color:var(--text-tertiary);'>请先上传图片</div></div>", format_status("请先上传测试图片")
                return

            # 先显示骨架屏（保持图片不变）
            yield gr.skip(), gr.skip(), SKELETON_HTML, format_status("正在加载模型...", is_loading=True)

            # 确保模型已加载
            success, message = detector.load_model(model_key, dataset)
            if not success:
                yield image, image, f"<div class='result-card' style='text-align:center;padding:48px 28px;'><div style='font-size:14px;color:var(--bad);'>{message}</div></div>", format_status(message)
                return

            # 执行推理
            original, heatmap, result = detector.predict(image)
            yield original, heatmap, result, format_status("推理完成")
        
        def on_image_upload(image):
            """图片上传事件"""
            if image is not None:
                return format_status("图片已就绪，点击推理")
            return format_status("等待上传图片...")

        # ── 对比模式状态 HTML 片段 ──
        COMPARE_PENDING = '''<div class="compare-result-card compare-pending">
            <div class="compare-loading"><div class="loading-spinner muted"></div><span>等待处理…</span></div>
        </div>'''
        COMPARE_ACTIVE = '''<div class="compare-result-card compare-active">
            <div class="compare-loading"><div class="loading-spinner"></div><span>正在推理…</span></div>
        </div>'''
        COMPARE_NO_IMAGE = '<div class="compare-result-card"><div style="font-size:13px;color:var(--bad);text-align:center;padding:12px;">请先上传图片</div></div>'

        def on_compare_click(dataset, image):
            """四模型对比 — 流式渐进渲染：每完成一个模型立即更新结果"""
            if image is None:
                yield tuple([image, image, COMPARE_NO_IMAGE] * 4)
                return

            models = ['patchcore', 'padim', 'fre', 'draem']
            model_names = ['PatchCore', 'PaDiM', 'FRE', 'DRAEM']
            # 初始化所有槽位为「等待处理」
            results = [image, image, COMPARE_PENDING] * 4

            for i, m in enumerate(models):
                base = i * 3
                # 将当前槽位切换为「正在推理」
                results[base] = image
                results[base + 1] = image
                results[base + 2] = COMPARE_ACTIVE
                yield tuple(results)

                success, msg = detector.load_model(m, dataset)
                if success:
                    orig, heat, result_html = detector.predict(image)
                    results[base] = orig
                    results[base + 1] = heat
                    results[base + 2] = result_html
                else:
                    results[base] = image
                    results[base + 1] = image
                    results[base + 2] = (
                        f'<div class="compare-result-card">'
                        f'<div style="font-size:13px;color:var(--bad);text-align:center;padding:12px;">'
                        f'{model_names[i]}: {msg}</div></div>'
                    )

                yield tuple(results)

        # 绑定Tab选择事件 - 更新current_algo
        def on_tab_select(tab_name):
            """Tab选择事件"""
            algo_map = {"PatchCore": "patchcore", "FRE": "fre", "DRAEM": "draem", "PaDiM": "padim"}
            return algo_map.get(tab_name, "patchcore")
        
        # tab_patchcore.select removed
        # tab_fre.select removed
        # tab_draem.select removed
        
        # 绑定推理按钮事件 - 使用current_algo
        run_button.click(
            fn=on_run_click,
            inputs=[algo_dropdown, dataset_dropdown, image_input],
            outputs=[original_output, heatmap_output, result_output, load_status]
        )
        
        image_input.change(
            fn=on_image_upload,
            inputs=image_input,
            outputs=load_status
        )
        
        # 绑定算法选择事件
        algo_dropdown.change(
            fn=on_model_change,
            inputs=[algo_dropdown, dataset_dropdown],
            outputs=[load_status, result_output]  # 更新状态 + 骨架屏
        )

        # 绑定四模型对比事件
        compare_button.click(
            fn=on_compare_click,
            inputs=[dataset_dropdown, image_input],
            outputs=[compare_orig_pc, compare_heat_pc, compare_result_pc,
                     compare_orig_padim, compare_heat_padim, compare_result_padim,
                     compare_orig_fre, compare_heat_fre, compare_result_fre,
                     compare_orig_draem, compare_heat_draem, compare_result_draem]
        )

    return demo


def main():
    """主函数"""
    print("="*70)
    print("[UI] UI Demo Module (Anomalib 2.x)")
    print("="*70)
    print("\nStarting Gradio service...")
    print("Access: http://127.0.0.1:7860")
    print("="*70)
    
    # 不预加载模型，按需加载以加快启动速度
    print("\n[OK] 模型将在首次使用时加载")
    
    # 创建并启动界面
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        inbrowser=True,  # 自动打开浏览器
    )


if __name__ == '__main__':
    main()
