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
    python run_ui.py
================================================================================
"""

import os
import warnings
from pathlib import Path
from typing import Tuple, Optional, List
from dataclasses import dataclass

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
)

# 忽略警告
warnings.filterwarnings('ignore')


# ================================================================================
# 配置常量
# ================================================================================

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
- 重构法改进版，效果远超Ganomaly
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
    )
}

# 异常判定阈值
ANOMALY_THRESHOLD = 0.5


# ================================================================================
# 异常检测器类
# ================================================================================

class AnomalyDetector:
    """
    异常检测器
    
    管理模型加载和推理，支持3种算法切换
    """
    
    def __init__(self):
        self.current_model: Optional[str] = None
        self.model = None
        self.engine = None
    
    def load_model(self, model_key: str) -> Tuple[bool, str]:
        """
        加载指定模型
        
        Args:
            model_key: 模型标识 (autoencoder/patchcore/draem)
        
        Returns:
            Tuple[bool, str]: (是否成功, 状态信息)
        """
        # 如果模型已加载，直接返回
        if model_key == self.current_model and self.model is not None:
            return True, f"[OK] 模型已加载: {MODEL_CONFIGS[model_key].name}"
        
        config = MODEL_CONFIGS.get(model_key)
        if config is None:
            return False, f"[FAIL] 未知模型: {model_key}"
        
        # 查找权重文件
        weight_path = Path(config.weight_path)
        if not weight_path.exists():
            # 尝试查找其他可能的权重文件（递归搜索 anomalib 2.x 的嵌套结构）
            search_base = Path('./results')
            model_dir = search_base / model_key
            
            if model_dir.exists():
                # 优先查找 lightning/model.ckpt（anomalib 2.x 标准路径）
                for pattern in [
                    f'{model_key}/**/lightning/model.ckpt',
                    f'{model_key}/**/*.ckpt',
                ]:
                    candidates = list(search_base.glob(pattern))
                    if candidates:
                        # 取最新修改的文件
                        weight_path = max(candidates, key=lambda p: p.stat().st_mtime)
                        break
        
        if not weight_path.exists():
            return False, (
                f"[FAIL] 未找到模型权重: {config.weight_path}\n\n"
                f"请先训练模型:\n"
                f"```bash\n"
                f"python modules/algorithm/trainer.py --model {model_key} --category <your_category> --data_path <data_path>\n"
                f"```"
            )
        
        try:
            # 创建模型实例（使用配置的自定义参数）
            model_kwargs = getattr(config, 'model_kwargs', {}) or {}
            self.model = config.model_class(**model_kwargs)
            
            # 创建 Engine
            self.engine = Engine()
            
            # 加载权重
            checkpoint = torch.load(weight_path, map_location='cpu')
            if 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                self.model.load_state_dict(checkpoint, strict=False)
            
            self.model.eval()
            self.current_model = model_key
            
            return True, f"[OK] 成功加载模型: {config.name}"
        
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
        
        try:
            # 确保图片格式正确
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            
            # 保存临时文件用于 PredictDataset（传入文件路径而不是目录）
            temp_dir = Path('./temp_predict')
            temp_dir.mkdir(exist_ok=True)
            temp_path = temp_dir / 'temp_image.png'
            cv2.imwrite(str(temp_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            
            # 创建预测数据集 - 传入文件路径
            dataset = PredictDataset(
                path=temp_path,  # 传入文件路径而不是目录
                image_size=(256, 256),
            )
            
            # 执行推理
            predictions = self.engine.predict(
                model=self.model,
                dataset=dataset,
            )
            
            # 清理临时文件
            temp_path.unlink(missing_ok=True)
            
            # 获取预测结果
            if predictions is not None:
                # 转换为列表（如果是生成器）
                if not isinstance(predictions, list):
                    predictions = list(predictions)
                
                if len(predictions) > 0:
                    prediction = predictions[0]
                    
                    # 提取结果
                    anomaly_map = prediction.anomaly_map
                    pred_score = float(prediction.pred_score)
                    pred_label = int(prediction.pred_label)
                    
                    # 生成热力图
                    heatmap = self._generate_heatmap(image, anomaly_map)
                    
                    # 生成结果文本
                    result_text = self._format_result(pred_score, pred_label)
                    
                    return image, heatmap, result_text
            
            return image, image, "[FAIL] 推理失败: 未获取到预测结果"
        
        except Exception as e:
            import traceback
            traceback.print_exc()
            return image, image, f"[FAIL] 推理失败: {str(e)}"
    
    def _generate_heatmap(
        self,
        original: np.ndarray,
        anomaly_map: torch.Tensor
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
        
        return overlay
    
    def _format_result(self, score: float, label: int) -> str:
        """格式化结果文本 - 工业仪表盘风格"""
        model_config = MODEL_CONFIGS[self.current_model]
        is_anomaly = label == 1
        confidence = score if is_anomaly else 1 - score
        
        # 莫兰迪色系
        status_normal = "var(--status-normal-text)"
        status_anomaly = "var(--status-anomaly-text)"
        status_color = status_anomaly if is_anomaly else status_normal
        
        return f"""
<div class="result-card fade-in">
    <!-- 状态标题 -->
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px;">
        <div>
            <div style="font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px;">检测算法</div>
            <div style="font-size: 18px; font-weight: 600; color: var(--accent-primary);">{model_config.name}</div>
        </div>
        <div class="status-badge {'anomaly' if is_anomaly else 'normal'}">
            <span>{'⚠' if is_anomaly else '✓'}</span>
            <span>{'异常' if is_anomaly else '正常'}</span>
        </div>
    </div>
    
    <!-- 核心数据展示 -->
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 24px;">
        <!-- 异常得分 -->
        <div class="core-metric">
            <div class="label">异常得分</div>
            <div class="value {'anomaly' if is_anomaly else 'normal'}">{score:.4f}</div>
            <div class="progress-container">
                <div class="progress-bar-mini">
                    <div class="progress-fill {'anomaly' if is_anomaly else 'normal'}" style="width: {score*100}%;"></div>
                </div>
                <div class="threshold-line">
                    <div class="threshold-marker" style="left: {ANOMALY_THRESHOLD*100}%;"></div>
                    <div class="threshold-label" style="left: {ANOMALY_THRESHOLD*100}%;">{ANOMALY_THRESHOLD}</div>
                </div>
            </div>
        </div>
        
        <!-- 置信度 -->
        <div class="core-metric">
            <div class="label">置信度</div>
            <div class="value" style="color: var(--accent-primary);">{confidence:.1%}</div>
            <div class="progress-container">
                <div class="progress-bar-mini">
                    <div class="progress-fill" style="width: {confidence*100}%; background: linear-gradient(90deg, var(--accent-subtle), var(--accent-primary));"></div>
                </div>
            </div>
        </div>
    </div>
    
    <!-- 结果解读 -->
    <div style="background: var(--bg-tertiary); border-radius: var(--radius-sm); padding: 16px; border-left: 2px solid {status_color};">
        <div style="font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 12px;">结果解读</div>
        <div style="font-size: 13px; color: var(--text-secondary); line-height: 1.8;">
            <div>得分 <b style="color: {status_color};">{score:.4f}</b> {'>' if is_anomaly else '<'} 阈值 <b>{ANOMALY_THRESHOLD}</b>，判定为 <b style="color: {status_color};">{'异常' if is_anomaly else '正常'}</b></div>
            <div style="margin-top: 8px; color: var(--text-muted);">热力图中偏红区域表示异常概率较高</div>
        </div>
    </div>
</div>
"""


# 全局检测器实例
detector = AnomalyDetector()


# ================================================================================
# Gradio 界面构建
# ================================================================================

def create_interface() -> gr.Blocks:
    """创建 Gradio 界面"""
    
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
    
    with gr.Blocks(css=css, title="工业异常检测系统") as demo:
        
        # ==================== 标题区域 ====================
        gr.Markdown("""
        <div class="title">工业图像异常检测系统</div>
        <div class="subtitle">基于 anomalib 2.x 的无监督异常检测 | 实时推理 | 精准定位</div>
        """)
        
        # ==================== 算法选择 Tabs ====================
        with gr.Tabs() as tabs:
            with gr.Tab("PatchCore", elem_classes=["algorithm-tab"]) as tab_patchcore:
                gr.HTML("""
                <div class="algo-card">
                    <h4 class="recommended">PatchCore — 特征建模法</h4>
                    <p>使用预训练CNN提取局部特征，构建记忆库存储正常样本特征。测试时通过计算测试样本特征与记忆库的最近邻距离来判断异常。无需训练，推理速度最快，工业界最佳方案。</p>
                </div>
                """)
            with gr.Tab("FRE", elem_classes=["algorithm-tab"]) as tab_fre:
                gr.HTML("""
                <div class="algo-card">
                    <h4>FRE — 特征重构法</h4>
                    <p>使用预训练ResNet提取特征，通过线性自编码器重构特征。重构误差即异常分数。效果优秀，适合需要解释性的场景。</p>
                </div>
                """)
            with gr.Tab("DRAEM", elem_classes=["algorithm-tab"]) as tab_draem:
                gr.HTML("""
                <div class="algo-card">
                    <h4>DRAEM — 自监督学习</h4>
                    <p>通过数据增强合成异常样本，训练判别网络区分正常和异常区域。无需真实异常样本，对小缺陷检测效果好。</p>
                </div>
                """)
        
        # 隐藏的选择器用于跟踪当前算法
        current_algo = gr.State(value="patchcore")
        
        # ==================== 主体区域 ====================
        with gr.Row():
            
            # -------- 左侧：控制面板 --------
            with gr.Column(scale=1, min_width=300):
                # 使用隐藏的Dropdown保持功能
                model_dropdown = gr.Dropdown(
                    choices=[('PatchCore', 'patchcore'), ('FRE', 'fre'), ('DRAEM', 'draem')],
                    value='patchcore',
                    label="",
                    visible=False
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
                            value='''<div class="status-panel">
                                <div class="loading-spinner"></div>
                                <div style="color: #666; font-size: 12px; margin-top: 8px;">等待上传图片</div>
                            </div>''',
                            scale=2
                        )
            
            # -------- 右侧：结果展示 --------
            with gr.Column(scale=2, min_width=500):
                gr.Markdown("### 检测结果", elem_classes=["panel-title"])
                
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
                        <div class="heatmap-legend">
                            <div style="font-size: 10px; color: #666; text-align: center; margin-bottom: 8px;">异常度</div>
                            <div style="display: flex; flex-direction: row; height: 260px; align-items: stretch;">
                                <div class="heatmap-legend-bar"></div>
                                <div class="heatmap-legend-labels">
                                    <span>1.0</span>
                                    <span>0.75</span>
                                    <span>0.50</span>
                                    <span>0.25</span>
                                    <span>0.0</span>
                                </div>
                            </div>
                        </div>
                        """)
                
                # 结果数据展示区
                result_output = gr.HTML(
                    """<div class="result-card" style="text-align: center; color: #666;">
                        <div style="padding: 40px;">等待推理...</div>
                    </div>"""
                )
        
        # ==================== 底部说明 ====================
        gr.Markdown("""
        <div class="footer-section">
            <div class="footer-title">使用说明</div>
            <div class="footer-content">
                <div class="footer-item">
                    <h5>操作流程</h5>
                    <ul>
                        <li>选择算法（顶部Tabs）</li>
                        <li>上传待检测图片</li>
                        <li>点击推理按钮</li>
                        <li>查看检测结果</li>
                    </ul>
                </div>
                <div class="footer-item">
                    <h5>热力图解读</h5>
                    <p>颜色越偏红表示异常概率越高。右侧比例尺显示异常度范围（0-1）。</p>
                </div>
                <div class="footer-item">
                    <h5>注意事项</h5>
                    <p>首次切换算法需加载模型权重。请确保模型已训练完成。</p>
                </div>
            </div>
        </div>
        """)
        
        # ==================== 事件绑定 ====================
        
        def format_status(message, is_loading=False):
            """格式化状态消息为HTML - 工业暗色风格"""
            if is_loading:
                return f'''<div class="status-panel">
                    <div class="loading-spinner"></div>
                    <div style="color: var(--accent-primary); font-size: 12px; margin-top: 8px;">{message}</div>
                </div>'''
            elif "[OK]" in message or "成功" in message or "完成" in message:
                return f'''<div class="status-panel" style="background: rgba(45, 106, 79, 0.15); border-color: var(--status-normal);">
                    <div style="color: var(--status-normal-text); font-size: 12px;">{message.replace("[OK]", "").strip()}</div>
                </div>'''
            elif "[FAIL]" in message or "失败" in message or "错误" in message:
                return f'''<div class="status-panel" style="background: rgba(155, 34, 38, 0.15); border-color: var(--status-anomaly);">
                    <div style="color: var(--status-anomaly-text); font-size: 12px;">{message.replace("[FAIL]", "").strip()}</div>
                </div>'''
            elif "[WARN]" in message or "警告" in message or "请先" in message:
                return f'''<div class="status-panel" style="background: rgba(184, 134, 11, 0.15); border-color: var(--status-warning);">
                    <div style="color: var(--status-warning-text); font-size: 12px;">{message.replace("[WARN]", "").strip()}</div>
                </div>'''
            else:
                return f'''<div class="status-panel">
                    <div class="loading-spinner" style="opacity: 0.3;"></div>
                    <div style="color: #666; font-size: 12px; margin-top: 8px;">{message}</div>
                </div>'''
        
        def on_model_change(model_key):
            """算法切换事件 - 使用生成器实现渐进式更新"""
            import time
            config = MODEL_CONFIGS[model_key]
            
            # 更新顶部算法介绍卡片
            algo_descriptions = {
                'patchcore': '<h4 class="recommended">PatchCore — 特征建模法</h4><p>使用预训练CNN提取局部特征，构建记忆库存储正常样本特征。测试时通过计算测试样本特征与记忆库的最近邻距离来判断异常。无需训练，推理速度最快，工业界最佳方案。</p>',
                'fre': '<h4>FRE — 特征重构法</h4><p>使用预训练ResNet提取特征，通过线性自编码器重构特征。重构误差即异常分数。效果优秀，适合需要解释性的场景。</p>',
                'draem': '<h4>DRAEM — 自监督学习</h4><p>通过数据增强合成异常样本，训练判别网络区分正常和异常区域。无需真实异常样本，对小缺陷检测效果好。</p>'
            }
            
            # 第一步：立即返回加载中状态
            yield algo_descriptions.get(model_key, ''), format_status(f"正在加载 {config.name}...", is_loading=True)
            
            # 第二步：执行实际加载
            success, message = detector.load_model(model_key)
            
            # 第三步：返回最终结果
            yield algo_descriptions.get(model_key, ''), format_status(message)
        
        def on_run_click(model_key, image):
            """推理按钮点击事件"""
            if image is None:
                return None, None, "<div class='result-card' style='text-align: center; color: #666;'><div style='padding: 40px;'>请先上传图片</div></div>", format_status("请先上传测试图片")
            
            # 确保模型已加载
            success, message = detector.load_model(model_key)
            if not success:
                return image, image, f"<div class='result-card' style='color: var(--status-anomaly-text); padding: 20px;'>{message}</div>", format_status(message)
            
            # 执行推理
            original, heatmap, result = detector.predict(image)
            return original, heatmap, result, format_status("推理完成")
        
        def on_image_upload(image):
            """图片上传事件"""
            if image is not None:
                return format_status("图片已就绪，点击推理")
            return format_status("等待上传图片...")
        
        # 绑定Tab选择事件 - 更新current_algo
        def on_tab_select(tab_name):
            """Tab选择事件"""
            algo_map = {"PatchCore": "patchcore", "FRE": "fre", "DRAEM": "draem"}
            return algo_map.get(tab_name, "patchcore")
        
        tab_patchcore.select(
            fn=lambda: "patchcore",
            outputs=current_algo
        )
        tab_fre.select(
            fn=lambda: "fre",
            outputs=current_algo
        )
        tab_draem.select(
            fn=lambda: "draem",
            outputs=current_algo
        )
        
        # 绑定推理按钮事件 - 使用current_algo
        run_button.click(
            fn=on_run_click,
            inputs=[current_algo, image_input],
            outputs=[original_output, heatmap_output, result_output, load_status]
        )
        
        image_input.change(
            fn=on_image_upload,
            inputs=image_input,
            outputs=load_status
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
