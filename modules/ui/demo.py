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
    EfficientAd,
    Ganomaly,
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


# 4 种算法的配置
MODEL_CONFIGS = {
    'efficientad': ModelConfig(
        name='EfficientAd',
        direction='基于轻量级特征对齐',
        description='''
**算法原理**: 使用预训练教师网络和学生网络，通过特征对齐误差检测异常。
结合了知识蒸馏和特征对齐的思想。

**特点**:
- 速度快，精度高，适合工业部署
- 模型轻量级，推理效率高
- 训练稳定，收敛快
''',
        weight_path='./results/efficientad/weights/model.ckpt',
        model_class=EfficientAd,
        model_kwargs={},
    ),
    'ganomaly': ModelConfig(
        name='Ganomaly',
        direction='基于重构 (GAN)',
        description='''
**算法原理**: 训练GAN学习正常样本分布，异常样本无法良好重构。
通过重构误差检测异常。

**特点**:
- 经典重构方法升级版
- 基于GAN实现
- 适合理解重构思想
''',
        weight_path='./results/ganomaly/weights/model.ckpt',
        model_class=Ganomaly,
        model_kwargs={
            'n_features': 32,
            'latent_vec_size': 64,
            'extra_layers': 0,
            'batch_size': 16,
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
        weight_path='./results/patchcore/weights/model.ckpt',
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
        weight_path='./results/draem/weights/model.ckpt',
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
            
            # 保存临时文件用于 PredictDataset
            temp_dir = Path('./temp_predict')
            temp_dir.mkdir(exist_ok=True)
            temp_path = temp_dir / 'temp_image.png'
            cv2.imwrite(str(temp_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            
            # 创建预测数据集
            dataset = PredictDataset(
                path=temp_dir,
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
        """格式化结果文本"""
        model_config = MODEL_CONFIGS[self.current_model]
        status = "🔴 异常 (Anomaly)" if label == 1 else "🟢 正常 (Normal)"
        confidence = score if label == 1 else 1 - score
        
        return f"""
### 检测结果

| 项目 | 值 |
|:---|:---|
| **算法** | {model_config.name} |
| **方向** | {model_config.direction} |
| **判定结果** | {status} |
| **异常得分** | {score:.4f} |
| **置信度** | {confidence:.2%} |

---
**说明**: 
- 异常得分 > {ANOMALY_THRESHOLD} 判定为异常
- 热力图中**红色区域**表示高异常概率
"""


# 全局检测器实例
detector = AnomalyDetector()


# ================================================================================
# Gradio 界面构建
# ================================================================================

def create_interface() -> gr.Blocks:
    """创建 Gradio 界面"""
    
    css = """
    .model-info { font-size: 13px; line-height: 1.6; }
    .result-text { font-size: 14px; }
    .image-display { max-height: 400px; }
    """
    
    with gr.Blocks(css=css, title="工业异常检测演示") as demo:
        
        # ==================== 标题区域 ====================
        gr.Markdown("""
        # 🔍 工业图像异常检测演示系统
        
        基于 **anomalib 2.x** 的四种主流算法实现
        
        | 算法 | 方向 | 特点 |
        |:---|:---:|:---|
        | **EfficientAd** | 轻量级特征对齐 | 速度快，适合部署 |
        | **Ganomaly** | 基于重构 (GAN) | 经典重构方法 |
        | **PatchCore** | 特征建模 | 工业界最佳，无需训练 |
        | **DRAEM** | 自监督 | 无需异常样本，定位精准 |
        """)
        
        # ==================== 主体区域 ====================
        with gr.Row():
            
            # -------- 左侧：控制面板 --------
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ 控制面板")
                
                # 算法选择下拉菜单
                model_dropdown = gr.Dropdown(
                    choices=list(MODEL_CONFIGS.keys()),
                    value='patchcore',
                    label="选择算法",
                    info="切换不同的异常检测算法"
                )
                
                # 算法说明
                model_info = gr.Markdown(
                    MODEL_CONFIGS['patchcore'].description,
                    elem_classes=["model-info"]
                )
                
                # 图片上传
                image_input = gr.Image(
                    type="numpy",
                    label="📤 上传测试图片",
                    image_mode="RGB"
                )
                
                # 开始推理按钮
                run_button = gr.Button("🚀 开始推理", variant="primary", size="lg")
                
                # 加载状态
                load_status = gr.Textbox(
                    label="模型状态",
                    value="请选择算法并上传图片",
                    interactive=False
                )
            
            # -------- 右侧：结果展示 --------
            with gr.Column(scale=2):
                gr.Markdown("### [STAT] 检测结果")
                
                with gr.Row():
                    # 原图
                    original_output = gr.Image(
                        type="numpy",
                        label="原图",
                        elem_classes=["image-display"]
                    )
                    
                    # 缺陷热力图
                    heatmap_output = gr.Image(
                        type="numpy",
                        label="缺陷热力图",
                        elem_classes=["image-display"]
                    )
                
                # 结果文本
                result_output = gr.Markdown(
                    "等待推理...",
                    elem_classes=["result-text"]
                )
        
        # ==================== 底部说明 ====================
        gr.Markdown("""
        ---
        ### 📖 使用说明
        
        **操作步骤**:
        1. **选择算法**: 从下拉菜单选择要使用的算法
        2. **上传图片**: 点击上传区域选择测试图片（支持 PNG, JPG, BMP）
        3. **开始推理**: 点击"🚀 开始推理"按钮执行检测
        4. **查看结果**: 右侧显示原图和缺陷热力图，下方显示详细结果
        
        **热力图解读**:
        - 🔵 **蓝色/绿色**: 正常区域
        - 🟡 **黄色**: 疑似异常
        - 🔴 **红色**: 高概率异常区域
        
        **注意事项**:
        - 首次切换算法时会自动加载模型权重
        - 如果提示权重不存在，请先运行训练脚本
        - 推荐使用 GPU 进行推理以获得更快速度
        """)
        
        # ==================== 事件绑定 ====================
        
        def on_model_change(model_key):
            """算法切换事件"""
            config = MODEL_CONFIGS[model_key]
            success, message = detector.load_model(model_key)
            return config.description, message
        
        def on_run_click(model_key, image):
            """推理按钮点击事件"""
            if image is None:
                return None, None, "[WARN] 请先上传图片"
            
            # 确保模型已加载
            success, message = detector.load_model(model_key)
            if not success:
                return image, image, message
            
            # 执行推理
            original, heatmap, result = detector.predict(image)
            return original, heatmap, result
        
        # 绑定事件
        model_dropdown.change(
            fn=on_model_change,
            inputs=model_dropdown,
            outputs=[model_info, load_status]
        )
        
        run_button.click(
            fn=on_run_click,
            inputs=[model_dropdown, image_input],
            outputs=[original_output, heatmap_output, result_output]
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
    
    # 预加载默认模型
    print("\n[WAIT] Preloading default model (PatchCore)...")
    success, message = detector.load_model('patchcore')
    print(f"   {message}")
    
    # 创建并启动界面
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )


if __name__ == '__main__':
    main()
