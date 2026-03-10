"""
Gradio UI 演示界面
用于展示三种异常检测算法的推理结果

功能:
1. 下拉菜单选择算法模型（PatchCore / AutoEncoder / DRAEM）
2. 上传测试图片
3. 展示原始图片、缺陷热力图、异常得分/判定结果

使用方法:
    python app.py
    
    然后访问输出的本地 URL（通常是 http://127.0.0.1:7860）
"""

import os
import warnings
from pathlib import Path
from typing import Tuple, Optional
import tempfile

import numpy as np
import torch
import gradio as gr
from PIL import Image
import cv2

# 导入 anomalib 推理器
try:
    from anomalib.deploy import TorchInferencer, OpenVINOInferencer
    from anomalib.pre_processing.transforms import Denormalize
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("请确保已安装 anomalib: pip install anomalib")
    raise

# 忽略警告
warnings.filterwarnings('ignore')

# ============================================
# 配置参数 - 3 个最易复现的算法
# ============================================

# 模型权重路径配置（请根据实际情况修改）
MODEL_WEIGHTS = {
    'patchcore': {
        'path': './results/patchcore/checkpoints/model.ckpt',
        'config': './configs/patchcore.yaml',
        'description': '''
**方向**: 特征建模

**原理**: 使用预训练 CNN 提取局部特征，构建记忆库存储正常样本特征。测试时通过计算测试样本特征与记忆库中最近邻特征的距离来判断异常。

**特点**:
- ⭐ 极易复现，无需训练 epoch
- ⚡⚡⚡ 推理速度最快
- 🎯 适合实时检测场景
'''
    },
    'autoencoder': {
        'path': './results/autoencoder/checkpoints/model.ckpt',
        'config': './configs/autoencoder.yaml',
        'description': '''
**方向**: 重构

**原理**: 训练编码器-解码器网络，学习正常样本的压缩表示。异常样本无法被良好重构，通过比较输入和输出的重构误差检测异常。

**特点**:
- ⭐⭐ 容易复现，经典方法
- ⚡⚡ 推理速度中等
- 🔧 模型结构简单直观
'''
    },
    'draem': {
        'path': './results/draem/checkpoints/model.ckpt',
        'config': './configs/draem.yaml',
        'description': '''
**方向**: 自监督学习

**原理**: 在训练阶段通过数据增强生成合成异常样本，训练一个判别网络来学习区分正常区域和异常区域。无需真实异常样本即可训练。

**特点**:
- ⭐⭐⭐ 中等难度，但无需真实异常样本
- ⚡ 推理速度较慢
- 🎨 对小缺陷检测效果较好
'''
    }
}

# 异常判定阈值
ANOMALY_THRESHOLD = 0.5

# ============================================
# 模型管理类
# ============================================

class AnomalyDetector:
    """异常检测器管理类"""
    
    def __init__(self):
        self.current_model = None
        self.current_model_name = None
        self.inferencer = None
    
    def load_model(self, model_name: str) -> bool:
        """
        加载指定模型
        
        Args:
            model_name: 模型名称
        
        Returns:
            bool: 是否加载成功
        """
        if model_name == self.current_model_name and self.inferencer is not None:
            return True  # 模型已加载
        
        model_info = MODEL_WEIGHTS.get(model_name)
        if not model_info:
            print(f"❌ 未知模型: {model_name}")
            return False
        
        weight_path = Path(model_info['path'])
        config_path = Path(model_info['config'])
        
        # 检查文件是否存在，尝试多种可能的路径
        if not weight_path.exists():
            alternatives = [
                weight_path.parent / 'model.ckpt',
                weight_path.parent / 'model-v1.ckpt',
                weight_path.parent.parent / 'weights' / 'model.ckpt',
                weight_path.parent.parent / 'weights' / 'lightning_model.ckpt',
                weight_path.parent / 'last.ckpt',
                weight_path.parent / 'epoch=00.ckpt',
            ]
            for alt in alternatives:
                if alt.exists():
                    weight_path = alt
                    break
            else:
                print(f"⚠️  模型权重不存在: {weight_path}")
                print(f"   请先运行训练: python train_evaluate.py --model {model_name}")
                return False
        
        try:
            print(f"🔄 正在加载模型: {model_name}...")
            
            # 使用 TorchInferencer 进行推理
            self.inferencer = TorchInferencer(
                config=config_path,
                model_source=weight_path,
                device='auto'  # 自动选择 GPU/CPU
            )
            
            self.current_model_name = model_name
            self.current_model = model_name
            print(f"✅ 模型加载成功: {model_name}")
            return True
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def predict(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        对单张图片进行异常检测
        
        Args:
            image: 输入图片 (H, W, C) 格式
        
        Returns:
            Tuple: (原始图片, 热力图, 结果文本)
        """
        if self.inferencer is None:
            return image, np.zeros_like(image), "❌ 模型未加载"
        
        try:
            # 确保图片格式正确
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            
            # 进行推理
            predictions = self.inferencer.predict(image=image)
            
            # 提取结果
            anomaly_map = predictions.anomaly_map  # 异常热力图
            pred_score = predictions.pred_score    # 异常得分
            pred_label = predictions.pred_label    # 预测标签
            
            # 生成热力图可视化
            heatmap = self._generate_heatmap(image, anomaly_map)
            
            # 生成结果文本
            result_text = self._format_result(pred_score, pred_label, anomaly_map)
            
            return image, heatmap, result_text
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return image, np.zeros_like(image), f"❌ 推理失败: {str(e)}"
    
    def _generate_heatmap(self, original_image: np.ndarray, anomaly_map: np.ndarray) -> np.ndarray:
        """
        生成异常热力图可视化
        
        Args:
            original_image: 原始图片
            anomaly_map: 异常分数图
        
        Returns:
            np.ndarray: 叠加热力图后的图片
        """
        # 归一化异常图到 0-255
        anomaly_map_norm = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
        anomaly_map_uint8 = (anomaly_map_norm * 255).astype(np.uint8)
        
        # 调整异常图尺寸匹配原图
        h, w = original_image.shape[:2]
        anomaly_map_resized = cv2.resize(anomaly_map_uint8, (w, h))
        
        # 应用颜色映射 (JET: 蓝->绿->黄->红)
        heatmap_colored = cv2.applyColorMap(anomaly_map_resized, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # 叠加到原图
        alpha = 0.5  # 热力图透明度
        overlay = cv2.addWeighted(original_image, 1 - alpha, heatmap_colored, alpha, 0)
        
        return overlay
    
    def _format_result(self, score: float, label: int, anomaly_map: np.ndarray) -> str:
        """
        格式化预测结果文本
        
        Args:
            score: 异常得分
            label: 预测标签 (0=正常, 1=异常)
            anomaly_map: 异常图
        
        Returns:
            str: 格式化后的结果文本
        """
        # 判定结果
        status = "🔴 异常 (Anomaly)" if label == 1 else "🟢 正常 (Normal)"
        
        # 置信度
        confidence = score if label == 1 else 1 - score
        
        # 计算异常区域占比
        anomaly_ratio = np.mean(anomaly_map > ANOMALY_THRESHOLD) * 100
        
        result = f"""
### 检测结果

| 项目 | 值 |
|:---|:---|
| **模型** | {self.current_model_name.upper()} |
| **方向** | {MODEL_WEIGHTS[self.current_model_name].get('方向', 'N/A')} |
| **判定结果** | {status} |
| **异常得分** | {score:.4f} |
| **置信度** | {confidence:.2%} |
| **异常区域占比** | {anomaly_ratio:.2f}% |
| **阈值** | {ANOMALY_THRESHOLD} |

---
**说明**: 
- 异常得分 > {ANOMALY_THRESHOLD} 判定为异常
- 热力图中红色区域表示高异常概率
"""
        return result


# 创建全局检测器实例
detector = AnomalyDetector()


# ============================================
# Gradio 界面函数
# ============================================

def process_image(model_name: str, input_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    处理图片的 Gradio 回调函数
    
    Args:
        model_name: 选择的模型名称
        input_image: 上传的图片
    
    Returns:
        Tuple: (原始图片, 热力图, 结果文本)
    """
    if input_image is None:
        return None, None, "⚠️ 请先上传图片"
    
    # 加载模型
    success = detector.load_model(model_name)
    if not success:
        return (input_image, input_image, 
                f"❌ 无法加载模型: {model_name}\n\n"
                f"请先运行训练脚本:\n"
                f"```bash\n"
                f"python train_evaluate.py --model {model_name} --category <your_category>\n"
                f"```")
    
    # 进行推理
    original, heatmap, result = detector.predict(input_image)
    
    return original, heatmap, result


def get_model_description(model_name: str) -> str:
    """获取模型描述"""
    model_info = MODEL_WEIGHTS.get(model_name, {})
    return model_info.get('description', '暂无描述')


# ============================================
# 构建 Gradio 界面
# ============================================

def create_ui():
    """创建 Gradio UI 界面"""
    
    # 自定义 CSS
    css = """
    .input-image { max-height: 400px; }
    .output-image { max-height: 400px; }
    .result-text { font-size: 14px; }
    .model-desc { font-size: 13px; line-height: 1.6; }
    """
    
    with gr.Blocks(css=css, title="工业异常检测演示") as demo:
        
        # 标题区域
        gr.Markdown("""
        # 🔍 无监督工业图像异常检测演示
        
        基于 **anomalib** 的三种最易复现算法对比演示
        
        ### 支持的算法对比
        
        | 算法 | 方向 | 复现难度 | 推理速度 |
        |:---|:---:|:---:|:---:|
        | **PatchCore** | 特征建模 | ⭐ 极易 | ⚡⚡⚡ 最快 |
        | **AutoEncoder** | 重构 | ⭐⭐ 容易 | ⚡⚡ 中等 |
        | **DRAEM** | 自监督学习 | ⭐⭐⭐ 中等 | ⚡ 较慢 |
        """)
        
        with gr.Row():
            # 左侧：控制面板
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ 控制面板")
                
                # 模型选择
                model_dropdown = gr.Dropdown(
                    choices=['patchcore', 'autoencoder', 'draem'],
                    value='patchcore',
                    label="选择算法模型",
                    info="选择用于推理的异常检测算法"
                )
                
                # 模型描述
                model_desc = gr.Markdown(
                    get_model_description('patchcore'),
                    elem_classes=["model-desc"]
                )
                
                # 图片上传
                image_input = gr.Image(
                    type="numpy",
                    label="📤 上传测试图片",
                    image_mode="RGB"
                )
                
                # 运行按钮
                run_button = gr.Button("🚀 开始检测", variant="primary")
                
                # 示例图片（如果存在）
                example_paths = []
                for ext in ['*.png', '*.jpg', '*.jpeg']:
                    example_paths.extend(Path('./data').rglob(f'test/good/{ext}'))
                    if len(example_paths) < 2:
                        example_paths.extend(Path('./data').rglob(f'test/defect/{ext}'))
                    if len(example_paths) >= 2:
                        break
                
                if example_paths:
                    gr.Markdown("### 📋 示例图片")
                    gr.Examples(
                        examples=[[str(p)] for p in example_paths[:2]],
                        inputs=image_input,
                        label="点击加载示例"
                    )
            
            # 右侧：结果展示
            with gr.Column(scale=2):
                gr.Markdown("### 📊 检测结果")
                
                with gr.Row():
                    # 原始图片
                    original_output = gr.Image(
                        type="numpy",
                        label="原始图片",
                        elem_classes=["output-image"]
                    )
                    
                    # 热力图
                    heatmap_output = gr.Image(
                        type="numpy",
                        label="异常热力图",
                        elem_classes=["output-image"]
                    )
                
                # 结果文本
                result_output = gr.Markdown(
                    label="检测结果",
                    elem_classes=["result-text"]
                )
        
        # 底部说明
        gr.Markdown("""
        ---
        ### 📖 使用说明
        
        1. **选择模型**: 从下拉菜单选择要使用的算法
           - **PatchCore**: 无需训练，推理最快，推荐作为 baseline
           - **AutoEncoder**: 经典重构方法，容易理解和调试
           - **DRAEM**: 自监督方法，适合无真实异常样本的场景
        
        2. **上传图片**: 点击上传按钮选择测试图片（支持 PNG, JPG, BMP 格式）
        
        3. **开始检测**: 点击"开始检测"按钮进行推理
        
        4. **查看结果**: 
           - 左侧显示原始图片
           - 右侧显示异常热力图（红色区域表示异常概率高）
           - 下方显示详细的检测结果和得分
        
        ### ⚠️ 注意事项
        
        - 首次使用某模型时，系统会自动加载模型权重
        - 如果模型权重不存在，请先运行训练脚本：`python train_evaluate.py --model <model_name> --category <category>`
        - **PatchCore** 无需训练 epoch，直接构建特征记忆库即可推理
        - 推荐使用 GPU 进行推理以获得更快的速度
        """)
        
        # 事件绑定
        model_dropdown.change(
            fn=get_model_description,
            inputs=model_dropdown,
            outputs=model_desc
        )
        
        run_button.click(
            fn=process_image,
            inputs=[model_dropdown, image_input],
            outputs=[original_output, heatmap_output, result_output]
        )
        
        # 图片上传后自动触发
        image_input.change(
            fn=process_image,
            inputs=[model_dropdown, image_input],
            outputs=[original_output, heatmap_output, result_output]
        )
    
    return demo


# ============================================
# 主函数
# ============================================

def main():
    """主函数"""
    print("="*60)
    print("🚀 启动工业异常检测演示界面")
    print("="*60)
    print("\n📋 本次演示包含以下 3 个最易复现的算法:")
    print("   1. PatchCore    - 特征建模 (⭐ 极易)")
    print("   2. AutoEncoder  - 重构 (⭐⭐ 容易)")
    print("   3. DRAEM        - 自监督 (⭐⭐⭐ 中等)")
    
    # 检查模型权重是否存在
    print("\n📂 检查模型权重...")
    for model_name, info in MODEL_WEIGHTS.items():
        weight_path = Path(info['path'])
        found = False
        if weight_path.exists():
            found = True
        else:
            # 尝试其他路径
            for alt in weight_path.parent.glob('*.ckpt'):
                found = True
                break
        
        if found:
            print(f"   ✅ {model_name}: 已找到权重")
        else:
            print(f"   ⚠️  {model_name}: 权重不存在，请先训练")
            print(f"      训练命令: python train_evaluate.py --model {model_name} --category <your_category>")
    
    # 创建界面
    demo = create_ui()
    
    # 启动服务
    print("\n🌐 正在启动 Gradio 服务...")
    demo.launch(
        server_name="0.0.0.0",  # 允许外部访问
        server_port=7860,
        share=False,  # 设置为 True 可生成公共链接
        show_error=True,
        quiet=False
    )


if __name__ == '__main__':
    main()
