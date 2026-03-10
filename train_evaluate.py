"""
训练和评估脚本
调用 anomalib 训练 3 个模型并输出评估指标

支持的算法:
- PatchCore: 基于特征建模 (极易复现，无需训练)
- AutoEncoder: 基于重构 (容易复现，经典方法)
- DRAEM: 基于自监督学习 (中等难度，无需异常样本训练)

评估指标:
- 图像级: AUROC, AUPR
- 像素级: Pixel-level AUROC, PRO

使用方法:
    # 训练单个模型
    python train_evaluate.py --model patchcore --data_path ./data/my_dataset --category my_product
    
    # 训练所有模型
    python train_evaluate.py --model all --data_path ./data/my_dataset --category my_product
    
    # 仅评估（已有权重）
    python train_evaluate.py --model patchcore --data_path ./data/my_dataset --category my_product --eval_only --checkpoint results/patchcore/checkpoints/model.ckpt
"""

import os
import sys
import argparse
import json
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Mapping

import torch
import pandas as pd
from tqdm import tqdm

# 导入 anomalib 相关模块
try:
    from anomalib.config import get_configurable_parameters
    from anomalib.data import get_datamodule
    from anomalib.models import get_model
    from anomalib.utils.callbacks import LoadModelCallback, get_callbacks
    from anomalib.deploy import OpenVINOInferencer, TorchInferencer
    from pytorch_lightning import Trainer, seed_everything
    from omegaconf import OmegaConf, DictConfig, ListConfig
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("请确保已安装 anomalib: pip install anomalib")
    sys.exit(1)


# ============================================
# 支持的模型列表 (3 个最易复现的算法)
# ============================================
SUPPORTED_MODELS = ["patchcore", "autoencoder", "draem"]

# 模型说明（用于对比报告）
MODEL_DESCRIPTIONS = {
    "patchcore": {
        "方向": "特征建模",
        "原理": "预训练CNN提取局部特征，构建记忆库存储正常样本特征",
        "训练难度": "⭐ 极易（无需训练，仅构建记忆库）",
        "推理速度": "⚡⚡⚡ 最快"
    },
    "autoencoder": {
        "方向": "重构",
        "原理": "训练编码器-解码器网络，通过重构误差检测异常",
        "训练难度": "⭐⭐ 容易",
        "推理速度": "⚡⚡ 中等"
    },
    "draem": {
        "方向": "自监督",
        "原理": "通过数据增强生成合成异常，训练判别网络",
        "训练难度": "⭐⭐⭐ 中等",
        "推理速度": "⚡ 较慢"
    }
}


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练和评估异常检测模型')
    
    # 基本参数
    parser.add_argument('--model', type=str, default='patchcore',
                        choices=SUPPORTED_MODELS + ['all'],
                        help='选择要训练的模型 (patchcore/autoencoder/draem/all)')
    parser.add_argument('--data_path', type=str, default='./data/my_dataset',
                        help='数据集根目录路径')
    parser.add_argument('--category', type=str, required=True,
                        help='产品类别名称（对应 MVTec AD 格式的子文件夹名）')
    
    # 训练和评估参数
    parser.add_argument('--eval_only', action='store_true',
                        help='仅进行评估（不训练）')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='评估时使用的模型权重路径')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='覆盖配置文件的 batch size')
    parser.add_argument('--epochs', type=int, default=None,
                        help='覆盖配置文件的训练 epoch 数')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='结果输出目录')
    parser.add_argument('--save_predictions', action='store_true', default=True,
                        help='保存预测结果')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--device', type=str, default='auto',
                        help='计算设备 (auto/cpu/cuda)')
    
    return parser.parse_args()


def load_config(model_name: str, args) -> Any:
    """
    加载并修改配置文件
    
    Args:
        model_name: 模型名称
        args: 命令行参数
    
    Returns:
        Any: 配置对象 (DictConfig)
    """
    config_path = Path(f'./configs/{model_name}.yaml')
    
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    # 加载配置文件
    config = OmegaConf.load(config_path)
    
    # 修改配置参数
    config.dataset.path = args.data_path
    config.dataset.category = args.category
    config.project.path = args.output_dir
    config.project.default_root_dir = f"{args.output_dir}/{model_name}"
    
    # 覆盖 batch size（如果指定）
    if args.batch_size:
        config.dataset.train_batch_size = args.batch_size
        config.dataset.eval_batch_size = args.batch_size
    
    # 覆盖 epoch（如果指定）
    if args.epochs:
        config.trainer.max_epochs = args.epochs
    
    # 设置设备
    if args.device != 'auto':
        config.trainer.accelerator = args.device
    
    # 设置随机种子
    config.project.seed = args.seed
    
    return config


def train_model(config: Any, model_name: str) -> Dict[str, Any]:
    """
    训练单个模型
    
    Args:
        config: 配置对象
        model_name: 模型名称
    
    Returns:
        Dict: 训练结果（包含最佳指标）
    """
    model_desc = MODEL_DESCRIPTIONS.get(model_name, {})
    
    print(f"\n{'='*60}")
    print(f"🚀 开始训练模型: {model_name.upper()}")
    print(f"{'='*60}")
    print(f"📌 方向: {model_desc.get('方向', 'N/A')}")
    print(f"📖 原理: {model_desc.get('原理', 'N/A')}")
    print(f"📊 训练难度: {model_desc.get('训练难度', 'N/A')}")
    print(f"⚡ 推理速度: {model_desc.get('推理速度', 'N/A')}")
    print(f"{'='*60}")
    
    # 设置随机种子
    seed_everything(config.project.seed)
    
    # 创建数据模块
    print("\n📊 加载数据集...")
    datamodule = get_datamodule(config)
    datamodule.setup()
    
    print(f"   训练集样本数: {len(datamodule.train_data)}")
    print(f"   测试集样本数: {len(datamodule.test_data)}")
    
    # 创建模型
    print(f"\n🔧 创建 {model_name} 模型...")
    model = get_model(config)
    
    # 创建回调函数
    callbacks = get_callbacks(config, model)
    
    # 创建训练器
    trainer = Trainer(
        max_epochs=config.trainer.max_epochs,
        accelerator=config.trainer.accelerator,
        devices=config.trainer.devices,
        callbacks=callbacks,
        gradient_clip_val=config.trainer.get('gradient_clip_val', 0),
        accumulate_grad_batches=config.trainer.get('accumulate_grad_batches', 1),
        check_val_every_n_epoch=config.trainer.get('check_val_every_n_epoch', 1),
        default_root_dir=config.project.default_root_dir,
    )
    
    # 训练模型
    print("\n⏳ 开始训练...")
    if model_name == 'patchcore':
        print("   💡 PatchCore 无需训练 epoch，正在构建特征记忆库...")
    
    trainer.fit(model=model, datamodule=datamodule)  # type: ignore
    
    # 测试模型
    print("\n🧪 开始测试...")
    results = trainer.test(model=model, datamodule=datamodule)  # type: ignore
    
    # 返回测试结果
    return results[0] if results else {}


def evaluate_model(config: Any, checkpoint_path: str) -> Dict[str, Any]:
    """
    评估已有模型
    
    Args:
        config: 配置对象
        checkpoint_path: 模型权重路径
    
    Returns:
        Dict: 评估结果
    """
    print(f"\n{'='*60}")
    print(f"🧪 开始评估模型")
    print(f"{'='*60}")
    
    # 创建数据模块
    print("\n📊 加载数据集...")
    datamodule = get_datamodule(config)
    datamodule.setup()
    
    # 创建模型
    print("\n🔧 加载模型...")
    model = get_model(config)
    
    # 从 checkpoint 加载权重
    callbacks = [LoadModelCallback(weights_path=checkpoint_path)]
    
    # 创建训练器
    trainer = Trainer(
        accelerator=config.trainer.accelerator,
        devices=config.trainer.devices,
        callbacks=callbacks,
        default_root_dir=config.project.default_root_dir,
    )
    
    # 评估模型
    print("\n⏳ 开始评估...")
    results = trainer.test(model=model, datamodule=datamodule)  # type: ignore
    
    return results[0] if results else {}


def export_results(results: Dict[str, Any], model_name: str, output_dir: str, category: str):
    """
    导出评估结果到文件
    
    Args:
        results: 评估结果字典
        model_name: 模型名称
        output_dir: 输出目录
        category: 产品类别
    """
    # 创建结果目录
    result_dir = Path(output_dir) / 'comparison'
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # 添加元信息
    results_with_meta = {
        'model': model_name,
        'category': category,
        'timestamp': datetime.now().isoformat(),
        'metrics': results
    }
    
    # 保存为 JSON
    json_path = result_dir / f'{model_name}_{category}_results.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_with_meta, f, indent=2, ensure_ascii=False)
    
    print(f"\n📄 结果已保存: {json_path}")


def compare_models(results_dict: Dict[str, Dict[str, Any]], output_dir: str, category: str):
    """
    对比多个模型的结果并生成报告
    
    Args:
        results_dict: 模型名称到结果的映射
        output_dir: 输出目录
        category: 产品类别
    """
    print(f"\n{'='*60}")
    print("📊 模型对比结果")
    print(f"{'='*60}")
    
    # 提取关键指标
    metrics_data = []
    for model_name, results in results_dict.items():
        model_desc = MODEL_DESCRIPTIONS.get(model_name, {})
        row = {
            'Model': model_name.upper(),
            '方向': model_desc.get('方向', 'N/A'),
            'Image AUROC (%)': results.get('image_AUROC', 0) * 100,
            'Image AUPR (%)': results.get('image_AUPR', 0) * 100,
            'Pixel AUROC (%)': results.get('pixel_AUROC', 0) * 100,
            'PRO (%)': results.get('pixel_PRO', 0) * 100,
        }
        metrics_data.append(row)
    
    # 创建 DataFrame
    df = pd.DataFrame(metrics_data)
    
    # 打印表格
    print("\n" + df.to_string(index=False))
    
    # 保存为 CSV
    result_dir = Path(output_dir) / 'comparison'
    result_dir.mkdir(parents=True, exist_ok=True)
    csv_path = result_dir / f'comparison_{category}.csv'
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n📊 对比表格已保存: {csv_path}")
    
    # 生成 Markdown 报告
    md_path = result_dir / f'report_{category}.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# 异常检测模型对比报告\n\n")
        f.write(f"**产品类别**: {category}\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 模型说明
        f.write("## 模型说明\n\n")
        for model_name, desc in MODEL_DESCRIPTIONS.items():
            f.write(f"### {model_name.upper()}\n")
            f.write(f"- **方向**: {desc['方向']}\n")
            f.write(f"- **原理**: {desc['原理']}\n")
            f.write(f"- **训练难度**: {desc['训练难度']}\n")
            f.write(f"- **推理速度**: {desc['推理速度']}\n\n")
        
        # 评估结果
        f.write("## 评估指标对比\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n## 指标说明\n\n")
        f.write("- **Image AUROC**: 图像级 ROC 曲线下面积（越高越好）\n")
        f.write("- **Image AUPR**: 图像级 PR 曲线下面积（越高越好）\n")
        f.write("- **Pixel AUROC**: 像素级 ROC 曲线下面积（越高越好）\n")
        f.write("- **PRO**: 区域重叠率 Per-Region Overlap（越高越好）\n")
    
    print(f"📄 报告已保存: {md_path}")


def main():
    """主函数"""
    args = parse_args()
    
    print("="*60)
    print("🔬 无监督工业异常检测 - 训练与评估")
    print("="*60)
    print(f"\n📂 数据集路径: {args.data_path}")
    print(f"📦 产品类别: {args.category}")
    print(f"🎲 随机种子: {args.seed}")
    print(f"\n📋 本次评估的 3 个算法:")
    for i, model in enumerate(SUPPORTED_MODELS, 1):
        desc = MODEL_DESCRIPTIONS[model]
        print(f"   {i}. {model.upper()} ({desc['方向']}) - {desc['训练难度']}")
    
    # 确定要训练的模型
    models_to_run = SUPPORTED_MODELS if args.model == 'all' else [args.model]
    
    # 存储所有结果
    all_results = {}
    
    # 遍历训练/评估每个模型
    for model_name in models_to_run:
        try:
            # 加载配置
            config = load_config(model_name, args)
            
            if args.eval_only and args.checkpoint:
                # 仅评估模式
                results = evaluate_model(config, args.checkpoint)
            else:
                # 训练模式
                results = train_model(config, model_name)
            
            # 保存结果
            all_results[model_name] = results
            export_results(results, model_name, args.output_dir, args.category)
            
        except Exception as e:
            print(f"\n❌ 模型 {model_name} 运行失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 生成对比报告（如果训练了多个模型）
    if len(all_results) > 1:
        compare_models(all_results, args.output_dir, args.category)
    
    print("\n" + "="*60)
    print("✅ 所有任务已完成!")
    print("="*60)


if __name__ == '__main__':
    # 忽略警告
    warnings.filterwarnings('ignore')
    main()
