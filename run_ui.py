#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
入口脚本 4: 启动 UI 演示
用法: python run_ui.py
"""

import sys
import io
import argparse
from pathlib import Path

# 设置 Windows 终端编码为 UTF-8
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))


def print_banner():
    """打印欢迎横幅"""
    print()
    print("=" * 70)
    print("🌐 UI 演示模块")
    print("🎮 基于 Gradio 的交互式 Web 界面")
    print("=" * 70)
    print()
    print("📋 功能:")
    print("   🔽 切换算法 (Ganomaly / PatchCore / DRAEM)")
    print("   📤 选择/上传测试图片")
    print("   🚀 执行异常检测推理")
    print("   🖼️  并排显示原图和缺陷热力图")
    print()
    print("=" * 70)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='UI 演示 - Gradio Web 界面',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_ui.py                        # 使用默认端口 7860
  python run_ui.py --port 8080            # 指定端口
  python run_ui.py --share                # 创建分享链接
        """
    )
    parser.add_argument('--port', '-p', type=int, default=7860,
                        help='Web 服务端口（默认 7860）')
    parser.add_argument('--share', '-s', action='store_true',
                        help='创建公开分享链接')
    parser.add_argument('--host', type=str, default='127.0.0.1',
                        help='服务地址（默认 127.0.0.1）')
    
    return parser.parse_args()


def main():
    """主函数"""
    print_banner()
    
    args = parse_args()
    
    # 打印配置信息
    print()
    print("⚙️  配置信息")
    print("-" * 70)
    print(f"   🌐 服务地址:   {args.host}")
    print(f"   🔌 服务端口:   {args.port}")
    print(f"   🔗 分享链接:   {'是' if args.share else '否'}")
    print("-" * 70)
    print()
    
    # 启动 UI
    from modules.ui.demo import main as ui_main
    
    print("🚀 正在启动 Web 服务...")
    print()
    print(f"🌐 请在浏览器访问: http://{args.host}:{args.port}")
    print()
    print("💡 按 Ctrl+C 可停止服务")
    print("=" * 70)
    print()
    
    # 这里需要传递参数给 UI 模块
    # 由于 demo.py 的 main() 可能需要参数，我们先加载它
    import gradio as gr
    from modules.ui.demo import (
        create_ui,
        load_model,
        predict_anomaly,
        MODEL_CONFIGS,
    )
    
    # 创建并启动 UI
    ui = create_ui()
    ui.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == '__main__':
    main()
