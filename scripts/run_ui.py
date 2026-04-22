#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
入口脚本 4: 启动 UI 演示
用法: python scripts/run_ui.py
"""

import io
import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 设置 Windows 终端编码为 UTF-8
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 添加项目根目录到路径
def _configure_runtime_temp() -> None:
    temp_dir = PROJECT_ROOT / "temp"
    pycache_dir = temp_dir / "pycache"
    temp_dir.mkdir(exist_ok=True)
    pycache_dir.mkdir(exist_ok=True)
    sys.pycache_prefix = str(pycache_dir)
    os.environ["PYTHONPYCACHEPREFIX"] = str(pycache_dir)


_configure_runtime_temp()

sys.path.insert(0, str(PROJECT_ROOT))


def print_banner():
    """打印欢迎横幅"""
    print()
    print("=" * 70)
    print("🌐 UI 演示模块")
    print("🎮 基于 Gradio 的交互式 Web 界面")
    print("=" * 70)
    print()
    print("📋 功能:")
    print("   🔽 切换算法 (FRE / PatchCore / DRAEM)")
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
  python scripts/run_ui.py                        # 使用默认端口 7860
  python scripts/run_ui.py --port 8080            # 指定端口
  python scripts/run_ui.py --share                # 创建分享链接
        """
    )
    parser.add_argument('--port', '-p', type=int, default=7860,
                        help='Web 服务端口（默认 7860）')
    parser.add_argument('--share', '-s', action='store_true',
                        help='创建公开分享链接')
    parser.add_argument('--category', '-c', type=str, default='region1',
                        help='数据集 (region1/bottle)')
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
    print(f"   📦 数据集:     {args.category}")
    print("-" * 70)
    print()
    
    print("🚀 正在启动 Web 服务...")
    print()
    print(f"🌐 请在浏览器访问: http://{args.host}:{args.port}")
    print()
    print("💡 按 Ctrl+C 可停止服务")
    print("=" * 70)
    print()
    
    try:
        # 这里需要传递参数给 UI 模块
        # demo.py 定义了 create_interface() 函数
        from modules.ui.demo import create_interface, detector, MODEL_CONFIGS
        
        # 不预加载模型，让用户上传图片时才加载
        # 避免卡在模型加载步骤
        
        # 创建并启动 UI
        ui = create_interface(default_dataset=args.category)
        ui.launch(
            server_name=args.host,
            server_port=args.port,
            share=args.share,
            show_error=True,
            inbrowser=True,  # 自动打开浏览器
        )
    except KeyboardInterrupt:
        print("\n[INFO] 用户中断，程序退出")
    except Exception as e:
        print(f"\n[ERROR] UI 启动失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 保持进程运行
    input("\n按回车键退出...")


if __name__ == '__main__':
    main()
