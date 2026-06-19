#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
UI 启动入口 — FastAPI (Phase 2) / Gradio (fallback)
================================================================================

用法:
    python scripts/run_ui.py              # FastAPI UI → http://127.0.0.1:8000 (自动打开浏览器)
    python scripts/run_ui.py --no-browser # 不自动打开浏览器
    python scripts/run_ui.py --gradio     # Gradio UI → http://127.0.0.1:7860
    python scripts/run_ui.py --port 3000  # 自定义端口
================================================================================
"""

import io
import argparse
import sys
import threading
import webbrowser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 设置 Windows 终端编码为 UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, str(PROJECT_ROOT))
from modules._runtime import configure_runtime_temp

configure_runtime_temp()


def print_banner(mode: str, host: str, port: int):
    """打印启动横幅。"""
    print()
    print("=" * 70)
    if mode == "fastapi":
        print("[UI] 启动 FastAPI UI (Phase 2)")
    else:
        print("[UI] 启动 Gradio UI (legacy fallback)")
    print("=" * 70)
    print(f"  访问地址: http://{host}:{port}")
    if mode == "fastapi":
        print(f"  API 文档: http://{host}:{port}/docs")
    print("=" * 70)
    print()


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="工业异常检测 UI 启动器 — FastAPI (默认) / Gradio (fallback)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/run_ui.py              # FastAPI → http://127.0.0.1:8000
  python scripts/run_ui.py --no-browser # 不自动打开浏览器
  python scripts/run_ui.py --gradio     # Gradio → http://127.0.0.1:7860
  python scripts/run_ui.py --port 3000  # 自定义端口
        """
    )
    parser.add_argument("--port", type=int, default=8000,
                        help="服务端口 (FastAPI 默认 8000, Gradio 默认 7860)")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="绑定地址（默认 127.0.0.1）")
    parser.add_argument("--gradio", action="store_true",
                        help="启动 Gradio UI (legacy fallback)")
    parser.add_argument("--no-browser", action="store_true",
                        help="不自动打开浏览器")
    parser.add_argument("--share", action="store_true",
                        help="Gradio: 生成公开分享链接")
    parser.add_argument("--category", type=str, default=None,
                        help="Gradio: 默认数据集 (region1/bottle/...)")

    return parser.parse_args()


def main():
    """主函数。"""
    args = parse_args()

    if args.gradio:
        # ── Gradio fallback ──
        # Gradio 默认端口 7860，如果用户未显式指定则使用 7860
        gradio_port = args.port if args.port != 8000 else 7860
        print_banner("gradio", args.host, gradio_port)

        print("[INFO] 正在加载模块...")
        print()

        try:
            from modules.ui.demo import create_interface, detector, MODEL_CONFIGS

            print(f"[OK] 已加载 {len(MODEL_CONFIGS)} 个模型配置")
            print()

            demo = create_interface(default_dataset=args.category)
            demo.launch(
                server_name=args.host,
                server_port=gradio_port,
                share=args.share,
                show_error=True,
                inbrowser=True,
            )
        except KeyboardInterrupt:
            print("\n[INFO] 用户中断，程序退出")
        except Exception as e:
            print(f"\n[ERROR] UI 启动失败: {e}")
            import traceback
            traceback.print_exc()
            input("\n按回车键退出...")
    else:
        # ── FastAPI UI (default) ──
        try:
            import uvicorn
        except ImportError:
            print("[ERROR] 未安装 uvicorn，请执行: pip install uvicorn")
            sys.exit(1)

        print_banner("fastapi", args.host, args.port)

        # 在启动前验证关键依赖可用
        try:
            from modules.ui.demo import detector, MODEL_CONFIGS, get_available_datasets
            datasets = get_available_datasets()
            print(f"[OK] 检测到 {len(datasets)} 个数据集: {datasets}")
            print(f"[OK] 检测到 {len(MODEL_CONFIGS)} 个模型: {list(MODEL_CONFIGS.keys())}")
        except Exception as e:
            print(f"[WARN] 依赖检查部分失败: {e}")
            print("[WARN] 服务仍将启动，部分功能可能不可用")

        print()
        print("[INFO] 按 Ctrl+C 停止服务")
        print()

        # 自动打开浏览器（默认开启，--no-browser 可禁用）
        if not args.no_browser:
            url = f"http://{args.host}:{args.port}"
            threading.Timer(1.5, lambda: webbrowser.open(url)).start()

        uvicorn.run(
            "modules.ui.server:app",
            host=args.host,
            port=args.port,
            reload=False,
            log_level="info",
        )


if __name__ == "__main__":
    main()
