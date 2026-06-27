"""
UI 静态资源回归测试（TDD）

验证 Apple 风格重设计所需的新资源文件、HTML 引用与关键结构类名是否到位。
这些测试在改造完成前会失败，完成后应全部通过。
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC = PROJECT_ROOT / "modules" / "ui" / "static"
HTML = STATIC / "index.html"


def _html_text():
    return HTML.read_text(encoding="utf-8")


def test_apple_redesign_css_exists():
    assert (STATIC / "css" / "apple-redesign.css").exists()


def test_hero_visual_js_exists():
    assert (STATIC / "js" / "hero-visual.js").exists()


def test_algo_carousel_js_exists():
    assert (STATIC / "js" / "algo-carousel.js").exists()


def test_index_html_links_redesign_assets():
    text = _html_text()
    assert "/static/css/apple-redesign.css" in text
    assert "/static/js/hero-visual.js" in text
    assert "/static/js/algo-carousel.js" in text


def test_index_html_has_new_structure():
    text = _html_text()
    assert "hero-visual" in text
    assert "bento-grid" in text
    assert "algo-carousel" in text
    assert "algo-carousel-track" in text
    assert "workbench" in text
    assert "compare-wall" in text


def test_page_indicator_uses_dynamic_page_count():
    """页码 label 应使用 pageCount 动态绑定（5 页逻辑视图：封面 + 4 内容页），
    而非硬编码总数，避免页数变化时显示错误。"""
    text = _html_text()
    assert "pageCount" in text
    assert "String(pageCount).padStart" in text
    assert "currentPage" in text
    # 总数必须由变量驱动，不能硬编码
    assert "' / 05'" not in text
    assert "' / 04'" not in text


def test_app_js_selects_snap_pages_by_class():
    """滚动观察应通过 .snap-page 选取所有 section，而不是依赖 $refs.section*，
    否则嵌套 x-data（如四模型对比区）会导致 section2 被漏掉。"""
    app_js = (STATIC / "js" / "app.js").read_text(encoding="utf-8")
    assert "querySelectorAll('.snap-page')" in app_js
    assert "self.sectionCount = sections.length" in app_js
    # 不应再只依赖 $refs.section0/1/2 来构造 sections 数组
    assert "$refs.section0" not in app_js
    assert "$refs.section1" not in app_js
    assert "$refs.section2" not in app_js


def test_training_draem_uses_safe_batch_default():
    """DRAEM 在 8GB VRAM 上 batch_size=32 会 CUDA OOM，训练在首个 epoch 前崩溃
    → 零 metric 事件 → 无训练曲线。Training Studio 必须为 DRAEM 提供小的默认 batch。"""
    import re
    training_js = (STATIC / "js" / "training.js").read_text(encoding="utf-8")
    assert "modelDefaultBatch" in training_js, "training.js 缺少 modelDefaultBatch 映射"
    # 提取 modelDefaultBatch 对象块
    block = re.search(r"modelDefaultBatch:\s*\{([^}]*)\}", training_js, re.S)
    assert block, "无法解析 modelDefaultBatch 块"
    draem_match = re.search(r"draem:\s*(\d+)", block.group(1))
    assert draem_match, "modelDefaultBatch 缺少 draem 条目"
    draem_batch = int(draem_match.group(1))
    assert draem_batch <= 8, (
        f"DRAEM 默认 batch_size={draem_batch} 过大，会在 8GB VRAM 上 CUDA OOM 导致无训练曲线（应 ≤8）"
    )



def test_training_gallery_does_not_stretch_monitor():
    """大量训练样本上传后，画廊应使用垂直网格内部滚动，不能撑开右侧区域把监控面板挤出视口。"""
    css = (STATIC / "css" / "app.css").read_text(encoding="utf-8")
    assert ".training-right {" in css
    assert "min-width: 0" in css.split(".training-right {")[1].split("}")[0]
    assert ".training-gallery {" in css
    gallery_block = css.split(".training-gallery {")[1].split("}")[0]
    assert "display: grid" in gallery_block
    assert "overflow-y: auto" in gallery_block
    assert "max-height:" in gallery_block


def test_result_dashboard_uses_split_layout():
    """推理结果卡应为左图右信息分栏（I3），移除旧标题栏与重复 badge（I4/I5）。"""
    text = _html_text()
    assert "result-dashboard-grid" in text
    assert "result-dashboard-aside" in text
    assert "inference-metrics--stack" in text
    # 旧的标题栏与重复 badge 已移除
    assert "result-dashboard-header" not in text
    assert "result-badge" not in text
    assert "inference-verdict-wrap" not in text
