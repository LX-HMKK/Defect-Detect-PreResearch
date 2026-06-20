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


def test_index_html_links_redesign_assets():
    text = _html_text()
    assert "/static/css/apple-redesign.css" in text
    assert "/static/js/hero-visual.js" in text


def test_index_html_has_new_structure():
    text = _html_text()
    assert "hero-visual" in text
    assert "bento-grid" in text
    assert "workbench" in text
    assert "compare-wall" in text


def test_page_indicator_uses_dynamic_section_count():
    """页码 label 应使用 sectionCount，而不是硬编码 3，避免 section 数量变化时显示错误。"""
    text = _html_text()
    assert "sectionCount" in text
    assert "x-text=\"(currentSection + 1) + ' / ' + sectionCount\"" in text
    assert "' / 3'" not in text


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
