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
