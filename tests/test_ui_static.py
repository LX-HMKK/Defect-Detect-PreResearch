import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC = PROJECT_ROOT / "modules" / "ui" / "static"
HTML = STATIC / "index.html"


def test_apple_redesign_css_exists():
    assert (STATIC / "css" / "apple-redesign.css").exists()


def test_hero_visual_js_exists():
    assert (STATIC / "js" / "hero-visual.js").exists()


def test_index_html_links_redesign_assets():
    text = HTML.read_text(encoding="utf-8")
    assert "/static/css/apple-redesign.css" in text
    assert "/static/js/hero-visual.js" in text


def test_index_html_has_new_structure():
    text = HTML.read_text(encoding="utf-8")
    assert "hero-visual" in text
    assert "bento-grid" in text
    assert "workbench" in text
    assert "compare-wall" in text
