"""
主题管理器 — Apple 风格亮/暗双模式色板定义与 CSS 生成。

将色板从 styles.css 硬编码抽离为 Python 字典（DARK/LIGHT），提供：
- build_css_variables(): 色板 → CSS 自定义属性块编译函数
- get_light_css(): 亮色模式 CSS（html[data-theme="light"] + @media 降级兜底）
- get_theme_switch_html(): 太阳/月亮 SVG 切换按钮 HTML + 内联样式
- get_theme_js(): 主题切换 <script> 标签（localStorage + data-theme）
- get_favicon_html(): SVG 菱形图标 <link> 标签

独立运行：python modules/ui/theme.py → 输出完整 CSS 到 stdout。
"""

from pathlib import Path


# ============================================================================
# 色板定义 — 仅含主题切换时需要改变的变量
# 不变量（accent/ok/bad/warn/圆角/字体/动效）保留在 styles.css 的 :root 块
# ============================================================================

DARK = {
    # ── 基底色阶 ──
    'bg_root': '#000000',
    'bg_system': '#1c1c1e',
    'bg_secondary': '#2c2c2e',
    'bg_tertiary': '#3a3a3c',
    # ── 分隔（透明表达层级）──
    'sep_subtle': 'rgba(255, 255, 255, 0.06)',
    'sep_default': 'rgba(255, 255, 255, 0.10)',
    'sep_strong': 'rgba(255, 255, 255, 0.16)',
    # ── 文字层级 ──
    'text': 'rgba(255, 255, 255, 0.92)',
    'text_secondary': 'rgba(255, 255, 255, 0.60)',
    'text_tertiary': 'rgba(255, 255, 255, 0.36)',
    # ── 阴影（暗色：亮边模拟物理边缘反射 + 深影深度）──
    'shadow_sm': (
        '0 0 0 0.5px rgba(255, 255, 255, 0.04), '
        '0 1px 4px rgba(0, 0, 0, 0.2)'
    ),
    'shadow_md': (
        '0 0 0 0.5px rgba(255, 255, 255, 0.06), '
        '0 2px 8px rgba(0, 0, 0, 0.3), '
        '0 8px 32px rgba(0, 0, 0, 0.4)'
    ),
    'shadow_lg': (
        '0 0 0 0.5px rgba(255, 255, 255, 0.08), '
        '0 4px 16px rgba(0, 0, 0, 0.35), '
        '0 16px 48px rgba(0, 0, 0, 0.5)'
    ),
    'shadow_glow': (
        '0 0 0 0.5px rgba(255, 255, 255, 0.06), '
        '0 2px 8px rgba(0, 0, 0, 0.3), '
        '0 0 32px var(--accent-glow)'
    ),
    # ── 骨架屏 shimmer 色 ──
    'shimmer_color': 'rgba(255, 255, 255, 0.06)',
    # ── Gradio 系统变量覆盖 ──
    'body_background_fill': 'var(--bg-root)',
    'background_fill_primary': 'var(--bg-system)',
    'background_fill_secondary': 'var(--bg-secondary)',
    'border_color_primary': 'var(--sep-subtle)',
    'input_background_fill': 'var(--bg-secondary)',
}

LIGHT = {
    # ── 基底色阶 ──
    'bg_root': '#f0f0f0',
    'bg_system': '#ffffff',
    'bg_secondary': '#f5f5f7',
    'bg_tertiary': '#e8e8ed',
    # ── 分隔 ──
    'sep_subtle': 'rgba(0, 0, 0, 0.06)',
    'sep_default': 'rgba(0, 0, 0, 0.10)',
    'sep_strong': 'rgba(0, 0, 0, 0.16)',
    # ── 文字层级 ──
    'text': 'rgba(0, 0, 0, 0.88)',
    'text_secondary': 'rgba(0, 0, 0, 0.55)',
    'text_tertiary': 'rgba(0, 0, 0, 0.30)',
    # ── 阴影（亮色：暗边更轻，阴影更散）──
    'shadow_sm': (
        '0 0 0 0.5px rgba(0, 0, 0, 0.04), '
        '0 1px 4px rgba(0, 0, 0, 0.06)'
    ),
    'shadow_md': (
        '0 0 0 0.5px rgba(0, 0, 0, 0.04), '
        '0 1px 4px rgba(0, 0, 0, 0.06), '
        '0 8px 24px rgba(0, 0, 0, 0.08)'
    ),
    'shadow_lg': (
        '0 0 0 0.5px rgba(0, 0, 0, 0.06), '
        '0 4px 12px rgba(0, 0, 0, 0.08), '
        '0 16px 40px rgba(0, 0, 0, 0.12)'
    ),
    'shadow_glow': (
        '0 0 0 0.5px rgba(0, 0, 0, 0.04), '
        '0 2px 8px rgba(0, 0, 0, 0.08), '
        '0 0 32px rgba(41, 151, 255, 0.20)'
    ),
    # ── 骨架屏 shimmer 色 ──
    'shimmer_color': 'rgba(0, 0, 0, 0.06)',
    # ── Gradio 系统变量覆盖 ──
    'body_background_fill': 'var(--bg-root)',
    'background_fill_primary': 'var(--bg-system)',
    'background_fill_secondary': 'var(--bg-secondary)',
    'border_color_primary': 'var(--sep-subtle)',
    'input_background_fill': 'var(--bg-secondary)',

    # ── Gradio 中性色阶（亮色模式）──
    # 暗色模式下 Gradio 通过 @media 设置暗色中性色阶，需显式覆盖
    'neutral_700': '#e4e4e7',
    'neutral_800': '#d4d4d8',
    'neutral_900': '#a1a1aa',
    'neutral_950': '#71717a',

    # ── Gradio 组件级变量（亮色值）──
    'block_background_fill': 'var(--bg-system)',
    'block_label_background_fill': 'var(--bg-system)',
    'panel_background_fill': 'var(--bg-secondary)',
    'body_text_color': 'var(--text)',
    'body_text_color_subdued': 'var(--text-secondary)',
    'block_label_text_color': 'var(--text-secondary)',
    'block_title_text_color': 'var(--text-secondary)',
    'block_border_color': 'var(--sep-subtle)',
    'panel_border_color': 'var(--sep-subtle)',
    'input_border_color': 'var(--sep-subtle)',
    'accordion_text_color': 'var(--text)',
}


# ============================================================================
# CSS 生成函数
# ============================================================================

def _key_to_var(key: str) -> str:
    """将色板 key 转换为 CSS 变量名。bg_root → --bg-root"""
    return '--' + key.replace('_', '-')


def build_css_variables(palette: dict, selector: str = ':root') -> str:
    """
    将色板字典编译为 CSS 自定义属性块。

    Args:
        palette: 色板字典，key 为 snake_case，value 为 CSS 值。
        selector: CSS 选择器，默认 ':root'。

    Returns:
        str: 格式化的 CSS 块（带缩进）。
    """
    lines = [f'{selector} {{']
    for key, value in palette.items():
        var_name = _key_to_var(key)
        lines.append(f'    {var_name}: {value};')
    lines.append('}')
    return '\n'.join(lines)


def get_dark_css() -> str:
    """
    暗色模式 CSS（默认 :root 块）。

    注意：暗色变量实际定义在 styles.css 的 :root 块中，
    此函数仅用于独立测试输出完整 CSS。运行时暗色变量由 styles.css 提供。
    """
    return build_css_variables(DARK, ':root')


def get_light_css() -> str:
    """
    亮色模式 CSS。

    包含两层：
    1. html[data-theme="light"] — JS 手动切换时启用（高优先级）
    2. @media (prefers-color-scheme: light) — JS 禁用时的降级兜底

    通过 gr.HTML 注入以绕过 Gradio 6 CSS 作用域处理。
    """
    manual = build_css_variables(LIGHT, 'html[data-theme="light"]')

    # @media 降级兜底：仅当 JS 禁用（无 data-theme 属性）时跟随系统主题
    # :not([data-theme]) 确保手动偏好（data-theme="dark"）不被 @media 覆盖
    media_fallback = build_css_variables(LIGHT, '@media (prefers-color-scheme: light) {\n    :root:not([data-theme])')

    # 亮色模式 body 环境光（比暗色更淡）
    body_glow = (
        'html[data-theme="light"] body::before {\n'
        '    background: radial-gradient(ellipse, rgba(41, 151, 255, 0.02) 0%, transparent 70%);\n'
        '}'
    )

    return f'{manual}\n\n{body_glow}\n\n{media_fallback}\n    }}'


# ============================================================================
# 主题切换按钮
# ============================================================================

def get_theme_switch_html() -> str:
    """
    主题切换按钮 HTML + 内联样式。

    两个圆形按钮并排：太阳（亮色模式）/ 月亮（暗色模式）。
    当前激活的图标高亮（var(--accent)），另一个灰色（var(--text-tertiary)）。
    尺寸 28×28px，间距 4px。
    """
    return '''
<div class="theme-switch" id="theme-switch" style="display: flex; gap: 4px; align-items: center;">
    <button class="theme-btn theme-btn-light"
            data-theme="light"
            title="亮色模式"
            style="
                width: 28px; height: 28px;
                border: none; border-radius: 50%;
                background: transparent;
                cursor: pointer;
                display: flex; align-items: center; justify-content: center;
                padding: 0;
                color: var(--text-tertiary);
                transition: color 180ms cubic-bezier(0,0,0.2,1),
                            background 180ms cubic-bezier(0,0,0.2,1),
                            transform 0.35s cubic-bezier(0.22,0.8,0.3,1.15);
            "
            onmouseover="this.style.background='rgba(128,128,128,0.12)'"
            onmouseout="this.style.background='transparent'"
    >
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
            <circle cx="8" cy="8" r="3.5" stroke="currentColor" stroke-width="1.2"/>
            <line x1="8" y1="1" x2="8" y2="2.5" stroke="currentColor" stroke-width="1.2" stroke-linecap="round"/>
            <line x1="8" y1="13.5" x2="8" y2="15" stroke="currentColor" stroke-width="1.2" stroke-linecap="round"/>
            <line x1="1" y1="8" x2="2.5" y2="8" stroke="currentColor" stroke-width="1.2" stroke-linecap="round"/>
            <line x1="13.5" y1="8" x2="15" y2="8" stroke="currentColor" stroke-width="1.2" stroke-linecap="round"/>
            <line x1="3.05" y1="3.05" x2="4.1" y2="4.1" stroke="currentColor" stroke-width="1.2" stroke-linecap="round"/>
            <line x1="11.9" y1="11.9" x2="12.95" y2="12.95" stroke="currentColor" stroke-width="1.2" stroke-linecap="round"/>
            <line x1="3.05" y1="12.95" x2="4.1" y2="11.9" stroke="currentColor" stroke-width="1.2" stroke-linecap="round"/>
            <line x1="11.9" y1="4.1" x2="12.95" y2="3.05" stroke="currentColor" stroke-width="1.2" stroke-linecap="round"/>
        </svg>
    </button>
    <button class="theme-btn theme-btn-dark"
            data-theme="dark"
            title="暗色模式"
            style="
                width: 28px; height: 28px;
                border: none; border-radius: 50%;
                background: transparent;
                cursor: pointer;
                display: flex; align-items: center; justify-content: center;
                padding: 0;
                color: var(--accent);
                transition: color 180ms cubic-bezier(0,0,0.2,1),
                            background 180ms cubic-bezier(0,0,0.2,1),
                            transform 0.35s cubic-bezier(0.22,0.8,0.3,1.15);
            "
            onmouseover="this.style.background='rgba(128,128,128,0.12)'"
            onmouseout="this.style.background='transparent'"
    >
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M13.5 10.5A6 6 0 0 1 5.5 2.5 5 5 0 1 0 13.5 10.5Z"
                  stroke="currentColor" stroke-width="1.2" stroke-linejoin="round"/>
        </svg>
    </button>
</div>'''


# ============================================================================
# 主题切换 JavaScript
# ============================================================================

def get_theme_js() -> str:
    """
    主题切换 JavaScript 注入。

    Gradio 6 基于 Svelte，通过 innerHTML 注入的 <script> 标签不会被执行。
    解决方案：
    1. <script type="text/plain"> 保存完整代码（浏览器不执行，文本保留）
    2. <img onerror> bootstrapper —— 从隐藏区读取代码，
       用 document.createElement('script') 动态创建真实脚本注入 <head>
    """
    static_dir = Path(__file__).parent / 'static'
    js_path = static_dir / 'theme.js'
    js_content = js_path.read_text(encoding='utf-8')

    # Bootstrapper: 从 text/plain 容器读出代码，创建真实 <script> 注入 <head>
    bootstrap = (
        '<img src="x" style="display:none" onerror="'
        'var s=document.createElement(\'script\');'
        's.textContent=document.getElementById(\'theme-js-source\').textContent;'
        'document.head.appendChild(s);'
        '">'
    )

    return (
        f'<script type="text/plain" id="theme-js-source">\n{js_content}\n</script>\n'
        f'{bootstrap}'
    )


# ============================================================================
# Favicon
# ============================================================================

# 供 theme.js 引用的 favicon data URI（模块级常量）
# 注意：SVG 属性使用单引号，避免双引号与 HTML href="..." 属性冲突
_FAVICON_DARK_SVG = (
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'>"
    "<circle cx='16' cy='16' r='16' fill='%231c1c1e'/>"
    "<polygon points='16,4 28,16 16,28 4,16' fill='%232997ff' opacity='0.9'/>"
    "</svg>"
)

FAVICON_DARK_URI = f'data:image/svg+xml,{_FAVICON_DARK_SVG}'


def get_favicon_html() -> str:
    """
    SVG 菱形图标 Favicon。

    返回单个 <link rel="icon" id="favicon"> 标签，默认使用暗色 favicon。
    theme.js 会在主题切换时动态更新 href 属性（亮色 URI 内联在 JS 中）。
    """
    return f'<link rel="icon" id="favicon" href="{FAVICON_DARK_URI}" type="image/svg+xml">'


# ============================================================================
# 推理交互 JavaScript
# ============================================================================


def get_inference_js() -> str:
    """
    推理结果交互增强 JS 注入。

    使用与 get_theme_js() 相同的 bootstrapper 模式绕过 Svelte 的 <script> 封锁。
    JS 文件：modules/ui/static/inference-interact.js
    """
    static_dir = Path(__file__).parent / 'static'
    js_path = static_dir / 'inference-interact.js'
    js_content = js_path.read_text(encoding='utf-8')

    bootstrap = (
        '<img src="x" style="display:none" onerror="'
        'var s=document.createElement(\'script\');'
        's.textContent=document.getElementById(\'inference-js-source\').textContent;'
        'document.head.appendChild(s);'
        '">'
    )

    return (
        f'<script type="text/plain" id="inference-js-source">\n{js_content}\n</script>\n'
        f'{bootstrap}'
    )


# ============================================================================
# 独立运行
# ============================================================================

def get_all_css() -> str:
    """
    返回完整 CSS（供独立测试）。

    包含暗色 :root 块 + 亮色 [data-theme] 块 + @media 降级兜底。
    """
    dark = get_dark_css()
    light = get_light_css()
    return (
        f'/* ===== 暗色模式（默认:root）===== */\n'
        f'{dark}\n\n'
        f'/* ===== 亮色模式（手动切换 + 系统降级）===== */\n'
        f'{light}\n'
    )


if __name__ == '__main__':
    print(get_all_css())
