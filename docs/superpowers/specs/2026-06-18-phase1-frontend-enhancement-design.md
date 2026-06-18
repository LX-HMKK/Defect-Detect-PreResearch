# Phase 1 前端增强设计 — 主题模块 + 手动切换 + 图标

> 版本 1.0 · 2026-06-18 · 依赖关系：无外部依赖，今明两天可交付

---

## 1. 目标

- A2：主题管理器 `theme.py`，色板定义 + CSS 生成，从 `styles.css` 抽离硬编码色值
- I1：手动亮/暗切换按钮，存储偏好到 `localStorage`，覆盖系统设定
- V1：浏览器 Favicon（SVG 内联菱形图标）

---

## 2. 模块结构

```
modules/ui/
├── __init__.py          # 空（已有）
├── demo.py              # 主界面（修改：注入 theme + toggle + favicon）
├── styles.css           # 样式（修改：加 [data-theme="light"] 选择器）
├── theme.py             # 新建：主题管理器
└── static/
    └── theme.js         # 新建：手动切换逻辑
```

---

## 3. A2 — theme.py 设计

### 3.1 接口

```python
# 常量
DARK: dict   # 暗色模式色板
LIGHT: dict  # 亮色模式色板

# 函数
def build_css_variables(palette: dict) -> str
    """将色板编译为 :root { ... } CSS 变量块"""

def get_dark_css() -> str
    """暗色模式 CSS（默认 :root 块）"""

def get_light_css() -> str
    """亮色模式 CSS（[data-theme="light"] 选择器块）"""

def get_theme_switch_html() -> str
    """主题切换按钮 HTML（太阳/月亮图标）"""

def get_theme_js() -> str
    """主题切换 JS 代码（内联 <script> 标签）"""

def get_favicon_html() -> str
    """SVG Favicon HTML"""
```

### 3.2 色板 keys

`bg_root`, `bg_system`, `bg_secondary`, `bg_tertiary`, `sep_subtle`, `sep_default`, `sep_strong`, `text`, `text_secondary`, `text_tertiary`, `shadow_sm`, `shadow_md`, `shadow_lg`, `shadow_glow`

### 3.3 从 styles.css 中删除的内容

- 亮色模式变量定义（已迁移到 `theme.py` 动态生成）
- 暗色模式 `:root` 块中的重复色板（可用 `build_css_variables(DARK)` 生成）

---

## 4. I1 — 手动切换按钮

### 4.1 位置

标题区域右侧浮动：

```
缺陷检测                                    ☀ · ◑
无监督异常检测系统 · Anomalib 2.3
```

### 4.2 切换逻辑（theme.js）

```
页面加载:
  1. 读 localStorage.getItem("theme")
  2. 如果有值 ("light"|"dark") → 设置 html.dataset.theme
  3. 如果没有 → 读 matchMedia("prefers-color-scheme: dark") → 设对应值
  4. 更新按钮图标（太阳/月亮）

点击切换:
  1. 读取当前 data-theme
  2. 切换为相反值
  3. localStorage.setItem("theme", 新值)
  4. 更新 html.dataset.theme
  5. 更新按钮图标

清除偏好（双击按钮）:
  1. localStorage.removeItem("theme")
  2. 恢复跟随系统 matchMedia
```

### 4.3 CSS 选择器

```css
/* 暗色（默认，无 data-theme 或 data-theme="dark"） */
:root { --bg-root: #000000; ... }

/* 亮色（手动切换或系统亮色） */
[data-theme="light"] { --bg-root: #f0f0f0; ... }
```

`@media (prefers-color-scheme: light)` 不再使用，改为 JS 初始化时读取。

### 4.4 按钮设计

- 两个圆形图标按钮并排：太阳 ◑（亮色）、月亮 ☽（暗色）
- 当前激活的图标高亮（`var(--accent)`），另一个灰色（`var(--text-tertiary)`）
- 过渡动画：`opacity 200ms ease-out`
- 尺寸：28×28px，间距 4px

---

## 5. V1 — Favicon

### 5.1 设计

32×32 SVG 内联，深色圆底 + 蓝色菱形：

```html
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">
  <circle cx="16" cy="16" r="16" fill="#1c1c1e"/>
  <polygon points="16,4 28,16 16,28 4,16" fill="#2997ff" opacity="0.9"/>
</svg>
```

亮色模式下圆底改为 `#e8e8ed`，菱形保持 `#2997ff`。

### 5.2 注入方式

`gr.HTML` 注入 `<link rel="icon">` 或在 `<head>` 中插入（Gradio 6 可能需通过 `js=` 参数操作 `document.head`）。

---

## 6. 文件变更清单

| 文件 | 操作 | 预估 |
|------|------|------|
| `modules/ui/theme.py` | 新建 | ~100 行 |
| `modules/ui/static/theme.js` | 新建 | ~70 行 |
| `modules/ui/demo.py` | 修改：注入 theme JS/CSS + toggle 按钮 + favicon | ~40 行 |
| `modules/ui/styles.css` | 修改：加 `[data-theme="light"]` 选择器，删除硬编码亮色块 | ~30 行 |
| `docs/superpowers/specs/2026-06-18-phase1-frontend-enhancement-design.md` | 本文件 | — |

---

## 7. 自审清单

- [x] 无 TBD、TODO
- [x] 模块边界清晰：theme.py → 数据 + 生成；theme.js → 交互；demo.py → 组装
- [x] 色板 keys 覆盖所有已用 CSS 变量
- [x] 与现有 Apple UI 设计规范一致
- [x] 降级策略：JS 禁用时回退到系统 `prefers-color-scheme`（styles.css 中保留 `@media` 兜底）
