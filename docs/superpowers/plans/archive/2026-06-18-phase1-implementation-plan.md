# Phase 1 实现计划 — 主题模块 + 手动切换 + 图标

> 状态：设计已确认，等待实现 | 预估：2 个会话 | 依赖：无

---

## 概述

实现 A2（主题管理器）+ I1（手动亮暗切换按钮）+ V1（Favicon）。将硬编码的色板从 `styles.css` 抽离到 `theme.py`，添加 localStorage 持久化的主题切换功能，告别纯依赖 `@media` 的方式。

## 文件清单

| # | 文件 | 操作 | 说明 |
|---|------|------|------|
| 1 | `modules/ui/theme.py` | **新建** | 色板定义 + CSS 生成 + 切换按钮 HTML + Favicon |
| 2 | `modules/ui/static/theme.js` | **新建** | 主题切换交互逻辑（localStorage + data-theme） |
| 3 | `modules/ui/demo.py` | 修改 | 调用 theme.py 注入 CSS/JS/按钮/favicon |
| 4 | `modules/ui/styles.css` | 修改 | 用 `[data-theme="light"]` 替代 `@media` 亮色块 |

## 实现顺序

### 步骤 1：创建 `modules/ui/static/` 目录

```bash
mkdir -p modules/ui/static
```

### 步骤 2：创建 `modules/ui/theme.py`

核心模块，包含：

- `DARK` / `LIGHT` 两个色板字典（keys: bg_root, bg_system, bg_secondary, bg_tertiary, sep_subtle, sep_default, sep_strong, text, text_secondary, text_tertiary, shadow_sm, shadow_md, shadow_lg, shadow_glow）
- `build_css_variables(palette)` → 将字典编译为 `:root { --bg-root: ...; }` CSS 字符串
- `get_dark_css()` → 暗色默认 `:root` 块
- `get_light_css()` → 亮色 `html[data-theme="light"]` 块（注意：用 `html[data-theme="light"]` 而非 `[data-theme="light"]`，确保选择器优先级高于 `:root`）
- `get_theme_switch_html()` → 太阳 ☀ / 月亮 ◑ 两个圆形按钮的 HTML
- `get_theme_js()` → `<script>` 标签包裹的 theme.js
- `get_favicon_svg()` → 32×32 SVG 菱形图标

### 步骤 3：创建 `modules/ui/static/theme.js`

逻辑：
1. 页面加载时读 `localStorage.theme`，设置 `document.documentElement.dataset.theme`
2. 无存储值时 fallback 到 `matchMedia('(prefers-color-scheme: dark)')`
3. 监听系统主题变化事件 `matchMedia(...).addEventListener('change', ...)`
4. 点击按钮切换：更新 `dataset.theme` + `localStorage.theme`
5. 双击按钮清除偏好，恢复跟随系统
6. 更新按钮高亮状态

### 步骤 4：修改 `modules/ui/demo.py`

在 `create_interface()` 中：
1. 文件顶部 `import` theme 模块
2. `gr.Blocks()` 之前，添加 `gr.HTML` 注入：favicon + 暗色默认 CSS + 亮色 CSS（通过 `gr.HTML` 绕过 Gradio scoping）
3. 移除旧的 `<style>@media (prefers-color-scheme: light)</style>` 注入块（theme.py 替代）
4. 标题区 Markdown 内嵌入切换按钮 HTML（`theme.get_theme_switch_html()`）
5. `gr.HTML` 注入 `theme.get_theme_js()`
6. 在新的 `gr.HTML` 中注入亮色 CSS（通过 theme.py 生成，使用 `[data-theme="light"]` 选择器）
7. 在 `gr.HTML` 中注入 favicon SVG data URI
8. 文件底部 `main()` 函数最后调用 `copy_static_files()`（见步骤 5）

### 步骤 5：补充 Gradio 静态文件挂载

在 `demo.py` 底部 `main()` 中，`demo.launch()` 调用之后，打印静态文件路径提示。如果需要在 launch 时挂载静态目录，使用 `gr.Blocks()` 的 `static_dir` 参数或直接在 `launch()` 中使用 `allowed_paths`。

Gradio 6 静态文件方式：在 `gr.Blocks()` 中不直接支持 `static_dir`。替代方案：theme.js 通过 `gr.HTML('<script>...</script>')` 内联注入，不依赖外部文件加载。

### 步骤 6：修改 `modules/ui/styles.css`

1. 删除 `@media (prefers-color-scheme: light)` 块（已迁移到 theme.py 动态生成）
2. 删除 `body::before` 的亮色覆盖（迁移到 theme.py）
3. 在暗色默认变量后添加注释：亮色变量由 theme.py 生成
4. 可选：保留 `@media (prefers-color-scheme: light)` 作为 JS 禁用时的降级兜底，但用更简单的版本

## 验证清单

- [ ] `python modules/ui/theme.py` 能独立运行并打印 CSS
- [ ] `python scripts/run_ui.py` 启动无报错
- [ ] 页面加载后标题栏出现 ☀/◑ 切换按钮
- [ ] 点击按钮，页面即时切换亮/暗色
- [ ] 刷新页面，主题偏好保留
- [ ] 双击按钮，恢复跟随系统
- [ ] 浏览器标签页显示菱形图标
- [ ] `git status` 无意外文件
- [ ] 暗色模式 0 个白色元素
- [ ] 亮色模式 0 个 `#000` 纯黑元素（或极少）

## 设计规范参考

- Apple UI 设计规范：`docs/superpowers/specs/2026-06-18-apple-ui-design-spec.md`
- Phase 1 详细设计：`docs/superpowers/specs/2026-06-18-phase1-frontend-enhancement-design.md`
- 项目编码规范：`CLAUDE.md`（中文文档、导入顺序、命名规范）

## 注意事项

1. Gradio 6 的 CSS scoping 会破坏 `@media` 内的 `:root` 选择器——所有需要通过 `@media` 动态切换的 CSS 变量必须通过 `gr.HTML` 注入 `<style>` 标签
2. `data-theme` 属性设在 `<html>` 上（`document.documentElement`），不是 `<body>`
3. 切换按钮不要用 emoji（不同平台渲染不一致），用 SVG
4. favicon 用 SVG data URI（`data:image/svg+xml,...`），避免产生额外 HTTP 请求
5. `theme.js` 需要放在 `gr.HTML` 中注入到页面底部（在 `</body>` 前），确保 DOM 就绪后执行
