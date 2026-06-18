/**
 * 主题切换交互逻辑 — Apple 风格亮/暗双模式
 *
 * 功能：
 * - localStorage 持久化用户偏好（key: "theme"）
 * - 系统 prefers-color-scheme 自动检测
 * - 点击太阳/月亮按钮即时切换
 * - 双击按钮清除偏好，恢复跟随系统
 * - 同步更新按钮高亮状态和 favicon
 */

(function () {
    'use strict';

    // ── 常量 ──────────────────────────────────────────────
    var THEME_KEY = 'theme';
    var THEME_LIGHT = 'light';
    var THEME_DARK = 'dark';

    // Favicon data URIs (与 theme.py 保持同步)
    var FAV_DARK = 'data:image/svg+xml,' + encodeURIComponent(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">' +
        '<circle cx="16" cy="16" r="16" fill="#1c1c1e"/>' +
        '<polygon points="16,4 28,16 16,28 4,16" fill="#2997ff" opacity="0.9"/>' +
        '</svg>'
    );
    var FAV_LIGHT = 'data:image/svg+xml,' + encodeURIComponent(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">' +
        '<circle cx="16" cy="16" r="16" fill="#e8e8ed"/>' +
        '<polygon points="16,4 28,16 16,28 4,16" fill="#2997ff" opacity="0.9"/>' +
        '</svg>'
    );

    // ── DOM 引用（延迟获取，确保 HTML 已注入）──────────────
    var htmlEl = document.documentElement;
    var faviconEl = null;
    var btnLight = null;
    var btnDark = null;

    // ── 系统主题检测 ──────────────────────────────────────
    var systemMedia = window.matchMedia('(prefers-color-scheme: dark)');

    function getSystemTheme() {
        return systemMedia.matches ? THEME_DARK : THEME_LIGHT;
    }

    // ── 应用主题 ──────────────────────────────────────────
    function applyTheme(theme) {
        htmlEl.setAttribute('data-theme', theme);

        // 同步 Gradio 的 .dark 类（Gradio 6 用 :root.dark / :root .dark 设暗色变量）
        // .dark 可能在 <html> 或 <body> 上，必须全部移除才能让亮色变量生效
        var darkElements = document.querySelectorAll('.dark');
        if (theme === THEME_LIGHT) {
            darkElements.forEach(function (el) { el.classList.remove('dark'); });
        } else {
            if (darkElements.length === 0) {
                document.body.classList.add('dark');
            }
        }

        updateButtons(theme);
        updateFavicon(theme);
    }

    // ── 更新按钮高亮 ──────────────────────────────────────
    function updateButtons(theme) {
        if (!btnLight || !btnDark) return;

        var accent = 'var(--accent)';
        var tertiary = 'var(--text-tertiary)';

        if (theme === THEME_LIGHT) {
            btnLight.style.color = accent;
            btnDark.style.color = tertiary;
        } else {
            btnLight.style.color = tertiary;
            btnDark.style.color = accent;
        }
    }

    // ── 更新 Favicon ──────────────────────────────────────
    function updateFavicon(theme) {
        if (!faviconEl) {
            faviconEl = document.getElementById('favicon');
        }
        if (!faviconEl) return;

        faviconEl.href = (theme === THEME_LIGHT) ? FAV_LIGHT : FAV_DARK;
    }

    // ── 获取当前主题 ──────────────────────────────────────
    function getCurrentTheme() {
        return htmlEl.getAttribute('data-theme') || THEME_DARK;
    }

    // ── 持久化并应用 ──────────────────────────────────────
    function setTheme(theme) {
        localStorage.setItem(THEME_KEY, theme);
        applyTheme(theme);
    }

    // ── 清除偏好，恢复跟随系统 ────────────────────────────
    function clearPreference() {
        localStorage.removeItem(THEME_KEY);
        applyTheme(getSystemTheme());
    }

    // ── 切换主题 ──────────────────────────────────────────
    function toggleTheme() {
        var current = getCurrentTheme();
        var next = (current === THEME_LIGHT) ? THEME_DARK : THEME_LIGHT;
        setTheme(next);
    }

    // ── 初始化 ────────────────────────────────────────────
    var retries = 0;
    var MAX_RETRIES = 20;  // 最多重试 20 次（约 2 秒）

    function init() {
        // 获取按钮引用（可能尚未渲染，Gradio 异步注入 HTML）
        btnLight = document.querySelector('.theme-btn-light');
        btnDark = document.querySelector('.theme-btn-dark');
        faviconEl = document.getElementById('favicon');

        // 如果按钮尚未渲染，延迟重试
        if (!btnLight || !btnDark) {
            if (retries < MAX_RETRIES) {
                retries++;
                setTimeout(init, 100);
                return;
            }
            // 超过最大重试次数，放弃绑定按钮事件
            console.warn('[theme] 主题按钮未找到，跳过事件绑定');
        }

        // 将 favicon 从 body 搬到 head（Gradio gr.HTML 注入在 body 中）
        if (faviconEl && faviconEl.parentElement !== document.head) {
            document.head.appendChild(faviconEl);
        }

        // 确定初始主题
        var saved = localStorage.getItem(THEME_KEY);
        if (saved === THEME_LIGHT || saved === THEME_DARK) {
            applyTheme(saved);
        } else {
            applyTheme(getSystemTheme());
        }

        // 绑定事件（仅首次成功时）
        if (retries === 0 || (btnLight && btnDark)) {
            bindEvents();
        }
    }

    // ── 事件绑定 ──────────────────────────────────────────
    function bindEvents() {
        if (btnLight) {
            btnLight.addEventListener('click', function (e) {
                e.preventDefault();
                setTheme(THEME_LIGHT);
            });
            btnLight.addEventListener('dblclick', function (e) {
                e.preventDefault();
                clearPreference();
            });
        }

        if (btnDark) {
            btnDark.addEventListener('click', function (e) {
                e.preventDefault();
                setTheme(THEME_DARK);
            });
            btnDark.addEventListener('dblclick', function (e) {
                e.preventDefault();
                clearPreference();
            });
        }

        // 监听系统主题变化（仅当无手动偏好时生效）
        systemMedia.addEventListener('change', function (e) {
            var saved = localStorage.getItem(THEME_KEY);
            if (!saved) {
                applyTheme(e.matches ? THEME_DARK : THEME_LIGHT);
            }
        });
    }

    // ── 启动 ──────────────────────────────────────────────
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function () {
            init();
            bindEvents();
        });
    } else {
        init();
        bindEvents();
    }
})();
